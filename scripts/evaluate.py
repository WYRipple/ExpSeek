# evaluate.py

import os
import json
import time
import argparse
from tqdm import tqdm
from multiprocessing import Pool, Lock
from openai import OpenAI
from functools import partial

JUDGE_PROMPT = """You are an evaluation assistant. Please determine if the predicted answer is equivalent to the labeled answer.
As long as the Labeled Answer is contained within the Predicted Answer, it is considered correct, even if additional, more detailed explanations are included.
You should focus on whether the Predicted Answer truly answers the question correctly.

# Question: 
{question}

# Labeled Answer: 
{correct_answer}

# Predicted Answer: 
{response}

Respond **only with** "Correct" or "Incorrect", no other tokens.
"""


# ── I/O helpers ─────────────────────────────────────────────────────────────

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def append_to_jsonl(file_path, data_dict):
    """Thread/process-safe append using file lock."""
    json_str = json.dumps(data_dict, ensure_ascii=False) + "\n"
    with open(file_path, "a", encoding="utf-8") as f:
        import fcntl
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json_str)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def resume_todo_list(save_path, todo_list):
    """Filter out already-evaluated questions for resumable evaluation."""
    if not os.path.exists(save_path):
        return todo_list
    saved_data = load_jsonl(save_path)
    finished_q = {item["question"] for item in saved_data if item.get("question")}
    return [item for item in todo_list if item.get("question") not in finished_q]


def detect_rollout_count(input_dir):
    """Auto-detect how many iter*.jsonl files exist in input_dir."""
    count = 0
    while os.path.exists(os.path.join(input_dir, f"iter{count + 1}.jsonl")):
        count += 1
    if count == 0:
        raise FileNotFoundError(f"No iter*.jsonl files found in: {input_dir}")
    return count


# ── LLM judge ───────────────────────────────────────────────────────────────

def call_judge(msgs, api_key, api_base, model, max_tries=20):
    for attempt in range(max_tries):
        try:
            client = OpenAI(api_key=api_key, base_url=api_base)
            resp = client.chat.completions.create(
                model=model,
                messages=msgs,
                max_tokens=10,
                temperature=0.0,
            )
            content = resp.choices[0].message.content.strip()
            if "Correct" in content or "Incorrect" in content:
                return content
            raise ValueError(f"Unexpected response: {content}")
        except Exception as e:
            print(f"[JudgeWarn] Attempt {attempt + 1}/{max_tries}: {e}")
            time.sleep(2)
    print("[JudgeError] Max retries reached, returning [ERROR].")
    return "[ERROR]"


# ── Per-sample worker ────────────────────────────────────────────────────────

def eval_one(args):
    sample, output_eval, api_key, api_base, model = args

    if sample.get("prediction") == "[Failed]":
        sample["eval_result"] = "Incorrect"
        append_to_jsonl(output_eval, sample)
        return

    prompt = JUDGE_PROMPT.format(
        question=sample["question"],
        correct_answer=sample["answer"],
        response=sample["prediction"],
    )
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    result = call_judge(msgs, api_key, api_base, model)
    sample["eval_result_context"] = result
    sample["eval_result"] = "Correct" if "Correct" in result else "Incorrect"
    append_to_jsonl(output_eval, sample)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate inference outputs using LLM judge.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the inference output directory (contains iter*.jsonl files).")
    parser.add_argument("--api_key", type=str, required=True,
                        help="API key for the judge model.")
    parser.add_argument("--api_base", type=str,
                        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
                        help="API base URL for the judge model.")
    parser.add_argument("--judge_model", type=str,
                        default="qwen3-235b-a22b-instruct-2507",
                        help="Judge model name.")
    parser.add_argument("--num_workers", type=int, default=25,
                        help="Number of parallel worker processes.")
    parser.add_argument("--debug", action="store_true",
                        help="Run in debug mode (serial execution).")
    args = parser.parse_args()

    # ── Resolve paths ────────────────────────────────────────────────────────
    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"input_dir does not exist: {input_dir}")

    # Output goes into input_dir/eval_results/
    output_dir = os.path.join(input_dir, "eval_results")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Input  dir : {input_dir}")
    print(f"[INFO] Output dir : {output_dir}")

    # ── Auto-detect rollout count ────────────────────────────────────────────
    total_K = detect_rollout_count(input_dir)
    print(f"[INFO] Detected {total_K} rollout(s).")

    # ── Evaluate each rollout ────────────────────────────────────────────────
    for k in range(1, total_K + 1):
        data_path = os.path.join(input_dir, f"iter{k}.jsonl")
        output_eval = os.path.join(output_dir, f"eval_round{k}.jsonl")

        data_list = load_jsonl(data_path)
        print(f"[Round {k}] Loaded {len(data_list)} samples from {data_path}")

        todo_list = resume_todo_list(output_eval, data_list)
        print(f"[Round {k}] Remaining: {len(todo_list)} (skipping {len(data_list) - len(todo_list)} already done)")

        if not todo_list:
            print(f"[Round {k}] All done, skipping.")
            continue

        tasks = [
            (sample, output_eval, args.api_key, args.api_base, args.judge_model)
            for sample in todo_list
        ]

        if args.debug:
            print(f"[Round {k}] Debug mode: serial execution.")
            for task in tqdm(tasks, desc=f"Evaluating Round {k}"):
                eval_one(task)
        else:
            with Pool(processes=args.num_workers) as pool:
                for _ in tqdm(
                    pool.imap(eval_one, tasks),
                    total=len(tasks),
                    desc=f"Evaluating Round {k}"
                ):
                    pass

    print("[INFO] Evaluation complete.")


if __name__ == "__main__":
    main()
