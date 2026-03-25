# step3_label_topic.py
# Input : {exp_kb_dir}/pair-EXP.jsonl
# Output: {exp_kb_dir}/EXP-KB-process-label.jsonl
#         {exp_kb_dir}/EXP-KB-final-label.jsonl

import os
import json
import argparse
import ast
import shutil
import re
from tqdm import tqdm
from openai import OpenAI
import time


# ── Prompts ───────────────────────────────────────────────────────────────────

TOPIC_PROMPT = """
A teacher is analyzing each step taken by students when solving complex problems. I will give you several "behavior + mistake" items that the teacher has summarized for students at certain steps, defined as:
```
behavior: A relatively general description introducing what the student saw and then did. The description does not involve error attribution, focuses on objectively stating the student's behavior, and does not evaluate whether the behavior is good/bad or right/wrong 
mistake: The student is satisfied with preliminary information (such as "Deputy Party Secretary") and fails to realize the need to cross-reference multiple search results to extract a complete and accurate answer, particularly overlooking that the specific name "xx xx xx" has already appeared in the first entry of the third set of search results.
```
# Overall Overview 
Your goal is to give **each behavior + mistake** a **scenario-narrative label**. A label's description should be concise enough to clearly express the characteristics of the behavior and be reusable. 
For each given new behavior + mistake, you can choose one of the following three actions: 
1. Reuse: Do not change any current labels, and select an existing label for the new behavior (recognizing the existing classification) 
2. Create: Do not change existing labels, create a new label for the new behavior (existing classification is incomplete) 
3. Modify: Modify certain current labels, and assign that label to the new behavior (existing classification is inaccurate) 

# Detailed Requirements 
1. Each label must be concise and clear, but needs to have certain semantic information that allows people to understand the characteristics of the current behavior + mistake without explanation. It should be at least a dozen or dozens of words (e.g., in the pattern of xxx: xxx xxx xx). 
2. There cannot be too many labels; each label should have distinguishability in scenario content. 
3. One label can correspond to multiple behaviors, so you must ensure their textual content is consistent. 
4. Use the given id as the unique identifier for behaviors. When outputting, you need to output the ids and labels of all existing behaviors and new behaviors. 
5. Try to keep the number of different labels balanced. 

# List of Behaviors Already Given Labels 
{exp_list} 

# List of New Behaviors 
{new_exp_list} 

# Output Format:
```
[
    {{
        'analysis': 'Write here the fine-grained analysis of all current states, the design and determination of the current state set, and the complete thinking process of how to select appropriate behaviors for each current state.',
    }}
    {{
        'id': 1,
        'label': 'Fill in the label determined after analysis here.',
    }},
    {{
        'id': 2,
        'label': 'Fill in the label determined after analysis here.',
    }},
    {{
        'id': 3,
        'label': 'Fill in the label determined after analysis here.',
    }},
    ...
]
```
"""


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(path, data_list):
    with open(path, "w", encoding="utf-8") as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ── Extract experiences from pair-EXP ────────────────────────────────────────

def extract_experiences(pair_exp_list):
    process_exp = []
    final_exp   = []
    pro_id = 1
    fin_id = 1

    for sample in pair_exp_list:
        step_num = len(sample["token_entropy_avg_list"][0])
        try:
            for step_exp in sample["dict_exp"][0]["STEP-EXP"]:
                if step_exp["step-exp"] == []:
                    continue
                record = {
                    "exp_id":   None,
                    "behavior": step_exp["step-exp"][0],
                    "mistake":  step_exp["step-exp"][1],
                    "guidance": step_exp["step-exp"][2],
                }
                if step_exp["step-id"] != step_num:
                    record["exp_id"] = pro_id
                    process_exp.append(record)
                    pro_id += 1
                else:
                    record["exp_id"] = fin_id
                    final_exp.append(record)
                    fin_id += 1
        except Exception as e:
            print(f"[Step3] Warning: failed to extract from sample "
                  f"'{sample.get('question','')[:50]}': {e}")
            continue

    print(f"[Step3] Extracted process experiences: {len(process_exp)}")
    print(f"[Step3] Extracted final experiences  : {len(final_exp)}")
    return process_exp, final_exp


# ── Batch file helpers ────────────────────────────────────────────────────────

def get_batch_dir(output_path):
    dir_name  = os.path.dirname(output_path)
    base_name = os.path.basename(output_path)
    name_stem = base_name[:-6] if base_name.endswith(".jsonl") else base_name
    return os.path.join(dir_name, f"{name_stem}-batches")


def get_batch_path(output_path, batch_num):
    batch_dir = get_batch_dir(output_path)
    os.makedirs(batch_dir, exist_ok=True)
    return os.path.join(batch_dir, f"batch{batch_num}.jsonl")


def find_latest_batch(output_path):
    batch_dir = get_batch_dir(output_path)
    if not os.path.exists(batch_dir):
        return 0, None
    batch_files = []
    for fname in os.listdir(batch_dir):
        m = re.match(r"batch(\d+)\.jsonl", fname)
        if m:
            batch_files.append((int(m.group(1)),
                                os.path.join(batch_dir, fname)))
    if not batch_files:
        return 0, None
    batch_files.sort(key=lambda x: x[0])
    return batch_files[-1]


# ── API call ──────────────────────────────────────────────────────────────────

def call_server(msgs, api_key, api_base, model, max_tries=20):
    for attempt in range(max_tries):
        try:
            client = OpenAI(api_key=api_key, base_url=api_base)
            resp = client.chat.completions.create(
                model=model,
                messages=msgs,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            time.sleep(2)
            print(f"[Step3] API error (attempt {attempt+1}/{max_tries}): {e}")
    return "[ERROR]"


# ── Label one split ───────────────────────────────────────────────────────────

def label_experiences(data_list, output_path, batch_size, api_key, api_base, model):
    # Resume
    latest_batch_num, latest_batch_file = find_latest_batch(output_path)

    if latest_batch_file and os.path.exists(latest_batch_file):
        labeled_list  = load_jsonl(latest_batch_file)
        processed_ids = {item["exp_id"] for item in labeled_list}
        remaining     = [item for item in data_list
                         if item["exp_id"] not in processed_ids]
        print(f"[Step3] Resuming from batch {latest_batch_num}, "
              f"{len(labeled_list)} done, {len(remaining)} remaining.")
    else:
        labeled_list     = []
        remaining        = data_list
        latest_batch_num = 0
        print(f"[Step3] Starting fresh, {len(remaining)} experiences to label.")

    if not remaining:
        print("[Step3] All experiences already labeled.")
        if latest_batch_file and latest_batch_file != output_path:
            shutil.copy(latest_batch_file, output_path)
        return

    total_batches     = (len(remaining) + batch_size - 1) // batch_size
    current_batch_num = latest_batch_num

    for batch_idx in tqdm(range(0, len(remaining), batch_size),
                          total=total_batches,
                          desc="[Step3] Labeling batches"):

        batch = remaining[batch_idx: batch_idx + batch_size]
        current_batch_num += 1

        exp_list = [
            {"id": item["exp_id"], "behavior": item["behavior"],
             "mistake": item["mistake"], "label": item["label"]}
            for item in labeled_list
        ]
        new_exp_list = [
            {"id": item["exp_id"], "behavior": item["behavior"],
             "mistake": item["mistake"]}
            for item in batch
        ]

        exp_list_str     = json.dumps(exp_list,     ensure_ascii=False, indent=2)
        new_exp_list_str = json.dumps(new_exp_list, ensure_ascii=False, indent=2)
        prompt = TOPIC_PROMPT.format(
            exp_list=exp_list_str,
            new_exp_list=new_exp_list_str,
        )
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": prompt},
        ]

        id_to_label = None
        for attempt in range(10):
            try:
                result = call_server(msgs, api_key, api_base, model)
                if result == "[ERROR]":
                    raise ValueError("API returned [ERROR]")
                parsed      = ast.literal_eval(result.strip("`").strip())
                parsed      = parsed[1:]   # skip analysis dict
                id_to_label = {item["id"]: item["label"] for item in parsed}
                break
            except Exception as e:
                print(f"[Step3] Parse error attempt {attempt+1}/10: {e}")
                continue

        if id_to_label is None:
            print(f"[Step3] Skipping batch {current_batch_num} after 10 failures.")
            continue

        # Update existing labels (may have been revised)
        for item in labeled_list:
            if item["exp_id"] in id_to_label:
                item["label"] = id_to_label[item["exp_id"]]

        # Append new samples
        for item in batch:
            new_item = item.copy()
            new_item["label"] = id_to_label.get(item["exp_id"], "unknown")
            labeled_list.append(new_item)

        # Save batch checkpoint
        batch_path = get_batch_path(output_path, current_batch_num)
        save_jsonl(batch_path, labeled_list)
        print(f"[Step3] Batch {current_batch_num} saved: {batch_path} "
              f"({len(labeled_list)} total)")

    # Save final output
    save_jsonl(output_path, labeled_list)
    print(f"[Step3] Final output saved: {output_path} "
          f"({len(labeled_list)} experiences)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step3: Label topic categories for experiences via LLM."
    )
    parser.add_argument("--exp_kb_dir",  type=str, required=True,
                        help="Experience KB directory (contains pair-EXP.jsonl).")
    parser.add_argument("--api_key",     type=str, required=True)
    parser.add_argument("--api_base",    type=str, required=True)
    parser.add_argument("--model",       type=str, required=True)
    parser.add_argument("--batch_size",  type=int, default=20)
    args = parser.parse_args()

    input_path     = os.path.join(args.exp_kb_dir, "pair-EXP.jsonl")
    output_process = os.path.join(args.exp_kb_dir, "EXP-KB-process-label.jsonl")
    output_final   = os.path.join(args.exp_kb_dir, "EXP-KB-final-label.jsonl")

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"pair-EXP.jsonl not found in {args.exp_kb_dir}. "
            "Please run step2 first."
        )

    pair_exp_list = load_jsonl(input_path)
    process_exp, final_exp = extract_experiences(pair_exp_list)

    print(f"\n[Step3] ── Labeling process experiences ──────────────")
    label_experiences(
        process_exp, output_process, args.batch_size,
        args.api_key, args.api_base, args.model,
    )

    print(f"\n[Step3] ── Labeling final experiences ────────────────")
    label_experiences(
        final_exp, output_final, args.batch_size,
        args.api_key, args.api_base, args.model,
    )


if __name__ == "__main__":
    main()

