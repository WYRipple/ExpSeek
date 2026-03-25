# step1_aggregate.py
# Input : eval_results/eval_round*.jsonl  (from evaluate.py)
# Output: {exp_kb_dir}/pair.jsonl

import os
import json
import argparse
from collections import defaultdict
from itertools import cycle
from tqdm import tqdm


# ── I/O helpers ──────────────────────────────────────────────────────────────

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


def detect_rollout_count(eval_dir):
    count = 0
    while os.path.exists(os.path.join(eval_dir, f"eval_round{count + 1}.jsonl")):
        count += 1
    if count == 0:
        raise FileNotFoundError(f"No eval_round*.jsonl found in: {eval_dir}")
    return count


# ── Entropy extraction ───────────────────────────────────────────────────────

def extract_entropy_avg_list(messages):
    """
    Extract token_entropy_avg from each assistant turn in order.
    Returns a list of floats, one per assistant turn.
    """
    return [
        msg.get("token_entropy_avg", 0.0)
        for msg in messages
        if msg.get("role") == "assistant"
    ]


# ── Merge rollouts ───────────────────────────────────────────────────────────

def merge_rollouts(eval_dir):
    """
    Read all eval_round*.jsonl and group by question.
    Returns a dict: {question -> aggregated item}.
    """
    total_K = detect_rollout_count(eval_dir)
    print(f"[Step1] Detected {total_K} rollout(s) in {eval_dir}")

    merged = defaultdict(lambda: {
        "question": None,
        "answer": None,
        "raw_item": None,
        "messages_list": [],
        "prediction_list": [],
        "termination_list": [],
        "eval_result_list": [],
        "token_entropy_avg_list": [],
    })

    for k in range(1, total_K + 1):
        path = os.path.join(eval_dir, f"eval_round{k}.jsonl")
        if not os.path.exists(path):
            print(f"[Step1] Warning: {path} not found, skipping.")
            continue

        for item in load_jsonl(path):
            q = item.get("question", "").strip()
            if not q:
                continue

            if merged[q]["question"] is None:
                merged[q]["question"] = item["question"]
                merged[q]["answer"]   = item.get("answer")
                merged[q]["raw_item"] = item.get("raw_item")

            messages = item.get("messages", [])
            merged[q]["messages_list"].append(messages)
            merged[q]["prediction_list"].append(item.get("prediction", ""))
            merged[q]["termination_list"].append(item.get("termination", ""))
            merged[q]["eval_result_list"].append(item.get("eval_result", "Incorrect"))
            merged[q]["token_entropy_avg_list"].append(extract_entropy_avg_list(messages))

        print(f"[Step1] Loaded eval_round{k}.jsonl")

    return merged


# ── Classify ─────────────────────────────────────────────────────────────────

def classify(merged):
    """Split merged data into all-correct / all-wrong / mixed buckets."""
    true_list, false_list, mix_list = [], [], []

    for q, data in merged.items():
        results = data["eval_result_list"]
        correct = results.count("Correct")
        total   = len(results)

        item = {k: v for k, v in data.items()}  # shallow copy

        if correct == total:
            true_list.append(item)
        elif correct == 0:
            false_list.append(item)
        else:
            mix_list.append(item)

    return true_list, false_list, mix_list


# ── Create pairs ─────────────────────────────────────────────────────────────

def create_pairs(mix_list):
    """
    For each incorrect trajectory in mix, pair it with a correct one.
    Skip conditions:
      - prediction == '[Failed]'
      - token_entropy_avg_list is empty
    Index 0 = incorrect, index 1 = correct.
    """
    pair_list = []
    skipped_failed  = 0
    skipped_entropy = 0

    for item in tqdm(mix_list, desc="[Step1] Creating pairs"):
        results       = item["eval_result_list"]
        incorrect_idxs = [i for i, r in enumerate(results) if r == "Incorrect"]
        correct_idxs   = [i for i, r in enumerate(results) if r == "Correct"]

        if not incorrect_idxs or not correct_idxs:
            continue

        correct_cycle = cycle(correct_idxs)

        for inc_idx in incorrect_idxs:
            pred         = item["prediction_list"][inc_idx]
            entropy_list = item["token_entropy_avg_list"][inc_idx]

            if pred == "[Failed]":
                skipped_failed += 1
                continue
            if not entropy_list:
                skipped_entropy += 1
                continue

            cor_idx = next(correct_cycle)
            pair_list.append({
                "question": item["question"],
                "answer":   item["answer"],
                "raw_item": item["raw_item"],
                "messages_list":          [item["messages_list"][inc_idx],
                                           item["messages_list"][cor_idx]],
                "prediction_list":        [item["prediction_list"][inc_idx],
                                           item["prediction_list"][cor_idx]],
                "termination_list":       [item["termination_list"][inc_idx],
                                           item["termination_list"][cor_idx]],
                "eval_result_list":       ["Incorrect", "Correct"],
                "token_entropy_avg_list": [item["token_entropy_avg_list"][inc_idx],
                                           item["token_entropy_avg_list"][cor_idx]],
            })

    print(f"[Step1] Pairs created    : {len(pair_list)}")
    print(f"[Step1] Skipped [Failed] : {skipped_failed}")
    print(f"[Step1] Skipped no-entropy: {skipped_entropy}")
    return pair_list


# ── Stats report ─────────────────────────────────────────────────────────────

def print_stats(true_list, false_list, mix_list, pair_list):
    total = len(true_list) + len(false_list) + len(mix_list)
    print(f"\n[Step1] ── Classification Report ──────────────────────")
    print(f"  Total questions : {total}")
    print(f"  All correct     : {len(true_list):4d}  ({len(true_list)/total*100:.1f}%)")
    print(f"  All wrong       : {len(false_list):4d}  ({len(false_list)/total*100:.1f}%)")
    print(f"  Mixed           : {len(mix_list):4d}  ({len(mix_list)/total*100:.1f}%)")
    print(f"  Pairs generated : {len(pair_list)}")
    print(f"────────────────────────────────────────────────────────")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step1: Aggregate rollouts and create incorrect/correct pairs."
    )
    parser.add_argument("--eval_dir", type=str, required=True,
                        help="Directory containing eval_round*.jsonl "
                             "(e.g. outputs/xxx/eval_results).")
    parser.add_argument("--exp_kb_dir", type=str, required=True,
                        help="Output directory for experience knowledge base "
                             "(e.g. experience_base/demo-Qwen3-8B).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run even if pair.jsonl already exists.")
    args = parser.parse_args()

    output_path = os.path.join(args.exp_kb_dir, "pair.jsonl")

    # Skip if already done
    if os.path.exists(output_path) and not args.overwrite:
        print(f"[Step1] pair.jsonl already exists, skipping. "
              f"Use --overwrite to re-run.")
        return

    os.makedirs(args.exp_kb_dir, exist_ok=True)

    merged                          = merge_rollouts(args.eval_dir)
    true_list, false_list, mix_list = classify(merged)
    pair_list                       = create_pairs(mix_list)

    print_stats(true_list, false_list, mix_list, pair_list)
    save_jsonl(output_path, pair_list)
    print(f"[Step1] Saved: {output_path}")


if __name__ == "__main__":
    main()
