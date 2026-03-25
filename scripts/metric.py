# metric.py

import os
import json
import argparse
from collections import defaultdict
from transformers import AutoTokenizer


# ── Token / Step helpers ─────────────────────────────────────────────────────

def count_tokens(messages, tokenizer):
    """Count total tokens in a message list using the local tokenizer."""
    cleaned = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages
    ]
    tokens = tokenizer.apply_chat_template(
        cleaned,
        tokenize=True,
        add_generation_prompt=False
    )
    return len(tokens)


def count_steps(messages):
    """Count number of assistant turns in a conversation."""
    return sum(1 for msg in messages if msg.get("role") == "assistant")


# ── Per-file loaders ─────────────────────────────────────────────────────────

def load_acc_of_file(path):
    total, correct = 0, 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            total += 1
            if obj.get("eval_result") == "Correct":
                correct += 1
    print(f"  [INFO] Total samples: {total}")
    return correct / total if total > 0 else 0.0


def load_results_by_sample(path):
    """Return {question: is_correct} dict for pass@k computation."""
    results = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            question = obj.get("question")
            if question is None:
                print(f"  [WARN] Entry missing 'question' field in {path}")
                continue
            results[question] = obj.get("eval_result") == "Correct"
    return results


def compute_stats_from_file(path, tokenizer):
    """
    Compute token and step statistics from a single eval file.
    Returns separate stats for successful samples and all samples.
    """
    success_tokens, success_steps, success_count = 0, 0, 0
    all_tokens, all_steps, all_count = 0, 0, 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            prediction = obj.get("prediction", "")
            messages = obj.get("messages", [])
            if not messages:
                continue

            all_count += 1
            steps = count_steps(messages)
            all_steps += steps

            try:
                tokens = count_tokens(messages, tokenizer)
                all_tokens += tokens
            except Exception as e:
                print(f"  [WARN] Token count failed: {e}")
                tokens = 0

            if prediction != "[Failed]":
                success_count += 1
                success_steps += steps
                success_tokens += tokens

    return {
        "success_count": success_count,
        "success_avg_tokens": success_tokens / success_count if success_count > 0 else 0,
        "success_avg_steps": success_steps / success_count if success_count > 0 else 0,
        "all_count": all_count,
        "all_avg_tokens": all_tokens / all_count if all_count > 0 else 0,
        "all_avg_steps": all_steps / all_count if all_count > 0 else 0,
    }


def compute_time_stats_from_file(path):
    """Compute elapsed_time statistics from a single eval file."""
    time_values = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            t = obj.get("elapsed_time")
            if t is not None:
                try:
                    time_values.append(float(t))
                except (ValueError, TypeError):
                    print(f"  [WARN] Invalid elapsed_time: {t}")

    if not time_values:
        return None

    sorted_t = sorted(time_values)
    n = len(sorted_t)
    median = (sorted_t[n // 2 - 1] + sorted_t[n // 2]) / 2 if n % 2 == 0 else sorted_t[n // 2]

    return {
        "count": n,
        "total_time": sum(time_values),
        "avg_time": sum(time_values) / n,
        "median_time": median,
        "max_time": max(time_values),
        "min_time": min(time_values),
        "time_values": time_values,
    }


def compute_guide_tag_stats_from_file(path):
    """Compute guide_tag distribution statistics from a single eval file."""
    tag_counts = defaultdict(int)
    total_assistants = 0
    total_samples = 0
    has_any = False

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            total_samples += 1
            for msg in obj.get("messages", []):
                if msg.get("role") == "assistant":
                    total_assistants += 1
                    if "guide_tag" in msg:
                        has_any = True
                        tag_counts[msg["guide_tag"]] += 1

    if not has_any:
        return None

    return {
        "tag_counts": dict(tag_counts),
        "total_assistants": total_assistants,
        "total_samples": total_samples,
    }


# ── Multi-file aggregators ───────────────────────────────────────────────────

def aggregate_guide_tag_stats(stats_list):
    tag_types = [1, 2, "1or2", 3, 4, 5]
    results = {}
    for tag_type in tag_types:
        ratios, per_sample = [], []
        for stats in stats_list:
            tag_counts = stats["tag_counts"]
            count = (
                tag_counts.get(1, 0) + tag_counts.get(2, 0)
                if tag_type == "1or2"
                else tag_counts.get(tag_type, 0)
            )
            ratios.append(count / stats["total_assistants"] if stats["total_assistants"] > 0 else 0)
            per_sample.append(count / stats["total_samples"] if stats["total_samples"] > 0 else 0)
        results[tag_type] = {
            "avg_ratio": sum(ratios) / len(ratios),
            "avg_per_sample": sum(per_sample) / len(per_sample),
        }
    return results


def compute_pass_at_k(sample_results_list):
    all_questions = set().union(*[r.keys() for r in sample_results_list])
    passed = sum(
        any(r.get(q, False) for r in sample_results_list)
        for q in all_questions
    )
    return passed / len(all_questions) if all_questions else 0.0


def compute_consistency_stats(sample_results_list):
    all_questions = set().union(*[r.keys() for r in sample_results_list])
    all_correct = all_wrong = mixed = 0
    for q in all_questions:
        correct_count = sum(r.get(q, False) for r in sample_results_list)
        total = len(sample_results_list)
        if correct_count == total:
            all_correct += 1
        elif correct_count == 0:
            all_wrong += 1
        else:
            mixed += 1
    total_q = len(all_questions)
    return {
        "total_samples": total_q,
        "all_correct": all_correct,
        "all_correct_ratio": all_correct / total_q if total_q > 0 else 0,
        "all_wrong": all_wrong,
        "all_wrong_ratio": all_wrong / total_q if total_q > 0 else 0,
        "mixed": mixed,
        "mixed_ratio": mixed / total_q if total_q > 0 else 0,
    }


def detect_rollout_count(eval_dir):
    """Auto-detect how many eval_round*.jsonl files exist."""
    count = 0
    while os.path.exists(os.path.join(eval_dir, f"eval_round{count + 1}.jsonl")):
        count += 1
    if count == 0:
        raise FileNotFoundError(f"No eval_round*.jsonl files found in: {eval_dir}")
    return count


# ── Main evaluate function ───────────────────────────────────────────────────

def evaluate(eval_dir, tokenizer, save_path):
    """
    Compute and save all metrics from eval_round*.jsonl files.

    Args:
        eval_dir:  directory containing eval_round*.jsonl (= input_dir/eval_results)
        tokenizer: loaded HuggingFace tokenizer
        save_path: where to write the .txt result summary
    """
    total_K = detect_rollout_count(eval_dir)
    print(f"[INFO] Detected {total_K} eval round(s) in {eval_dir}")

    acc_list = []
    sample_results_list = []
    stats_list = []
    guide_tag_stats_list = []
    time_stats_list = []

    for k in range(1, total_K + 1):
        path = os.path.join(eval_dir, f"eval_round{k}.jsonl")
        print(f"[Round {k}] Loading {path}")

        acc = load_acc_of_file(path)
        acc_list.append((k, acc))
        sample_results_list.append(load_results_by_sample(path))

        stats = compute_stats_from_file(path, tokenizer)
        stats_list.append((k, stats))

        guide_stats = compute_guide_tag_stats_from_file(path)
        if guide_stats is not None:
            guide_tag_stats_list.append(guide_stats)

        time_stats = compute_time_stats_from_file(path)
        if time_stats is not None:
            time_stats_list.append((k, time_stats))

    # ── Aggregated stats ─────────────────────────────────────────────────────
    overall_mean_acc = sum(a for _, a in acc_list) / len(acc_list)

    total_success = sum(s["success_count"] for _, s in stats_list)
    total_all = sum(s["all_count"] for _, s in stats_list)

    overall_success_avg_tokens = (
        sum(s["success_avg_tokens"] * s["success_count"] for _, s in stats_list) / total_success
        if total_success > 0 else 0
    )
    overall_success_avg_steps = (
        sum(s["success_avg_steps"] * s["success_count"] for _, s in stats_list) / total_success
        if total_success > 0 else 0
    )
    overall_all_avg_tokens = (
        sum(s["all_avg_tokens"] * s["all_count"] for _, s in stats_list) / total_all
        if total_all > 0 else 0
    )
    overall_all_avg_steps = (
        sum(s["all_avg_steps"] * s["all_count"] for _, s in stats_list) / total_all
        if total_all > 0 else 0
    )

    # Time
    overall_time_stats = None
    if time_stats_list:
        all_tv = []
        for _, ts in time_stats_list:
            all_tv.extend(ts["time_values"])
        n = len(all_tv)
        sorted_tv = sorted(all_tv)
        median = (sorted_tv[n // 2 - 1] + sorted_tv[n // 2]) / 2 if n % 2 == 0 else sorted_tv[n // 2]
        overall_time_stats = {
            "count": n,
            "total_time": sum(all_tv),
            "avg_time": sum(all_tv) / n,
            "median_time": median,
            "max_time": max(all_tv),
            "min_time": min(all_tv),
        }

    # Pass@K
    pass_at_k_list = [
        (k, compute_pass_at_k(sample_results_list[:k]))
        for k in range(1, len(sample_results_list) + 1)
    ]

    consistency_stats = compute_consistency_stats(sample_results_list)
    guide_tag_aggregated = aggregate_guide_tag_stats(guide_tag_stats_list) if guide_tag_stats_list else None

    # ── Build output text ────────────────────────────────────────────────────
    lines = []
    sep = "=" * 80
    thin = "-" * 40

    lines += [sep, f"Evaluation Results", f"Eval dir : {eval_dir}", f"Rounds   : {total_K}", sep, ""]

    lines += ["Individual Round Statistics:", thin]
    time_dict = {k: ts for k, ts in time_stats_list}
    for (k, acc), (_, stats) in zip(acc_list, stats_list):
        lines += [
            f"  K={k}:",
            f"    Accuracy              : {acc:.4f} ({acc:.2%})",
            f"    Success Samples       : {stats['success_count']}",
            f"    All Samples           : {stats['all_count']}",
            f"    Avg Tokens (Success)  : {stats['success_avg_tokens']:.2f}",
            f"    Avg Tokens (All)      : {stats['all_avg_tokens']:.2f}",
            f"    Avg Steps  (Success)  : {stats['success_avg_steps']:.2f}",
            f"    Avg Steps  (All)      : {stats['all_avg_steps']:.2f}",
        ]
        if k in time_dict:
            ts = time_dict[k]
            lines += [
                f"    Avg Time              : {ts['avg_time']:.2f}s",
                f"    Median Time           : {ts['median_time']:.2f}s",
                f"    Time Range            : [{ts['min_time']:.2f}s, {ts['max_time']:.2f}s]",
                f"    Total Time            : {ts['total_time']:.2f}s",
            ]
    lines.append("")

    lines += ["Overall Statistics:", thin]
    lines += [
        f"  Mean Accuracy              : {overall_mean_acc:.4f} ({overall_mean_acc:.2%})",
        f"  Total Success Samples      : {total_success}",
        f"  Total All Samples          : {total_all}",
        f"  Overall Avg Tokens (Success): {overall_success_avg_tokens:.2f}",
        f"  Overall Avg Tokens (All)   : {overall_all_avg_tokens:.2f}",
        f"  Overall Avg Steps  (Success): {overall_success_avg_steps:.2f}",
        f"  Overall Avg Steps  (All)   : {overall_all_avg_steps:.2f}",
        "",
    ]

    if overall_time_stats:
        lines += ["Time Statistics:", thin]
        lines += [
            f"  Total Samples with Time   : {overall_time_stats['count']}",
            f"  Avg Time per Sample       : {overall_time_stats['avg_time']:.2f}s",
            f"  Median Time per Sample    : {overall_time_stats['median_time']:.2f}s",
            f"  Time Range                : [{overall_time_stats['min_time']:.2f}s, {overall_time_stats['max_time']:.2f}s]",
            f"  Total Time (All Rounds)   : {overall_time_stats['total_time']:.2f}s"
            f" ({overall_time_stats['total_time'] / 3600:.2f}h)",
            "",
        ]
    else:
        lines += ["Time Statistics:", thin, "  No elapsed_time field found.", ""]

    if guide_tag_aggregated:
        lines += [
            "Guide Tag Statistics (averaged across all rounds):",
            thin,
            f"  Files with guide_tag: {len(guide_tag_stats_list)}/{len(acc_list)}",
            "",
        ]
        tag_names = {
            1:       "Tag=1 (Process Guidance)",
            2:       "Tag=2 (Answer Guidance)",
            "1or2":  "Tag=1 or 2 (Any Guidance)",
            3:       "Tag=3 (Token Limit)",
            4:       "Tag=4 (Content Safety)",
            5:       "Tag=5 (Format Error)",
        }
        for tag_type in [1, 2, "1or2", 3, 4, 5]:
            s = guide_tag_aggregated[tag_type]
            lines += [
                f"  {tag_names[tag_type]}:",
                f"    Avg Ratio (of assistants) : {s['avg_ratio']:.4f} ({s['avg_ratio']:.2%})",
                f"    Avg Per Sample            : {s['avg_per_sample']:.4f}",
            ]
        lines.append("")
    else:
        lines += ["Guide Tag Statistics:", thin, "  No guide_tag found in any file.", ""]

    lines += ["Pass@K:", thin]
    for k, pass_k in pass_at_k_list:
        lines.append(f"  Pass@{k}: {pass_k:.4f} ({pass_k:.2%})")
    lines.append("")

    lines += ["Consistency Statistics:", thin]
    lines += [
        f"  Total Samples : {consistency_stats['total_samples']}",
        f"  All Correct   : {consistency_stats['all_correct']} ({consistency_stats['all_correct_ratio']:.2%})",
        f"  All Wrong     : {consistency_stats['all_wrong']} ({consistency_stats['all_wrong_ratio']:.2%})",
        f"  Mixed         : {consistency_stats['mixed']} ({consistency_stats['mixed_ratio']:.2%})",
        "",
        sep,
    ]

    # ── Print & save ─────────────────────────────────────────────────────────
    output_text = "\n".join(lines)
    print(output_text)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(output_text)
    print(f"\n[INFO] Results saved to: {save_path}")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute metrics from evaluated output.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the inference output directory "
                             "(same as evaluate.py; must contain eval_results/ subfolder).")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to the local tokenizer directory.")
    args = parser.parse_args()

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"input_dir does not exist: {input_dir}")

    # eval_results/ is created by evaluate.py inside input_dir
    eval_dir = os.path.join(input_dir, "eval_results")
    if not os.path.isdir(eval_dir):
        raise NotADirectoryError(
            f"eval_results/ not found under {input_dir}. "
            "Please run evaluate.py first."
        )

    # Result summary saved alongside eval_results/
    save_path = os.path.join(input_dir, "metrics.txt")

    print(f"[INFO] Loading tokenizer from: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    evaluate(eval_dir, tokenizer, save_path)


if __name__ == "__main__":
    main()

