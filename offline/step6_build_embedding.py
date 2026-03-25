# step6_build_embedding.py
# Input : {exp_kb_dir}/EXP-KB.json
# Output: {exp_kb_dir}/embedding/EXP-KB-embedding.json

import os
import json
import argparse
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── Embedding ─────────────────────────────────────────────────────────────────

def get_embedding(text, api_key, api_base, model, max_tries=5):
    for attempt in range(max_tries):
        try:
            client = OpenAI(api_key=api_key, base_url=api_base)
            resp = client.embeddings.create(
                model=model,
                input=text,
                dimensions=1024,
                encoding_format="float",
            )
            return resp.model_dump()["data"][0]["embedding"]
        except Exception as e:
            time.sleep(2 ** attempt)
            print(f"[Step6] Embedding error (attempt {attempt+1}/{max_tries}): {e}")
    return []


def embed_record(args):
    """Worker function for ThreadPoolExecutor."""
    record, api_key, api_base, model = args
    rec = record.copy()
    rec["behavior_embedding"] = get_embedding(
        rec["behavior"], api_key, api_base, model
    )
    return rec


# ── Flatten + Embed ───────────────────────────────────────────────────────────

def flatten_exp(exp_dict):
    """Flatten label-grouped structure into a flat list, preserving label field."""
    all_records = []
    label_pool  = exp_dict.get("label_pool", [])
    for label in label_pool:
        if label in exp_dict:
            for rec in exp_dict[label]:
                item = rec.copy()
                item["label"] = label
                all_records.append(item)
    return all_records


def embed_all(records, api_key, api_base, model, num_workers):
    tasks   = [(rec, api_key, api_base, model) for rec in records]
    results = [None] * len(tasks)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {
            executor.submit(embed_record, task): idx
            for idx, task in enumerate(tasks)
        }
        for future in tqdm(as_completed(future_to_idx),
                           total=len(tasks),
                           desc="  Embedding"):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"[Step6] Record {idx} failed: {e}")
                results[idx] = tasks[idx][0]

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step6: Build embedding index for EXP-KB."
    )
    parser.add_argument("--exp_kb_dir",  type=str, required=True,
                        help="Experience KB directory (contains EXP-KB.json).")
    parser.add_argument("--api_key",     type=str, required=True)
    parser.add_argument("--api_base",    type=str, required=True)
    parser.add_argument("--model",       type=str, required=True,
                        help="Embedding model name.")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of concurrent embedding threads.")
    args = parser.parse_args()

    input_path  = os.path.join(args.exp_kb_dir, "EXP-KB.json")
    output_path = os.path.join(args.exp_kb_dir, "embedding", "EXP-KB-embedding.json")

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"EXP-KB.json not found in {args.exp_kb_dir}. "
            "Please run step4 first."
        )

    if os.path.exists(output_path):
        print(f"[Step6] EXP-KB-embedding.json already exists, skipping.")
        return

    kb = load_json(input_path)

    # ── Process exp ───────────────────────────────────────────────────────────
    print(f"\n[Step6] ── Processing process_exp ────────────────────")
    process_records = flatten_exp(kb.get("process_exp", {}))
    print(f"[Step6] Records to embed: {len(process_records)}")
    process_embedded = embed_all(
        process_records, args.api_key, args.api_base,
        args.model, args.num_workers
    )

    # ── Final exp ─────────────────────────────────────────────────────────────
    print(f"\n[Step6] ── Processing final_exp ──────────────────────")
    final_records = flatten_exp(kb.get("final_exp", {}))
    print(f"[Step6] Records to embed: {len(final_records)}")
    final_embedded = embed_all(
        final_records, args.api_key, args.api_base,
        args.model, args.num_workers
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    result = {
        "process_exp": process_embedded,
        "final_exp":   final_embedded,
    }
    save_json(output_path, result)

    print(f"\n[Step6] ── Summary ───────────────────────────────────")
    print(f"[Step6] process_exp records : {len(process_embedded)}")
    print(f"[Step6] final_exp records   : {len(final_embedded)}")
    print(f"[Step6] Total records       : "
          f"{len(process_embedded) + len(final_embedded)}")
    print(f"[Step6] Saved to            : {output_path}")


if __name__ == "__main__":
    main()
