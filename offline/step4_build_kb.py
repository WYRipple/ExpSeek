# step4_build_kb.py
# Input : {exp_kb_dir}/EXP-KB-process-label.jsonl
#         {exp_kb_dir}/EXP-KB-final-label.jsonl
# Output: {exp_kb_dir}/EXP-KB.json
#         {exp_kb_dir}/EXP-KB-embedding.json (optional)

import os
import json
import argparse
import time
from collections import defaultdict
from openai import OpenAI


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ── Build KB ──────────────────────────────────────────────────────────────────

def build_kb(process_list, final_list):
    """
    Organize labeled experiences into a structured KB.
    Structure:
    {
        "process_exp": {
            "label_pool": [...],
            "<label>": [{exp_id, behavior, mistake, guidance}, ...]
        },
        "final_exp": {
            "label_pool": [...],
            "<label>": [{exp_id, behavior, mistake, guidance}, ...]
        }
    }
    """
    def organize(data_list):
        label_dict = defaultdict(list)
        for item in data_list:
            label = item.get("label", "unknown")
            record = {
                "exp_id":   item["exp_id"],
                "behavior": item["behavior"],
                "mistake":  item["mistake"],
                "guidance": item["guidance"],
            }
            label_dict[label].append(record)
        label_pool = sorted(label_dict.keys())
        return {
            "label_pool": label_pool,
            **{k: v for k, v in label_dict.items()}
        }

    return {
        "process_exp": organize(process_list),
        "final_exp":   organize(final_list),
    }


# ── Embedding ─────────────────────────────────────────────────────────────────

def get_embedding(text, api_key, api_base, max_tries=5):
    for attempt in range(max_tries):
        try:
            client = OpenAI(api_key=api_key, base_url=api_base)
            resp = client.embeddings.create(
                model="text-embedding-v4",
                input=text,
                dimensions=1024,
                encoding_format="float",
            )
            return resp.model_dump()["data"][0]["embedding"]
        except Exception as e:
            time.sleep(2 ** attempt)
            print(f"[Step4] Embedding error (attempt {attempt+1}/{max_tries}): {e}")
    return []


def build_kb_with_embedding(kb, api_key, api_base):
    """
    Flatten KB and add behavior_embedding to each record.
    Structure:
    {
        "process_exp": [{exp_id, behavior, mistake, guidance, behavior_embedding}, ...],
        "final_exp":   [{exp_id, behavior, mistake, guidance, behavior_embedding}, ...]
    }
    """
    def flatten_and_embed(exp_dict):
        result = []
        for label in exp_dict["label_pool"]:
            for record in exp_dict.get(label, []):
                rec = record.copy()
                rec["label"]             = label
                rec["behavior_embedding"] = get_embedding(
                    rec["behavior"], api_key, api_base
                )
                result.append(rec)
        return result

    print("[Step4] Embedding process experiences...")
    process_embedded = flatten_and_embed(kb["process_exp"])

    print("[Step4] Embedding final experiences...")
    final_embedded = flatten_and_embed(kb["final_exp"])

    return {
        "process_exp": process_embedded,
        "final_exp":   final_embedded,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step4: Build experience knowledge base."
    )
    parser.add_argument("--exp_kb_dir", type=str, required=True,
                        help="Experience KB directory.")
    parser.add_argument("--embedding",  action="store_true",
                        help="Also compute behavior embeddings.")
    parser.add_argument("--api_key",    type=str, default=None,
                        help="API key (required if --embedding).")
    parser.add_argument("--api_base",   type=str, default=None,
                        help="API base (required if --embedding).")
    args = parser.parse_args()

    process_path = os.path.join(args.exp_kb_dir, "EXP-KB-process-label.jsonl")
    final_path   = os.path.join(args.exp_kb_dir, "EXP-KB-final-label.jsonl")
    kb_path      = os.path.join(args.exp_kb_dir, "EXP-KB.json")
    emb_path     = os.path.join(args.exp_kb_dir, "EXP-KB-embedding.json")

    for p in [process_path, final_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{os.path.basename(p)} not found in {args.exp_kb_dir}. "
                "Please run step3 first."
            )

    # Build text KB
    if os.path.exists(kb_path):
        print(f"[Step4] EXP-KB.json already exists, skipping text KB build.")
    else:
        process_list = load_jsonl(process_path)
        final_list   = load_jsonl(final_path)
        kb = build_kb(process_list, final_list)

        with open(kb_path, "w", encoding="utf-8") as f:
            json.dump(kb, f, ensure_ascii=False, indent=2)

        print(f"[Step4] EXP-KB.json saved: {kb_path}")
        print(f"[Step4] Process labels: {len(kb['process_exp']['label_pool'])}, "
              f"records: {sum(len(v) for k,v in kb['process_exp'].items() if k != 'label_pool')}")
        print(f"[Step4] Final labels  : {len(kb['final_exp']['label_pool'])}, "
              f"records: {sum(len(v) for k,v in kb['final_exp'].items() if k != 'label_pool')}")

    # Build embedding KB (optional)
    if args.embedding:
        if os.path.exists(emb_path):
            print(f"[Step4] EXP-KB-embedding.json already exists, skipping.")
        else:
            if not args.api_key or not args.api_base:
                raise ValueError(
                    "--api_key and --api_base are required when --embedding is set."
                )
            with open(kb_path, "r", encoding="utf-8") as f:
                kb = json.load(f)

            kb_emb = build_kb_with_embedding(kb, args.api_key, args.api_base)

            with open(emb_path, "w", encoding="utf-8") as f:
                json.dump(kb_emb, f, ensure_ascii=False, indent=2)

            print(f"[Step4] EXP-KB-embedding.json saved: {emb_path}")
            print(f"[Step4] Process records: {len(kb_emb['process_exp'])}")
            print(f"[Step4] Final records  : {len(kb_emb['final_exp'])}")


if __name__ == "__main__":
    main()
