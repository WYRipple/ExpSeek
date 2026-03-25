import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import json
import multiprocessing
from omegaconf import OmegaConf
from datetime import datetime as dt
from functools import partial
from tqdm import tqdm
from multiprocessing import Pool, Manager

from expseek.trigger.entropy_server import run_entropy_server, EntropyClient
from expseek.agent.react_agent import MultiTurnReactAgent


def main_worker(task, config, root_dir, lock, entropy_client=None):
    """
    Execute a single task:
      - Initialize the agent.
      - Run the agent on the task.
      - Write the result to the corresponding iter{N}.jsonl file.
    """
    rollout_id = task["rollout_id"]
    question_id = task.get("question_id", 0)
    output_file = task["output_file"]

    try:
        agent = MultiTurnReactAgent(
            function_list=["search", "visit"],
            config=config,
            root_dir=root_dir,
            lock=lock,
            entropy_client=entropy_client,
        )

        result = agent._run(
            task,
            config.model_name,
            rollout_id,
            question_id + 1
        )

        with lock:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    except Exception as e:
        print(f"[Worker] Task failed: {e}")
        error_result = {
            "question": task["item"]["question"],
            "answer": task["item"].get("answer", ""),
            "rollout_id": rollout_id,
            "raw_item": task['item'],
            "error": f"Worker execution failed: {e}",
            "messages": [],
            "prediction": "[Failed]",
            "elapsed_time": None,
        }
        with lock:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(error_result, ensure_ascii=False) + "\n")



def run_with_entropy(all_tasks, config, ROOT_DIR):
    """
    Run all tasks with entropy server enabled.
    Starts the entropy computation server in a separate process,
    waits for it to be ready, then dispatches tasks.
    """
    MODEL_PATH = config.entropy_model_path
    MODEL_STR = config.entropy_model_str
    entropy_devices = getattr(config, 'entropy_devices', 'auto')

    with Manager() as manager:
        entropy_queue = manager.Queue()
        entropy_result_dict = manager.dict()
        lock = manager.Lock()
        ready_event = manager.Event()

        print(f"[Main] Starting entropy server (model: {MODEL_STR}, devices: {entropy_devices})...")
        
        if entropy_devices != 'auto':
            os.environ["CUDA_VISIBLE_DEVICES"] = entropy_devices
        
        server_process = multiprocessing.Process(
            target=run_entropy_server,
            args=(entropy_queue, entropy_result_dict, MODEL_PATH, MODEL_STR, ready_event)
        )
        server_process.start()
        
        if entropy_devices != 'auto':
            del os.environ["CUDA_VISIBLE_DEVICES"]

        print("[Main] Waiting for entropy server to be ready (this may take a few minutes)...")
        ready_event.wait()
        print("[Main] Entropy server ready, starting main loop.")

        entropy_client = EntropyClient(entropy_queue, entropy_result_dict)

        worker_func = partial(
            main_worker,
            config=config,
            root_dir=ROOT_DIR,
            lock=lock,
            entropy_client=entropy_client,
        )

        _dispatch_tasks(all_tasks, worker_func, config)

        print("[Main] Shutting down entropy server...")
        try:
            entropy_queue.put(None)
            server_process.join(timeout=5)
            if server_process.is_alive():
                print("[Main] Server did not exit in time, terminating forcefully.")
                server_process.terminate()
        except Exception as e:
            print(f"[Main] Error during shutdown (ignorable): {e}")
        print("[Main] Entropy server stopped.")


def run_without_entropy(all_tasks, config, ROOT_DIR):
    """
    Run all tasks without entropy server.
    entropy_client is None, so no entropy computation is performed.
    """
    with Manager() as manager:
        lock = manager.Lock()

        worker_func = partial(
            main_worker,
            config=config,
            root_dir=ROOT_DIR,
            lock=lock,
            entropy_client=None,
        )

        _dispatch_tasks(all_tasks, worker_func, config)


def _dispatch_tasks(all_tasks, worker_func, config):
    """
    Dispatch tasks in debug (serial) or parallel mode based on config.
    """
    try:
        if config.use_debug:
            print("[Mode] Debug mode: serial execution.")
            for task in tqdm(all_tasks, total=len(all_tasks), desc="Processing (Debug)"):
                worker_func(task)
        else:
            print(f"[Mode] Parallel mode: max_workers={config.max_workers}.")
            with Pool(processes=config.max_workers) as pool:
                for _ in tqdm(
                    pool.imap(worker_func, all_tasks),
                    total=len(all_tasks),
                    desc="Processing (Pool)"
                ):
                    pass

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")


if __name__ == "__main__":
    # Must use 'spawn' to avoid CUDA context conflicts when forking.
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Run ExpSeek inference.")
    parser.add_argument("--timestamp", type=str, default="",
                        help="Optional timestamp string for output directory naming.")
    parser.add_argument("--config", type=str, default="configs/expseek.yaml",
                        help="Path to the YAML config file.")
    parser.add_argument("--root_dir", type=str, default=".",
                        help="Project root directory.")
    args = parser.parse_args()

    ROOT_DIR = args.root_dir

    # Load config
    config = OmegaConf.load(args.config)
    save_model_name = os.path.basename(config.model_name.rstrip('/'))

    # Timestamp
    if config.time_stamp == "now":
        time_stamp = args.timestamp if args.timestamp else dt.now().strftime(r"%Y%m%d_%H:%M:%S")
    else:
        time_stamp = config.time_stamp

    # Ablation mode suffix
    ablate_mode = getattr(config, 'ablate', 'full')
    if ablate_mode != 'full':
        time_stamp = f"{time_stamp}-{ablate_mode}"

    # Create output directory
    output_dir = os.path.join(
        ROOT_DIR, "outputs",
        f"{save_model_name}-{config.dataset}-{time_stamp}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Save current config
    config_save_path = os.path.join(
        output_dir,
        f"config_{dt.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    )
    with open(config_save_path, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(config))
    print(f"[INFO] Config saved to: {config_save_path}")

    # Load dataset
    data_filepath = os.path.join(ROOT_DIR, "data", f"{config.dataset}.jsonl")
    with open(data_filepath, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]

    # Load experience knowledge base
    exp_kb_dict = None
    if getattr(config, 'need_guidance', False) and not getattr(config, 'zero_exp', False):
        if config.exp_kb_path:
            with open(config.exp_kb_path, 'r', encoding='utf-8') as f:
                exp_kb_dict = json.load(f)
            config.exp_data = exp_kb_dict
        else:
            raise ValueError("need_guidance=True and zero_exp=False, but exp_kb_path is empty")

    # Load embedding knowledge base (only if not using guide model)
    if getattr(config, 'need_guidance', False) and not getattr(config, 'use_guide_model', True):
        if config.embedding_kb_path:
            with open(config.embedding_kb_path, 'r', encoding='utf-8') as f:
                config.emb_data = json.load(f)
        else:
            raise ValueError("use_guide_model=False, but embedding_kb_path is empty")

    # ── Build full task list with resume support ────────────────────────────
    all_tasks = []
    for rollout_idx in range(1, config.roll_out_count + 1):
        output_file = os.path.join(output_dir, f"iter{rollout_idx}.jsonl")
        processed_queries = set()

        if os.path.exists(output_file):
            saved_data = []
            fail_num = 0
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if data.get("prediction") != "[Failed]" and data.get("termination") == "answer":
                            saved_data.append(data)
                        else:
                            fail_num += 1
                    except json.JSONDecodeError:
                        pass

            if fail_num > 0:
                print(f"[Round {rollout_idx}] Removing {fail_num} failed/incomplete samples for re-run.")

            with open(output_file, "w", encoding="utf-8") as f:
                for data in saved_data:
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")

            for data in saved_data:
                if "question" in data:
                    processed_queries.add(data["question"].strip())

        tasks_to_run = []
        for question_id, item in enumerate(items):
            q = item.get("question", "").strip()
            if q and q not in processed_queries:
                tasks_to_run.append({
                    "item": item.copy(),
                    "rollout_id": rollout_idx,
                    "question_id": question_id,
                    "output_file": output_file
                })

        print(
            f"[Round {rollout_idx}] "
            f"Total: {len(items)}, "
            f"Done: {len(processed_queries)}, "
            f"Pending: {len(tasks_to_run)}"
        )
        all_tasks.extend(tasks_to_run)

    # ── Run tasks ───────────────────────────────────────────────────────────
    if not all_tasks:
        print("[INFO] All tasks already completed, nothing to run.")
    else:
        print(f"[INFO] Total tasks: {len(all_tasks)}, starting...")

        compute_entropy = getattr(config, 'compute_entropy', False)
        need_guidance = getattr(config, 'need_guidance', False)

        if need_guidance and not compute_entropy:
            raise ValueError("need_guidance=True requires compute_entropy=True")

        if compute_entropy:
            print("[INFO] Entropy mode enabled: loading entropy model...")
            run_with_entropy(all_tasks, config, ROOT_DIR)
        else:
            print("[INFO] Entropy mode disabled: running without entropy server.")
            run_without_entropy(all_tasks, config, ROOT_DIR)

    print(f"[INFO] All {config.roll_out_count} rollouts completed.")
