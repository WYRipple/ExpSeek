import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import uuid

# Chat template special tokens for Qwen-style models
SYS_MSG = "<|im_start|>system\n"
USR_MSG = "<|im_start|>user\n"
ASS_MSG = "<|im_start|>assistant\n"
END_MSG = "<|im_end|>\n"


def _fix_boundaries_by_matching(full_tokens, boundaries, tokenizer):
    """
    Fallback boundary correction when token count mismatches.
    Uses sliding window pattern matching to relocate segment boundaries.
    """
    fixed_boundaries = []
    current_search_start = 0

    for expected_start, expected_end, role, text in boundaries:
        if role == "system":
            fixed_boundaries.append((expected_start, expected_end, role, text))
            current_search_start = expected_end
            continue

        segment_tokens = tokenizer.encode(text, add_special_tokens=False)
        segment_len = len(segment_tokens)
        search_range = 50
        found = False

        for offset in range(-search_range, search_range):
            test_start = current_search_start + offset
            if test_start < 0 or test_start + segment_len > len(full_tokens):
                continue

            match_count = sum(
                1 for i in range(min(10, segment_len))
                if full_tokens[test_start + i] == segment_tokens[i]
            )

            if match_count >= min(8, segment_len):
                actual_end = test_start + segment_len
                fixed_boundaries.append((test_start, actual_end, role, text))
                current_search_start = actual_end
                found = True
                break

        if not found:
            fixed_boundaries.append((expected_start, expected_end, role, text))
            current_search_start = expected_end

    return fixed_boundaries


def get_token_logits(combined_text, raw_messages, model, tokenizer, model_str):
    """
    Compute per-token entropy for the last assistant turn only.

    Steps:
      1. Tokenize the full conversation on CPU.
      2. Run a single forward pass on GPU to get logits.
      3. Extract logits for the last assistant segment and compute entropy.

    Returns:
        List of dicts, each containing:
          - position: token index within the assistant segment
          - token_id: integer token id
          - token_text: decoded token string
          - token_entropy: Shannon entropy at this position
          - token_prob: model probability assigned to the actual token
    """
    ass_num = sum(1 for msg in raw_messages if msg.get("role") == "assistant")

    # Step 1: Tokenize full conversation on CPU
    full_tokens = tokenizer.encode(combined_text, add_special_tokens=False)
    token_boundaries = []

    # System message boundary
    sys_text = SYS_MSG + raw_messages[0]["content"] + END_MSG
    sys_tokens = tokenizer.encode(sys_text, add_special_tokens=False)
    current_pos = len(sys_tokens)
    token_boundaries.append((0, current_pos, "system", sys_text))

    # Per-turn boundaries
    for a_n in range(ass_num):
        # User turn
        user_text = USR_MSG + raw_messages[2 * a_n + 1]["content"] + END_MSG
        user_tokens = tokenizer.encode(user_text, add_special_tokens=False)
        user_start = current_pos
        user_end = current_pos + len(user_tokens)
        token_boundaries.append((user_start, user_end, "user", user_text))
        current_pos = user_end

        # Assistant prefix tokens (not part of response content)
        ass_prefix_tokens = tokenizer.encode(ASS_MSG, add_special_tokens=False)
        current_pos += len(ass_prefix_tokens)

        # Assistant response
        response_text = raw_messages[2 * a_n + 2]["content"]
        response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
        response_start = current_pos
        response_end = current_pos + len(response_tokens)
        token_boundaries.append((response_start, response_end, "assistant", response_text))
        current_pos = response_end

        # End token
        end_tokens = tokenizer.encode(END_MSG, add_special_tokens=False)
        current_pos += len(end_tokens)

    # Validate tokenization consistency; fix if needed
    if abs(current_pos - len(full_tokens)) > 5:
        token_boundaries = _fix_boundaries_by_matching(full_tokens, token_boundaries, tokenizer)

    # Step 2: Forward pass on GPU
    full_inputs = torch.tensor([full_tokens], dtype=torch.long).to(model.device)
    with torch.no_grad():
        outputs = model(full_inputs)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    # Step 3: Find last assistant boundary and compute entropy
    last_assistant_boundary = None
    for start, end, role, text in token_boundaries:
        if role == "assistant":
            last_assistant_boundary = (start, end, role, text)

    if last_assistant_boundary is None:
        return []

    token_logits_info = []

    try:
        start, end, role, text = last_assistant_boundary

        segment_tokens = full_tokens[start:end]
        segment_length = len(segment_tokens)

        if segment_length == 0:
            return []

        # logits[i-1] predicts token[i]
        logits_start = max(0, start - 1)
        logits_end = min(end - 1, logits.shape[1])
        segment_logits = logits[0, logits_start:logits_end, :]

        # Edge case: assistant at position 0
        if start == 0:
            segment_tokens = segment_tokens[1:]
            segment_length -= 1

        actual_length = min(segment_length, segment_logits.shape[0])
        segment_tokens = segment_tokens[:actual_length]
        segment_logits = segment_logits[:actual_length]

        # Decode all tokens at once for efficiency
        all_tokens_text = tokenizer.batch_decode(
            [[t] for t in segment_tokens],
            skip_special_tokens=False
        )

        # Compute entropy in batches to avoid OOM
        batch_size = 128
        for batch_start in range(0, actual_length, batch_size):
            batch_end = min(batch_start + batch_size, actual_length)

            batch_logits = segment_logits[batch_start:batch_end]
            batch_probs = torch.softmax(batch_logits, dim=-1)
            batch_entropies = -(batch_probs * torch.log(batch_probs + 1e-12)).sum(dim=1)

            for i in range(batch_end - batch_start):
                idx = batch_start + i
                token_id = segment_tokens[idx]
                token_logits_info.append({
                    'position': idx,
                    'token_id': int(token_id),
                    'token_text': all_tokens_text[idx],
                    'token_entropy': batch_entropies[i].item(),
                    'token_prob': batch_probs[i, token_id].item(),
                })

            del batch_logits, batch_probs, batch_entropies

        return token_logits_info

    finally:
        del outputs, logits
        if 'segment_logits' in locals():
            del segment_logits
        torch.cuda.empty_cache()


def apply_chat_template(messages, model_str):
    """
    Manually apply chat template to produce a single combined string.
    Used to prepare input for entropy computation.
    """
    ass_num = sum(1 for msg in messages if msg.get("role") == "assistant")
    combined_text = SYS_MSG + messages[0]["content"] + END_MSG
    for i in range(ass_num):
        combined_text += (
            USR_MSG + messages[2 * i + 1]["content"] + END_MSG
            + ASS_MSG + messages[2 * i + 2]["content"] + END_MSG
        )
    return combined_text


def run_entropy_server(input_queue, result_dict, model_path, model_str, ready_event, device_map="auto"):
    """
    Entropy computation server running in a separate process.
    Loads the model once, then processes requests from the queue indefinitely.
    Sends a ready signal via ready_event after model loading.
    Exits cleanly when it receives a None poison pill from the queue.
    """    
    print(f"[EntropyServer] Initializing, loading model: {model_path} ...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
            device_map=device_map,
        )
        model.eval()
        print("[EntropyServer] Model loaded successfully.")
        ready_event.set()

    except Exception as e:
        print(f"[EntropyServer] Failed to load model: {e}")
        ready_event.set()  # Notify main process to prevent deadlock
        return

    while True:
        try:
            task = input_queue.get()

            # Poison pill: graceful shutdown
            if task is None:
                print("[EntropyServer] Shutdown signal received, stopping.")
                break

            req_id, messages = task
            combined_text = apply_chat_template(messages, model_str)

            error_msg = None
            result = []

            try:
                result = get_token_logits(combined_text, messages, model, tokenizer, model_str)
            except Exception as e:
                print(f"[EntropyServer] Computation error: {e}")
                error_msg = str(e)
                torch.cuda.empty_cache()

            result_dict[req_id] = {"data": result, "error": error_msg}

        except Exception as e:
            print(f"[EntropyServer] Unexpected loop error: {e}")
            time.sleep(1)


class EntropyClient:
    """
    Client interface for sending entropy computation requests to EntropyServer.
    Communicates via multiprocessing Queue and shared Manager dict.
    """

    def __init__(self, input_queue, result_dict):
        self.input_queue = input_queue
        self.result_dict = result_dict

    def calculate(self, messages) -> list:
        """
        Send messages to the entropy server and block until result is ready.

        Returns:
            List of per-token entropy dicts, or empty list on error.
        """
        req_id = str(uuid.uuid4())
        self.input_queue.put((req_id, messages))

        while True:
            if req_id in self.result_dict:
                res = self.result_dict[req_id]
                del self.result_dict[req_id]
                if res.get("error"):
                    print(f"[EntropyClient] Server error: {res['error']}")
                    return []
                return res["data"]
            time.sleep(0.02)
