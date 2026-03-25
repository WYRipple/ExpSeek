import requests
from openai import OpenAI
import time
import json
import numpy as np


def get_llm_response(msgs, config, stop=None):
    """Call the main agent LLM and return the response content."""
    for attempt in range(config.max_retries + 10):
        try:
            if config.model_mode == "api":
                client = OpenAI(
                    api_key=config.api_key,
                    base_url=config.api_base,
                )
                if "qwen3" in config.model_name.lower() and "max" not in config.model_name.lower():
                    chat_response = client.chat.completions.create(
                        model=config.model_name,
                        messages=msgs,
                        stream=True,
                        extra_body={"enable_thinking": True}
                    )
                    reason = ""
                    answer = ""
                    for chunk in chat_response:
                        res = json.loads(chunk.model_dump_json())["choices"][0]["delta"]
                        content_reason = res.get("reasoning_content")
                        if content_reason:
                            reason += content_reason
                        content_piece = res.get("content")
                        if content_piece:
                            answer += content_piece
                    return "<think>" + reason + answer
                else:
                    chat_response = client.chat.completions.create(
                        model=config.model_name,
                        messages=msgs,
                    )

            elif config.model_mode == "vllm":
                client = OpenAI(
                    api_key=config.api_key,
                    base_url=config.api_base,
                )
                chat_response = client.chat.completions.create(
                    model=config.model_name,
                    messages=msgs,
                    stop=stop,
                    temperature=config.temperature,
                    top_p=config.top_p,
                )

            answer = chat_response.choices[0].message.content
            return answer

        except Exception as e:
            time.sleep(2 * attempt)
            if attempt == (config.max_retries + 10 - 1):
                print(f"[LLM] Final call failed after all retries: {e}")
                return "LLM server error"
            print(f"[LLM] Attempt {attempt} failed: {e}, retrying...")
            if hasattr(e, 'type') and e.type == "data_inspection_failed":
                print("[LLM] Content safety error, skipping...")
                return "Content safety error, no output."
            continue

    return "LLM server empty response"


def get_llm_summary(msgs, config):
    """Call the summary LLM to extract and summarize webpage content."""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            client = OpenAI(
                api_key=config.sum_api_key,
                base_url=config.sum_api_base,
            )
            chat_response = client.chat.completions.create(
                model=config.sum_model_name,
                messages=msgs,
                temperature=0,
                top_p=1,
            )
            answer = chat_response.choices[0].message.content
            return answer

        except Exception as e:
            time.sleep(2 ** attempt)
            if attempt == (max_retries - 1):
                print(f"[SUMMARY] Final call failed after all retries: {e}")
                return "LLM server error"
            print(f"[SUMMARY] Attempt {attempt} failed: {e}, retrying...")
            continue

    return "LLM server empty response"


def get_llm_guide(msgs, config):
    """Call the experience/guidance LLM to generate guidance content."""
    for attempt in range(config.max_retries):
        try:
            client = OpenAI(
                api_key=config.guide_api_key,
                base_url=config.guide_api_base,
            )
            chat_response = client.chat.completions.create(
                model=config.guide_model_name,
                messages=msgs,
                max_tokens=65536,
                temperature=0,
                top_p=1,
            )
            answer = chat_response.choices[0].message.content
            return answer

        except Exception as e:
            time.sleep(2 ** attempt)
            if attempt == (config.max_retries - 1):
                print(f"[GUIDE] Final call failed after all retries: {e}")
                return "LLM server error"
            print(f"[GUIDE] Attempt {attempt} failed: {e}, retrying...")
            continue

    return "LLM server empty response"


def get_llm_judge(msgs, config):
    """Call the judge LLM to determine whether guidance should be provided."""
    for attempt in range(config.max_retries):
        try:
            client = OpenAI(
                api_key=config.judge_api_key,
                base_url=config.judge_api_base,
            )
            chat_response = client.chat.completions.create(
                model=config.judge_model_name,
                messages=msgs,
                max_tokens=10000,
                temperature=0,
                top_p=1,
            )
            answer = chat_response.choices[0].message.content
            return answer

        except Exception as e:
            time.sleep(2 ** attempt)
            if attempt == (config.max_retries - 1):
                print(f"[JUDGE] Final call failed after all retries: {e}")
                return "LLM server error"
            print(f"[JUDGE] Attempt {attempt} failed: {e}, retrying...")
            continue

    return "LLM server empty response"


def get_embedding(input_text, config):
    for attempt in range(5):
        try:
            client = OpenAI(
                api_key=config.embedding_api_key,
                base_url=config.embedding_api_base,
            )
            completion = client.embeddings.create(
                model=config.embedding_model_name,      # ← 从config读
                input=input_text,
                dimensions=config.embedding_dimensions, # ← 从config读
                encoding_format="float"
            )
            completion_json = completion.model_dump()
            embedding = completion_json['data'][0]['embedding']
            return embedding

        except Exception as e:
            time.sleep(2 ** attempt)
            print(f"[EMBEDDING] Attempt {attempt} failed: {e}, retrying...")

    return []



def get_top_exp(this_embedding, emb_data, this_tag):
    """
    Retrieve the most similar experience from the embedding knowledge base
    using cosine similarity.

    Args:
        this_embedding: query embedding vector (list, 1024-dim)
        emb_data: embedding knowledge base loaded from JSON
        this_tag: "process" or "answer"

    Returns:
        dict: {"behavior": "...", "mistake": "...", "guidance": "..."}
    """
    # Select experience list based on step type
    if this_tag == "process":
        exp_list = emb_data.get("process_exp", [])
    elif this_tag == "answer":
        exp_list = emb_data.get("final_exp", [])
    else:
        raise ValueError(f"Invalid tag: {this_tag}, must be 'process' or 'answer'")

    if not exp_list:
        return {"behavior": "", "mistake": "", "guidance": ""}

    # Convert query to numpy array
    query_emb = np.array(this_embedding)

    # Compute cosine similarity for all experiences
    max_similarity = -1
    best_exp = None

    for exp in exp_list:
        behavior_emb = np.array(exp.get("behavior_embedding", []))
        similarity = np.dot(query_emb, behavior_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(behavior_emb) + 1e-10
        )
        # Track best match
        if similarity > max_similarity:
            max_similarity = similarity
            best_exp = exp

    # Return the most similar experience
    if best_exp:
        return {
            "behavior": best_exp.get("behavior", ""),
            "mistake": best_exp.get("mistake", ""),
            "guidance": best_exp.get("guidance", "")
        }
    else:
        return {"behavior": "", "mistake": "", "guidance": ""}
