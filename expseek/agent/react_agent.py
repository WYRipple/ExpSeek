import json
import os
import random
import datetime
import time
import copy
from typing import Dict, List, Optional, Union
from transformers import AutoTokenizer

from expseek.agent.base_agent import BaseAgent
from expseek.llm.client import get_llm_response, get_llm_guide, get_embedding, get_top_exp
from expseek.tools.tool_manager import tool_manager
from expseek.agent.prompt import (
    SYS_PROMPT,
    STAGE_ONE_GUIDE,
    STAGE_TWO_GUIDE,
    ZERO_EXP_GUIDE,
    EMB_GUIDE_PROMPT,
)


class MultiTurnReactAgent(BaseAgent):
    """
    Multi-turn ReAct agent with entropy-based experience guidance.

    At each reasoning step, the agent:
      1. Calls the LLM to produce a thought + tool call or final answer.
      2. Computes token-level entropy for the response.
      3. Decides whether to inject experience-based guidance based on entropy.
      4. Continues until a final answer is produced or the call budget is exhausted.
    """

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict]]] = None,
                 description: Optional[str] = None,
                 files: Optional[List[str]] = None,
                 config=None,
                 root_dir=None,
                 lock=None,
                 entropy_client=None,
                 **kwargs):
        self.config = config
        self.root_dir = root_dir
        self.lock = lock
        self.entropy_client = entropy_client

        self._import_tools()
        super().__init__(
            function_list=function_list,
            description=description,
            files=files,
            **kwargs
        )
        self.system_message = SYS_PROMPT

    def _import_tools(self):
        """Import and register tools with shared config and lock."""
        from expseek.tools.tool_manager import tool_manager
        tool_manager._default_init_kwargs = {
            "config": self.config,
            "root_dir": self.root_dir,
            "lock": self.lock
        }
        import expseek.tools.tool_search
        import expseek.tools.tool_visit

    def count_tokens(self, messages) -> int:
        """Count total tokens in a message list using the local tokenizer."""
        cleaned_messages = [
            {'role': msg['role'], 'content': msg['content']}
            for msg in messages
        ]
        tokenizer = AutoTokenizer.from_pretrained(f"{self.root_dir}/tokenizer")
        tokens = tokenizer.apply_chat_template(
            cleaned_messages,
            tokenize=True,
            add_generation_prompt=False
        )
        return len(tokens)

    def extract_innermost_tool_call(self, content: str) -> str:
        """
        Extract the innermost <tool_call> tag content.
        If no closing tag exists, returns content from <tool_call> to end of string.
        """
        start_tag = "<tool_call>"
        end_tag = "</tool_call>"
        stack = []
        pairs = []

        i = 0
        while i < len(content):
            if content.startswith(start_tag, i):
                stack.append(i + len(start_tag))
                i += len(start_tag)
            elif content.startswith(end_tag, i):
                if stack:
                    start_pos = stack.pop()
                    pairs.append((start_pos, i))
                i += len(end_tag)
            else:
                i += 1

        if pairs:
            start_pos, end_pos = pairs[-1]
            return content[start_pos:end_pos].strip()

        if stack:
            return content[stack[-1]:].strip()

        return ""

    def avg_token_logits(self, token_logits) -> float:
        """Compute average token entropy for the last assistant response."""
        if not token_logits:
            return 0.0
        entropies = [token['token_entropy'] for token in token_logits]
        return sum(entropies) / len(entropies)

    def interpolate_probability(self, value, start, end) -> int:
        """
        Linear interpolation-based trigger decision.

        Returns 1 (trigger guidance) with probability proportional to
        how far value is between start and end.

        Args:
            value: current average entropy
            start: lower bound (probability = 0)
            end:   upper bound (probability = 1)
        """
        if value >= end:
            return 1
        if value <= start:
            return 0
        probability = (value - start) / (end - start)
        return 1 if random.random() < probability else 0

    def parse_result_one(self, result) -> list:
        """
        Parse the guide model's stage-one response to extract topic index list.
        Expects the last non-comment line to be space-separated integers.
        """
        content = result.strip().removeprefix('```').removesuffix('```').strip()
        for line in reversed(content.split('\n')):
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    idxs = [int(x) for x in line.split()]
                    if idxs:
                        return idxs
                except ValueError:
                    continue
        return []

    def parse_result_two(self, result) -> str:
        """
        Parse the guide model's stage-two response to extract guidance content.
        Expects a '# Guidance Content' section header.
        """
        content = result.strip().removeprefix('```').removesuffix('```').strip()
        if '# Guidance Content' in content:
            return content.split('# Guidance Content', 1)[1].strip()
        return ""

    def format_topic_content(self, topic_idxs, exp_kb_dict) -> str:
        label_pool = exp_kb_dict['label_pool']
        result_parts = []

        for idx in topic_idxs:
            topic_name = label_pool[idx]
            result_parts.append(f"# {topic_name}")
            experiences = exp_kb_dict[topic_name]
            exp_texts = [
                f"behavior: {exp['behavior']}\n"
                f"mistake: {exp['mistake']}\n"
                f"guidance: {exp['guidance']}"
                for exp in experiences
            ]
            result_parts.append('\n\n'.join(exp_texts))

        return '\n\n'.join(result_parts)


    def get_guidance_content_and_tag(self, avg_entropy_this_turn, messages, guide_tag, tool_response):
        """
        Decide whether to inject guidance and generate guidance content.

        Args:
            avg_entropy_this_turn: average token entropy of the last assistant turn
            messages: full conversation history
            guide_tag: list of guidance tags from previous turns
            tool_response: tool response string if current step used a tool, else ""

        Returns:
            (guide_tag_int, guide_content_str, topic_idxs_list)
            guide_tag_int: 0=no guidance, 1=process guidance, 2=answer guidance
        """
        print(f"[GUIDE] Current guide_tag history: {guide_tag}")

        # Ablation mode check
        ablate_mode = getattr(self.config, 'ablate', 'full')
        current_stage = "process" if tool_response != "" else "answer"

        if ablate_mode == "only_process" and current_stage == "answer":
            print(f"[GUIDE] Ablation mode={ablate_mode}, skipping answer intervention.")
            return 0, "", []
        elif ablate_mode == "only_answer" and current_stage == "process":
            print(f"[GUIDE] Ablation mode={ablate_mode}, skipping process intervention.")
            return 0, "", []

        # Guidance interval check: avoid consecutive interventions
        if self.config.guidance_interval == 1:
            # Special case: first turn had process guidance, second turn can have answer guidance
            if len(guide_tag) == 1 and guide_tag[-1] == 1 and tool_response == "":
                this_tag = "answer"
            else:
                if len(guide_tag) > 0 and guide_tag[-1] in [1, 2]:
                    return 0, "", []
                this_tag = "process" if tool_response != "" else "answer"

        elif self.config.guidance_interval == 2:
            if len(guide_tag) > 0 and guide_tag[-1] in [1, 2]:
                return 0, "", []
            if len(guide_tag) > 1 and guide_tag[-2] in [1, 2]:
                return 0, "", []
            this_tag = "process" if tool_response != "" else "answer"

        elif self.config.guidance_interval == 0:
            this_tag = "process" if tool_response != "" else "answer"

        # Strip extra fields before passing to guide model
        cleaned_messages = [
            {'role': msg['role'], 'content': msg['content']}
            for msg in messages
        ]

        # Entropy-based trigger decision
        if this_tag == "process":
            use_guide = self.interpolate_probability(
                avg_entropy_this_turn,
                self.config.process_start,
                self.config.process_end
            )
            if use_guide == 0:
                print("[GUIDE] Entropy below threshold, skipping guidance.")
                return 0, "", []
            guide_tag_int = 1
            print("[GUIDE] Guidance triggered, tag=1 (process).")
            guide_exp_kb = self.config.exp_data['process_exp']
        else:
            use_guide = self.interpolate_probability(
                avg_entropy_this_turn,
                self.config.final_start,
                self.config.final_end
            )
            if use_guide == 0:
                print("[GUIDE] Entropy below threshold, skipping guidance.")
                return 0, "", []
            guide_tag_int = 2
            print("[GUIDE] Guidance triggered, tag=2 (answer).")
            guide_exp_kb = self.config.exp_data['final_exp']

        # Zero-experience mode: guide model generates guidance from scratch
        if self.config.zero_exp:
            from expseek.agent.prompt import ZERO_EXP_GUIDE
            zero_prompt = ZERO_EXP_GUIDE.format(history=cleaned_messages[1:])
            message_zero = [{"role": "user", "content": zero_prompt}]
            guide_content = ""
            for attempt in range(self.config.max_retries):
                try:
                    zero_response = get_llm_guide(message_zero, self.config)
                    print(f"[GUIDE] Zero-exp response:\n{zero_response}")
                    guide_content = self.parse_result_two(zero_response)
                    break
                except Exception as e:
                    print(f"[GUIDE] Attempt {attempt} failed: {e}, retrying...")
                    continue

            if guide_content == "":
                print("[GUIDE] Failed to generate guidance, skipping.")
                return 0, "", []
            return guide_tag_int, guide_content, []

        # Guide model mode: two-stage topic selection + guidance generation
        if self.config.use_guide_model:
            topic_num = len(guide_exp_kb['label_pool'])
            topic_list = '\n'.join(
                f"{i}: {item}" for i, item in enumerate(guide_exp_kb['label_pool'])
            )
            stage_one_prompt = STAGE_ONE_GUIDE.format(
                history=cleaned_messages[1:],
                topic_list=topic_list
            )
            message_one = [{"role": "user", "content": stage_one_prompt}]

            topic_idxs = []
            for attempt in range(self.config.max_retries):
                try:
                    stage_one_response = get_llm_guide(message_one, self.config)
                    print(f"[GUIDE] Stage-one response:\n{stage_one_response}")
                    topic_idxs = self.parse_result_one(stage_one_response)

                    if len(topic_idxs) != 3:
                        raise ValueError(f"Expected 3 topic indices, got {len(topic_idxs)}.")
                    if any(idx < 0 for idx in topic_idxs):
                        raise ValueError(f"Negative indices not allowed: {topic_idxs}.")
                    if any(idx >= topic_num for idx in topic_idxs):
                        raise ValueError(f"Index out of range (max={topic_num-1}): {topic_idxs}.")
                    break
                except Exception as e:
                    print(f"[GUIDE] Stage-one attempt {attempt} failed: {e}, retrying...")
                    continue

            if topic_idxs == []:
                print("[GUIDE] Stage-one failed, skipping guidance.")
                return 0, "", []

            topic_kb = self.format_topic_content(topic_idxs, guide_exp_kb)
            stage_two_prompt = STAGE_TWO_GUIDE.format(
                history=cleaned_messages[1:],
                topic_kb=topic_kb
            )
            message_two = [{"role": "user", "content": stage_two_prompt}]

            guide_content = ""
            for attempt in range(self.config.max_retries):
                try:
                    stage_two_response = get_llm_guide(message_two, self.config)
                    print(f"[GUIDE] Stage-two response:\n{stage_two_response}")
                    guide_content = self.parse_result_two(stage_two_response)
                    break
                except Exception as e:
                    print(f"[GUIDE] Stage-two attempt {attempt} failed: {e}, retrying...")
                    continue

            if guide_content == "":
                print("[GUIDE] Stage-two failed, skipping guidance.")
                return 0, "", []

            return guide_tag_int, guide_content, topic_idxs

        # Embedding retrieval mode
        else:
            input_text = (
                cleaned_messages[-2]['content'] if this_tag == "process"
                else cleaned_messages[-1]['content']
            )
            this_embedding = get_embedding(input_text, self.config)
            exp_dict = get_top_exp(this_embedding, self.config.emb_data, this_tag)

            if exp_dict['behavior'] != "":
                guide_content = EMB_GUIDE_PROMPT.format(
                    behavior=exp_dict['behavior'],
                    mistake=exp_dict['mistake'],
                    guidance=exp_dict['guidance']
                )
                return guide_tag_int, guide_content, []
            else:
                return 0, "", []

    def _compute_entropy(self, messages, question_id, round_num, tag):
        """
        Send messages to entropy server and store results in the last message.

        Args:
            messages: full conversation history (last message is assistant turn)
            question_id: for logging
            round_num: for logging
            tag: guide_tag value to assign (e.g. 3 for token-limit, 5 for format error)
        """
        if self.entropy_client is None:
            return 0.0

        print(f"-> [Entropy] Computing entropy for Question-{question_id} Round-{round_num}...")
        token_logits = self.entropy_client.calculate(messages)
        messages[-1]["token_entropy"] = token_logits
        avg_entropy = self.avg_token_logits(token_logits)
        messages[-1]["token_entropy_avg"] = avg_entropy
        messages[-1]["guide_tag"] = tag
        messages[-1]["topic_idxs"] = []
        print(f"<- [Entropy] Average entropy this turn: {avg_entropy}")
        return avg_entropy

    def _build_result(self, question, answer, data, messages, prediction, termination, start_time):
        """Package final result dict."""
        elapsed_time = time.time() - start_time
        return {
            "question": question,
            "answer": answer,
            "rollout_id": data['rollout_id'],
            "raw_item": data['item'],
            "messages": messages,
            "prediction": prediction,
            "termination": termination,
            "elapsed_time": elapsed_time,
        }

    def _run(self, data, model: str, roll_out_count: float, question_id: float, **kwargs) -> List[Dict]:
        """
        Main agent loop.

        Iterates until a final answer is produced or the LLM call budget is exhausted.
        At each step:
          1. Check token length budget.
          2. Call LLM and validate format.
          3. Compute entropy.
          4. Decide whether to inject guidance.
          5. Call tool if requested.
        """
        start_time = time.time()
        self.model = model

        question = data['item']['question']
        answer = data['item']['answer']

        # Build initial prompt
        self.user_prompt = question + "/no_think"
        time_str = "\nCurrent time: " + datetime.date.today().strftime("%Y-%m-%d")

        guide_tag = []
        messages = [
            {"role": "system", "content": self.system_message + time_str},
            {"role": "user", "content": self.user_prompt, "guide_content": ""}
        ]

        num_llm_calls_available = self.config.max_call_per_run
        round_num = 0

        # Format reminder for when the model produces invalid output
        FORMAT_REMINDER = (
            "Remember, if you are thinking, follow the <thought> thinking process here </thought> format; "
            "if you are calling a tool, follow the <tool_call> tool call here </tool_call> format; "
            "if you are responding to a tool call, follow the <tool_response> tool response here </tool_response> format; "
            "if you are answering the question, follow the <answer> answer here </answer> format. "
            "DO NOT use any other <> format tags."
        )

        TOKEN_LIMIT_PROMPT = (
            "You have now reached the maximum context length you can handle. "
            "You should stop making tool calls and, based on all the information above, "
            "think again and provide what you consider the most likely answer in the following format:"
            "<thought>your final thinking</thought>\n<answer>your answer</answer>/no_think"
        )

        TOOL_ERROR_MSG = (
            "The tool_call format is incorrect or the tool_call tag failed to be generated successfully. "
            "You should use the unique <tool_call>xxx</tool_call> tags to wrap the standard JSON format "
            "required by the system prompt in order to correctly invoke the external tool."
        )

        while num_llm_calls_available > 0:
            round_num += 1
            num_llm_calls_available -= 1
            print(f"[Round {round_num} Start] Rollout={roll_out_count} QuestionID={question_id}")

            # ── Token budget check ──────────────────────────────────────────
            token_count = self.count_tokens(messages)
            if token_count + self.config.response_budget > self.config.max_tokens:
                print("[TokenLimit] Context length exceeded, forcing final answer.")
                if token_count + 100 > self.config.max_tokens:
                    messages = messages[:-2]

                messages.append({
                    "role": "user",
                    "content": TOKEN_LIMIT_PROMPT,
                    "guide_content": ""
                })
                content = get_llm_response(
                    messages,
                    config=self.config,
                    stop=["\n<tool_response>", "<tool_response>"]
                )
                messages.append({"role": "assistant", "content": content.strip()})
                self._compute_entropy(messages, question_id, round_num, tag=3)
                guide_tag.append(3)

                termination = (
                    'generate an answer as token limit reached'
                    if '<answer>' in content
                    else 'format error: generate an answer as token limit reached'
                )
                return self._build_result(
                    question, answer, data, messages, "[Failed]", termination, start_time
                )

            # ── LLM call with format validation ────────────────────────────
            content = ""
            for attempt in range(self.config.max_retries):
                try:
                    content = get_llm_response(
                        messages,
                        config=self.config,
                        stop=["\n<tool_response>", "<tool_response>"]
                    )

                    has_think = '</thought>' in content
                    has_tool  = '<tool_call>' in content
                    has_ans   = '<answer>' in content

                    format_error = not has_ans and (not has_think or not has_tool)

                    if format_error:
                        print(f"[FormatWarn] Attempt {attempt+1}/{self.config.max_retries}: invalid format, retrying...")
                        time.sleep((1 + attempt) ** 2)
                        if attempt == self.config.max_retries - 1:
                            print("[FormatWarn] Max retries reached, using last content.")
                        continue
                    break

                except Exception as e:
                    print(f"[LLMError] Attempt {attempt+1}/{self.config.max_retries}: {e}")
                    if attempt == self.config.max_retries - 1:
                        print("[LLMError] Max retries reached.")
                    continue

            # ── Content safety block ────────────────────────────────────────
            if content == "Content safety error, no output.":
                messages.append({"role": "assistant", "content": content})
                messages[-1]["token_entropy"] = []
                messages[-1]["token_entropy_avg"] = -999
                messages[-1]["guide_tag"] = 4
                messages[-1]["topic_idxs"] = []
                guide_tag.append(4)
                break

            print(f"[LLM Output]\n{content}\n")

            # ── All-format-missing branch ───────────────────────────────────
            if not any(tag in content for tag in ['<thought>', '<tool_call>', '<answer>', '<tool_response>']):
                print("[FormatError] No valid tags found in content.")
                messages.append({"role": "assistant", "content": content.strip()})
                self._compute_entropy(messages, question_id, round_num, tag=5)
                guide_tag.append(5)
                messages.append({
                    "role": "user",
                    "content": FORMAT_REMINDER + "/no_think",
                    "guide_content": ""
                })
                continue

            # ── Truncate if model echoed tool_response ──────────────────────
            if '<tool_response>' in content:
                content = content[:content.find('<tool_response>')]

            # ── Append assistant message and compute entropy ────────────────
            messages.append({"role": "assistant", "content": content.strip()})
            if self.entropy_client is not None:
                print(f"-> [Entropy] Computing for Question-{question_id} Round-{round_num}...")
                token_logits = self.entropy_client.calculate(messages)
                messages[-1]["token_entropy"] = token_logits
                avg_entropy_this_turn = self.avg_token_logits(token_logits)
                messages[-1]["token_entropy_avg"] = avg_entropy_this_turn
                print(f"<- [Entropy] Average entropy: {avg_entropy_this_turn}")
            else:
                avg_entropy_this_turn = 0.0
                messages[-1]["token_entropy"] = []
                messages[-1]["token_entropy_avg"] = 0.0

            # ── Final answer branch ─────────────────────────────────────────
            if '<answer>' in content:
                if self.entropy_client is not None and getattr(self.config, 'need_guidance', False):
                    this_tag, guide_content, topic_idxs = self.get_guidance_content_and_tag(
                        avg_entropy_this_turn, messages, guide_tag, tool_response=""
                    )
                else:
                    this_tag, guide_content, topic_idxs = 0, "", []

                messages[-1]["guide_tag"] = this_tag
                messages[-1]["topic_idxs"] = topic_idxs
                guide_tag.append(this_tag)

                if this_tag == 0:
                    # No guidance needed, accept the answer
                    termination = 'answer'
                    break
                else:
                    # Inject guidance and ask the model to reconsider
                    use_guide_content = f"<guidance>\n{guide_content}\n</guidance>/no_think"
                    messages.append({
                        "role": "user",
                        "content": use_guide_content,
                        "guide_content": guide_content
                    })
                    continue

            # ── Tool call branch ────────────────────────────────────────────
            if '<tool_call>' in content:
                tool_call_str = self.extract_innermost_tool_call(content)
                try:
                    tool_call = json.loads(tool_call_str)
                    tool_name = tool_call.get('name', '')
                    tool_args = tool_call.get('arguments', {})
                    tool_result = self._call_tool(tool_name, tool_args)
                    tool_result_wrapped = f"<tool_response>\n{tool_result}\n</tool_response>"

                    # Decide guidance based on entropy + tool result context
                    this_message = copy.deepcopy(messages)
                    this_message.append({"role": "user", "content": tool_result_wrapped})

                    if self.entropy_client is not None and getattr(self.config, 'need_guidance', False):
                        this_tag, guide_content, topic_idxs = self.get_guidance_content_and_tag(
                            avg_entropy_this_turn, this_message, guide_tag, tool_response=tool_result_wrapped
                        )
                    else:
                        this_tag, guide_content, topic_idxs = 0, "", []

                    messages[-1]["guide_tag"] = this_tag
                    messages[-1]["topic_idxs"] = topic_idxs
                    guide_tag.append(this_tag)

                    use_guide_content = (
                        f"<guidance>\n{guide_content}\n</guidance>/no_think"
                        if guide_content != "" else "/no_think"
                    )
                    final_user_content = tool_result_wrapped + "\n" + use_guide_content
                    print(f"[ToolResult]\n{final_user_content[:300]}\n... (truncated)\n")
                    messages.append({
                        "role": "user",
                        "content": final_user_content,
                        "guide_content": guide_content
                    })

                except Exception as e:
                    print(f"[ToolCallError] {e}")
                    messages.append({
                        "role": "user",
                        "content": TOOL_ERROR_MSG + "/no_think",
                        "guide_content": ""
                    })
            else:
                # No tool call found despite expecting one
                messages.append({
                    "role": "user",
                    "content": TOOL_ERROR_MSG + "/no_think",
                    "guide_content": ""
                })

            token_count = self.count_tokens(messages)
            print(f"[Round {round_num} End] Rollout={roll_out_count} QuestionID={question_id} Tokens={token_count}")

        # ── Final answer extraction ─────────────────────────────────────────
        last_content = messages[-1]['content']
        if "<answer>" in last_content:
            after = last_content.split("<answer>", 1)[1]
            prediction = after.split("</answer>", 1)[0].strip() if "</answer>" in after else after.strip()
            termination = "answer"
        else:
            prediction = "[Failed]"
            termination = 'exceed available llm calls' if num_llm_calls_available == 0 else 'answer not found'

        print("===== Run completed =====")
        return self._build_result(question, answer, data, messages, prediction, termination, start_time)

