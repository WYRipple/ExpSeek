# step2_generate_exp.py
# Input : {exp_kb_dir}/pair.jsonl
# Output: {exp_kb_dir}/pair-EXP.jsonl

import os
import json
import argparse
import ast
from tqdm import tqdm
from multiprocessing import Pool
from openai import OpenAI
import time


# ── Prompts ───────────────────────────────────────────────────────────────────

EXP_PROMPT = """
# Questions for Students to Solve 
{question} 

# Standard Answer for the Question 
{answer} 

# This is a complete trajectory that ultimately got the correct answer as your reference:
```{true_traj}```

# This is a complete trajectory that ultimately got the wrong answer:
```{false_traj}```

# Pattern of All Trajectories: 
Question, R1, O1, R2, O2, R3, O3, R4...

# Instructions and Your Task 
1. Define a STEP as R_i+O_i, but the last STEP only has R_N 
2. Each R is a student's response, attempting to call tools to further solve the problem, but the second trajectory with wrong answer always has some issues 
3. Your core task is to answer this question for each STEP: ```In order to avoid the final error, if guidance is provided after this STEP ends, what should be done to make the agent perform better?```
4. Of course, a complete guidance is a triplet <student's current state, reason why this STEP leads to the final error, what to say before the next STEP to improve the current state>
```Explanation of the triplet: 
    - Student's current state: A relatively general description, introducing what the student saw and what they did. The description does not involve error attribution, focuses on objectively stating the student's behavior, and does not evaluate whether the behavior is good/bad or right/wrong 
    - Reason why this STEP leads to the final error: Unlike the current state, this part explicitly points out what mistake the student made in this STEP 
    - What to say before the next STEP to improve the current state: Based on the errors mentioned above, provide specific guidance that will help the student perform better in the next STEP if they follow it. Of course, do not directly tell the student the answer!
```
5. The guidance opinion in the triplet generated for STEP_i will be concatenated after O_i, which means the student can see it before generating R_i+1 
6. Not every STEP necessarily needs guidance, you can skip after analysis, but since the trajectory is wrong, **there must be at least one STEP that has issues and can be summarized into a triplet** 
7. Finally, briefly summarize what three good pieces of advice could be given before working on this problem 
8. **!!Must Note!!** The total number of rounds you analyze in the trajectory is **{step_num}**, you must generate the corresponding number of STEPs before you can continue to generate TOTAL!

# Output Format (strictly follow the markdown format I give you)
```
# STEP 1: 
## Analysis 
- Write analysis content here 
## Triplet 
(If there is no error, directly write "- None", do not generate a triplet when there is no error) 
- Student's current state: Write current state here 
- Reason why this STEP leads to the final error: Write reason here 
- What to say before the next STEP to improve the current state: Write guidance here 

# STEP 2: 
...

# TOTAL:
## Analysis
- Write analysis here
## Overall Advice
- Overall advice 1
- Overall advice 2
- Overall advice 3
```
"""

DICT_PROMPT = """
# Overall Goal
- You need to convert an input with markdown into a standard format of a list of dicts with tuples
- The input markdown always contains several STEPs and one TOTAL. The list will store dicts for STEPs that have error attribution triples, as well as a dict for the overall attribution

# Example
## Input
```
# STEP 1:
## Analysis
- The student correctly identified the two key pieces of information required by the problem in the first step: the flag presenter and the ceremony date, and reasonably used the `search` tool, setting three highly relevant and comprehensive keyword combinations (flag-presenting ceremony, event date, flag presenter), laying a good foundation for subsequently obtaining complete information. The search strategy is systematic and can effectively locate target web pages.

## Triples
- None

# STEP 2:
## Analysis
- The search results have already provided clear answer clues: the first link (article/2320) appeared repeatedly across multiple queries, with a title directly corresponding to the event, and the abstract containing information related to "July 7th" and "Deputy Secretary Chen Zhengyu". However, in the error trajectory, although the student saw the same results, they failed to dig deeper into key personnel information, only answering based on the general title of "Deputy Party Secretary" without noticing that the specific name appeared in another search result.

## Triples
- Student's current status: Has obtained multiple relevant web page results through search, with particular focus on article links from the School of Computer Science at Sun Yat-sen University's official website, but has not yet clicked to visit or carefully compared details across different results.
- The reason this STEP led to the final error: The student was satisfied with preliminary information (such as "Deputy Party Secretary"), without realizing the need to cross-reference multiple search results to extract a complete and accurate answer, particularly ignoring that the specific name "Chen Zhengyu" had already appeared in the first entry of the third group of search results.
- What to say before the next STEP to improve the current situation: "Note: Results returned by different search keywords may contain complementary information. Please pay special attention to specific names mentioned in search results related to 'flag presenter', combine with date information for verification, and ensure the answer is accurate to an individual rather than just a title."

# TOTAL:
## Analysis
- Comparison between the correct and error trajectories shows that the two are nearly identical in the search phase, with differences appearing in the information integration stage. The core failure of the error trajectory lies in: the student did not fully interpret all search results, particularly ignoring that the key name "Chen Zhengyu" had already been clearly listed in the "flag presenter" search results; and mistakenly treated "Deputy Party Secretary" as the default flag presenter without verifying the specific identity.

## Overall Comments
- After obtaining multiple search results, each result's unique information should be carefully examined one by one, especially when different keywords return content with different emphases, cross-validation is required.
- Avoid using general titles (such as "Deputy Party Secretary") instead of specific names when answering; if a specific name appears in the results, it should be prioritized and its match with the role and event confirmed.
- For dual-element questions involving person + time, it is recommended to separately verify whether the sources for both elements are reliable, and try to simultaneously verify both from the same authoritative source to improve answer consistency.
```

## Output
```
[
{{'STEP-EXP': [
        {{
            'step-id': 1, 
            'step-exp': []
        }},
        {{
            'step-id': 2, 
            'step-exp': [
                'Has obtained multiple relevant web page results through search, with particular focus on article links from the School of Computer Science at Sun Yat-sen University\'s official website, but has not yet clicked to visit or carefully compared details across different results.',
                'The student was satisfied with preliminary information (such as "Deputy Party Secretary"), without realizing the need to cross-reference multiple search results to extract a complete and accurate answer, particularly ignoring that the specific name "Chen Zhengyu" had already appeared in the first entry of the third group of search results.',
                'Note: Results returned by different search keywords may contain complementary information. Please pay special attention to specific names mentioned in search results related to \'flag presenter\', combine with date information for verification, and ensure the answer is accurate to an individual rather than just a title.',
            ]
        }}
    ]
}},
{{'TOTAL-EXP':
    [
        'After obtaining multiple search results, each result\'s unique information should be carefully examined one by one, especially when different keywords return content with different emphases, cross-validation is required.',
        'Avoid using general titles (such as "Deputy Party Secretary") instead of specific names when answering; if a specific name appears in the results, it should be prioritized and its match with the role and event confirmed.',
        'For dual-element questions involving person + time, it is recommended to separately verify whether the sources for both elements are reliable, and try to simultaneously verify both from the same authoritative source to improve answer consistency.',
    ]
}}
]
```

# Notes
1. Please note that if a certain STEP has no errors, an empty step-exp must be generated; generating triples when there are no errors is absolutely not allowed
2. Once again, it is not allowed to generate triples when there are no errors

# The markdown you need to process
{raw_exp}

# Begin Output:
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


def append_jsonl(path, item):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_finished_questions(path):
    if not os.path.exists(path):
        return set()
    finished = set()
    for item in load_jsonl(path):
        q = item.get("question")
        if q:
            finished.add(q)
    return finished


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
            print(f"[Step2] API error (attempt {attempt+1}/{max_tries}): {e}")
    return "[ERROR]"


# ── Worker ────────────────────────────────────────────────────────────────────

def process_sample(args):
    sample, output_path, api_key, api_base, model = args

    step_num = len(sample["token_entropy_avg_list"][0])

    exp_prompt = EXP_PROMPT.format(
        question=sample["question"],
        answer=sample["answer"],
        true_traj=str(sample["messages_list"][1][1:]),
        false_traj=str(sample["messages_list"][0][1:]),
        step_num=step_num,
    )

    for attempt in range(10):
        try:
            raw_exp = call_server(
                [{"role": "system", "content": "You are a helpful assistant."},
                 {"role": "user",   "content": exp_prompt}],
                api_key, api_base, model,
            )
            if raw_exp == "[ERROR]":
                raise ValueError("API returned [ERROR] on EXP prompt")

            dict_prompt = DICT_PROMPT.format(raw_exp=raw_exp)
            dict_raw = call_server(
                [{"role": "system", "content": "You are a helpful assistant."},
                 {"role": "user",   "content": dict_prompt}],
                api_key, api_base, model,
            )
            if dict_raw == "[ERROR]":
                raise ValueError("API returned [ERROR] on DICT prompt")

            dict_result = ast.literal_eval(dict_raw.strip("`").strip())

            if len(dict_result[0]["STEP-EXP"]) != step_num:
                raise ValueError(
                    f"STEP count mismatch: expected {step_num}, "
                    f"got {len(dict_result[0]['STEP-EXP'])}"
                )

            sample["raw_exp"]  = raw_exp
            sample["dict_exp"] = dict_result
            append_jsonl(output_path, sample)
            return True

        except Exception as e:
            print(f"[Step2] Error on attempt {attempt+1}/10: {e}")
            continue

    print(f"[Step2] Failed after 10 attempts: {sample['question'][:60]}")
    return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step2: Generate step-level experience via LLM."
    )
    parser.add_argument("--exp_kb_dir",  type=str, required=True,
                        help="Experience KB directory (contains pair.jsonl).")
    parser.add_argument("--api_key",     type=str, required=True)
    parser.add_argument("--api_base",    type=str, required=True)
    parser.add_argument("--model",       type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=25)
    args = parser.parse_args()

    input_path  = os.path.join(args.exp_kb_dir, "pair.jsonl")
    output_path = os.path.join(args.exp_kb_dir, "pair-EXP.jsonl")

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"pair.jsonl not found in {args.exp_kb_dir}. "
            "Please run step1 first."
        )

    all_samples  = load_jsonl(input_path)
    finished_qs  = load_finished_questions(output_path)
    todo_samples = [s for s in all_samples
                    if s.get("question") not in finished_qs]

    print(f"[Step2] Total pairs  : {len(all_samples)}")
    print(f"[Step2] Already done : {len(finished_qs)}")
    print(f"[Step2] To process   : {len(todo_samples)}")

    if not todo_samples:
        print("[Step2] All done, skipping.")
        return

    tasks = [
        (sample, output_path, args.api_key, args.api_base, args.model)
        for sample in todo_samples
    ]

    with Pool(processes=args.num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_sample, tasks),
            total=len(tasks),
            desc="[Step2] Generating experience",
        ))

    success = sum(results)
    print(f"[Step2] Finished: {success}/{len(tasks)} succeeded.")
    print(f"[Step2] Saved to: {output_path}")


if __name__ == "__main__":
    main()
