# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import asyncio
import os
import re
import random
import sys
from verl.utils.reward_score.countdown_judge import judge_countdown
MAX_CONCURRENT_CALLS = 2048

client = None # TODO: Your model judge client
print(f"[Model Judge] Using max {MAX_CONCURRENT_CALLS} concurrent API calls")

JUDGE_PROMPT = """Please judge if the student's answer is equivalent to the ground truth answer:
### Ground truth answer: {ground_truth}
### Student's answer: {solution_str}
Please provide your feedback in \\boxed{{...}}. If it is correct, return \\boxed{{True}}, otherwise return \\boxed{{False}}
"""

async def model_judge_call(solution_str, ground_truth, task):
    """Async function to call the model judge API"""
    if 'dapo' in task:
        return 0.0
    try:
        # Extract student answer from boxed format if present
        # student_answer = re.search(r'\\boxed\{([^}]+)\}', solution_str)
        # print('task: ', task)
        if not '\\boxed{' in solution_str:
            return 0.0
        if 'count' in task.lower():
            is_correct, reason = judge_countdown(ground_truth, solution_str)
            return 1.0 if is_correct else 0.0
        student_answer = solution_str.split('\\boxed{')[-1]
        if not '}' in student_answer:
            return 0.0
        idx = student_answer.rfind('}')
        student_answer = student_answer[:idx]

        if not student_answer or len(student_answer) > 200:
            # If no boxed format, try the whole string
            return 0.0

        prompt = JUDGE_PROMPT.format(ground_truth=ground_truth, solution_str=student_answer)
        # print(f'[Judge Request] student: {student_answer}, ground_truth: {ground_truth}')

        # Direct async API call
        response = await asyncio.wait_for(
            client.chat_completions(
                model=None, # TODO: Your model name
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.1,
                n=1
            ),
            timeout=3600.0  # 120 second timeout
        )

        # Extract the model's response
        model_output = response['choices'][0]['message']['content']

        # Parse the boxed response to extract True/False
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', model_output)
        if boxed_match:
            result = boxed_match.group(1).strip().lower()
            return 1.0 if result == "true" else 0.0
        else:
            # If no boxed response, try to find True/False in the text
            return 1.0 if "true" in model_output.lower() else 0.0

    except asyncio.TimeoutError:
        print(f"[Timeout] Model judge timed out after 60s")
        return 0.0
    except Exception as e:
        print(f"[Error] Model judge error: {e}")
        return 0.0

async def compute_score(task, solution_str, ground_truth, extra_info, **kwargs) -> float:
    """
    Async compute score function compatible with the updated PrimeRewardManager.
    Uses both rule-based matching and model judge.
    """
    if 'wrong' in task:
        return 1.0 if random.random() < 0.5 else 0.0
    # First try rule-based matching for efficiency
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
            if is_equiv(answer, ground_truth):
                # Rule-based match succeeded, no need for model judge
                # print(f"[Rule Match] {answer} == {ground_truth}")
                return 1.0
    except Exception as e:
        print(f"Rule-based matching error: {e}")

    # Use async model judge
    try:
        score = await model_judge_call(solution_str, ground_truth, task)
        return score
    except Exception as e:
        print(f"Model judge failed: {e}")
        return 0.0


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:  # noqa: E722
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:  # noqa: E722
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1).
    # Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string
