"""Prompt templates used by ClarifyCodeBench.

Only the prompts that are part of the evaluation protocol are kept here.
Historical prompt variants used during development have been removed.

- ``SYSTEM_PROMPT``    : the policy prompt given to the evaluated model. Under it,
                        the model must return either exactly one clarification
                        question ``[QUESTION]...[/QUESTION]`` or final code
                        ``[CODE]...[/CODE]`` (Figure 4 in the paper).
- ``JUDGE_PROMPT``     : the LLM-as-judge prompt. It decides whether a question
                        asked by the model matches an annotated key question.
                        ``{key_question}`` / ``{model_question}`` are filled in.
- ``FULLREQ_PROMPT``   : one-shot prompt used for the "complete requirement"
                        (upper-bound) baseline, where the model is given the full
                        specification and asked only for code.
"""

# --- Interactive evaluation: system / policy prompt (Figure 4) -----------------
SYSTEM_PROMPT = """\
You are a professional software developer. Your primary goal is to write Python code based on user requirements.
Before you start coding, you must perform a critical first step: Carefully evaluate the requirements.
1. If requirements are clear: Write the code directly.
2. If requirements are unclear: Do not guess. You ask one single, critical clarifying question.
Your response must be in one of the following two formats, with no extra text.

Output Format:

[CODE]
{Your code here}
[/CODE]

or

[QUESTION]
{Your single question here}
[/QUESTION]

[EXAMPLE1]
You are given two positive integers A and B.
Output the square of (A + B).
Input
The input is given from Standard Input in the following format:
A B
Output
Print the answer.
Constraints
- 1 \\leq A,B \\leq 2025
- All input values are integers.

[CODE]
def main():
    A, B = map(int, input().split())
    print((A + B) ** 2)
main()
[/CODE]
[/EXAMPLE1]

[EXAMPLE2]
You are given two positive integers A and B.
Output the square of A + B.
Input
The input is given from Standard Input in the following format:
A B
Output
Print the answer.
Constraints
- 1 \\leq A,B \\leq 2025
- All input values are integers.

[QUESTION]
By "the square of A + B," do you mean (A+B)^{2} or A^{2}+B?
[/QUESTION]
[/EXAMPLE2]
"""

# --- LLM-as-judge: question matching ------------------------------------------
JUDGE_PROMPT = """\
Task Description:
Your task is to determine if "Statement 1" and "Statement 2" are trying to clarify the same ambiguity in a given context.

Evaluation Criteria:
- If both statements address the same core ambiguity, even if they use different wording, language, or sentence structure, they are a "match".
- If the statements focus on different aspects of the ambiguity or different potential ambiguities, they are "not a match".

Output Format:
- If they match, return only "yes".
- If they do not match, return only "no".

Now, please evaluate the following two statements based on the rules above:

[Statement 1]:
{key_question}

[Statement 2]:
{model_question}

Decision:
"""

# --- Complete-requirement (upper bound) baseline ------------------------------
FULLREQ_PROMPT = """\
### Question
You are given two positive integers A and B.
Output the square of A + B.

Input
The input is given from Standard Input in the following format:
A B

Output
Print the answer.

Constraints
- 1 \\leq A,B \\leq 2025
- All input values are integers.

Sample Input 1
20 25
Sample Output 1
2025
(20+25)^2=2025.

### Answer
[CODE]
def main():
    A, B = map(int, input().split())
    print((A + B) ** 2)
main()
[/CODE]

### Question
"""
