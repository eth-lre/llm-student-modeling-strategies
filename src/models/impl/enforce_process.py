import re
from openai import OpenAI
from src.models.joint import JointModel
from src.prompt_util import prompt_openrouter, prompt_openai, prompt_vllm


def prompt_joint(context: dict[str,str], n: int, show_correct: bool, subject: str = "math") -> dict[str,str]:
    question = context["Problem"]["Question"]
    answer = context["Problem"]["Answer"]

    system_prompt = f"You will be given a {subject} question"
    user_prompt = f"Question: {question}"
    if show_correct:
        user_prompt += f"\nAnswer: {answer}"
        system_prompt += " and the correct answer"
    
    return {
        "system": f"""{system_prompt}. Please generate {n} incorrect distractor answers for the question to be used as multiple-choice options in a multiple-choice exam.

**RULES**
Solve correctly first:
1. First, solve the problem correctly and treat the most likely correct solution as fixed.
2. Identify the key concepts involved in the correct solution.
3. Specify the exact conditions that an answer must violate to be a valid distractor.

Error modeling:
4. Enumerate at most 7 common error primitives relevant to this problem. Each error primitive must be either:
   (a) a buggy rule (an incorrect transformation or procedure), or
   (b) a buggy commitment (a false assumption or misclassification treated as true).
5. Each error primitive must be specific, stable, and capable of producing a concrete deterministic answer.

Error simulation:
6. For each error primitive, assume the student fully commits to that single error and reasons correctly in all other respects.
7. Derive the final incorrect answer that results from that single error and collect it as a distractor candidate.
    - If the question is non-numerical, output the final incorrrect conclusion, classification or choice that follows from the error primitive.

Plausibility Assessment:
8. For each distractor candidate you assess its discriminative power:
    - check if the candidate can be derived deterministically
    - check if the candidate is truly incorrect under any reasonable interpretation
    - quantify how likely students are to stop here and select this candidate
    - make sure it is unambiguous and well-formed

Selection:
9. From the remaining candidates, select the {n} most distinct plausible distractors.

Output rules:
10. You should only output the final concise distractor values in the following template:

[Template]
Distractor1:
...
Distractor{n}:
""",
    "user": user_prompt
    }

def parse_joint_output(text: str) -> dict[str, str]:
        """
        Parse distractor output from the LLM response into a structured dictionary.
        """
        results = {
            "raw": text,
            "statistics": {}
        }
        text = text.strip()

        pattern = r"Distractor\s*(\d+)\s*:\s*(.*)"

        for (idx,ans) in re.findall(pattern, text):
            results[f"distractor{idx}_answer"] = ans.strip()

        return results


class OpenAIEnforceProcessJointModel(JointModel):
    """Model using OpenAI API to propose a list of distractors while enforcing the process from the paper"""

    def __init__(self, client: OpenAI, model_config: dict[str, str], show_correct: bool = False, subject: str = "math"):
        super().__init__(subject)
        self.client = client
        self.model_config = model_config
        self.show_correct = show_correct

    def generate_distractors(
        self,
        context: dict[str, str],
        num_distractors: int
    ) -> tuple[list[str], dict[str, str]]:
        prompts = prompt_joint(context, num_distractors, self.show_correct, self.subject)
        system_prompt = prompts.get("system")
        user_prompt = prompts.get("user")

        if not system_prompt or not user_prompt:
            raise ValueError(
                f"Prompt function did not produce valid 'system' and 'user' prompts "
                f"for context{context}"
            )

        response = prompt_openai(self.client, system_prompt, user_prompt, self.model_config)
        full_parsed_response = parse_joint_output(response)

        return [v for k, v in full_parsed_response.items() if k.endswith("_answer")], full_parsed_response


class DeepseekEnforceProcessJointModel(JointModel):
    """Model using Deepseek API to propose a list of distractors while enforcing the process from the paper"""

    def __init__(self, model_config: dict[str, str], show_correct: bool = False, subject: str = "math"):
        """
        Args:
            model_config: Dictionary with OpenAI model configuration (e.g., model name, temperature).
            subject: Subject area for prompts (e.g., "math", "science")
        """
        super().__init__(subject)
        self.model_config = model_config
        self.show_correct = show_correct

    def generate_distractors(
        self,
        context: dict[str, str],
        num_distractors: int
    ) -> tuple[list[str],dict[str,str]]:
        """
        Proposes a new list of distractors based on the given problem and reasoning.
        Returns (misconception, parsed_response_dict).
        """
        prompts = prompt_joint(context, num_distractors, self.show_correct, self.subject)

        system_prompt = prompts.get("system")
        user_prompt = prompts.get("user")

        if not system_prompt or not user_prompt:
            raise ValueError(
                f"Prompt function did not produce valid 'system' and 'user' prompts "
                f"for context{context}"
            )

        
        reasoning, response = prompt_openrouter(system_prompt, user_prompt, self.model_config, stream=True)
        full_parsed_response = parse_joint_output(response)

        return [v for k,v in full_parsed_response.items() if k.endswith("_answer")], {
            **full_parsed_response,
            "raw_reasoning": reasoning
        }
    

class OpenRouterEnforceProcessJointModel(JointModel):
    """Model using OpenRouter API to propose a list of distractors while enforcing the process from the paper"""

    def __init__(self, model_config: dict[str, str], show_correct: bool = False, subject: str = "math"):
        """
        Args:
            model_config: Dictionary with OpenRouter model configuration (api_key, model, temperature, etc.)
            subject: Subject area for prompts (e.g., "math", "science")
        """
        super().__init__(subject)
        self.model_config = model_config
        self.show_correct = show_correct

    def generate_distractors(
        self,
        context: dict[str, str],
        num_distractors: int,
        stream: bool = False,
    ) -> tuple[list[str],dict[str,str]]:
        """
        Proposes a new list of distractors based on the given problem and reasoning.
        Returns (list of distractors, parsed_response_dict with reasoning).
        """
        prompts = prompt_joint(context, num_distractors, self.show_correct, self.subject)

        system_prompt = prompts.get("system")
        user_prompt = prompts.get("user")

        if not system_prompt or not user_prompt:
            raise ValueError(
                f"Prompt function did not produce valid 'system' and 'user' prompts "
                f"for context{context}"
            )

        reasoning, response = prompt_openrouter(system_prompt, user_prompt, self.model_config, stream=stream)
        full_parsed_response = parse_joint_output(response)

        return [v for k,v in full_parsed_response.items() if k.endswith("_answer")], {
            **full_parsed_response,
            "raw_reasoning": reasoning
        }


class VLLMEnforceProcessJointModel(JointModel):
    """Model using a self-hosted vLLM (OpenAI-compatible) server to propose distractors via the
    learning-science-informed enforce_process prompt. Captures reasoning_content."""

    def __init__(self, client: OpenAI, model_config: dict[str, str], show_correct: bool = False, subject: str = "math"):
        super().__init__(subject)
        self.client = client
        self.model_config = model_config
        self.show_correct = show_correct

    def generate_distractors(
        self,
        context: dict[str, str],
        num_distractors: int
    ) -> tuple[list[str], dict[str, str]]:
        prompts = prompt_joint(context, num_distractors, self.show_correct, self.subject)
        system_prompt = prompts.get("system")
        user_prompt = prompts.get("user")

        if not system_prompt or not user_prompt:
            raise ValueError(
                f"Prompt function did not produce valid 'system' and 'user' prompts "
                f"for context{context}"
            )

        reasoning, response = prompt_vllm(self.client, system_prompt, user_prompt, self.model_config)
        full_parsed_response = parse_joint_output(response)

        return [v for k, v in full_parsed_response.items() if k.endswith("_answer")], {
            **full_parsed_response,
            "raw_reasoning": reasoning,
        }
