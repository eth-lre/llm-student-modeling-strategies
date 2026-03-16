import re
from openai import OpenAI
from src.models.joint import JointModel
from src.prompt_util import prompt_deepseek, prompt_openai, prompt_openrouter


def prompt_joint_naive(context: dict[str,str], n: int) -> dict[str,str]:
    question = context["Problem"]["Question"]

    return {
        "system": f"""You will be given a math question. Please generate {n} incorrect distractor answers for the question to be used as multiple-choice options in a multiple-choice exam.
Think step-by-step before giving your final answer. Output only your step-by-step reasoning and the final distractors like so:

[Step-By-Step]
Let's think step-by-step, ...

[Final Answer]
Distractor1: ...
...
Distractor{n}: ...
""",
    "user": f"""Question: {question}"""
    }

def parse_joint_output(text: str) -> dict[str, str]:
    results: dict[str, str] = {
        "raw": text,
        "step_by_step": None,
        "statistics": {}
    }

    if not text or not isinstance(text, str):
        return results

    text = text.strip()

    step_pattern = re.compile(
        r"\[Step-By-Step\](.*?)(?=\[Final Answer\]|\Z)",
        re.DOTALL | re.IGNORECASE
    )

    final_pattern = re.compile(
        r"\[Final Answer\](.*)",
        re.DOTALL | re.IGNORECASE
    )

    step_match = step_pattern.search(text)
    final_match = final_pattern.search(text)


    if step_match:
        results["step_by_step"] = step_match.group(1).strip()
    else:
        # Fallback: if no explicit step-by-step section,
        # but final answer exists, treat everything before it as reasoning
        if final_match:
            results["step_by_step"] = text[:final_match.start()].strip()

    if final_match:
        final_text = final_match.group(1)
    else:
        # Fallback: no final header → assume entire text may contain answers
        final_text = text


    distractor_pattern = re.compile(
        r"Distractor\s*(\d+)\s*:\s*(.+)",
        re.IGNORECASE
    )

    for idx, ans in distractor_pattern.findall(final_text):
        key = f"distractor{idx}_answer"
        results[key] = ans.strip()

    return results


class OpenAINaiveCoTJointModel(JointModel):
    """Model using OpenAI API to propose a list of distractors with the simple possible prompt"""

    def __init__(self, client: OpenAI, model_config: dict[str, str]):
        """
        Args:
            client: OpenAI client instance.
            model_config: Dictionary with OpenAI model configuration (e.g., model name, temperature)
        """
        self.client = client
        self.model_config = model_config

    def generate_distractors(
        self,
        context: dict[str, str],
        num_distractors: int
    ) -> tuple[list[str],dict[str,str]]:
        """
        Proposes a new list of distractors based on the given problem and reasoning.
        Returns (misconception, parsed_response_dict).
        """
        prompts = prompt_joint_naive(context, num_distractors)

        system_prompt = prompts.get("system")
        user_prompt = prompts.get("user")

        if not system_prompt or not user_prompt:
            raise ValueError(
                f"Prompt function did not produce valid 'system' and 'user' prompts "
                f"for context{context}"
            )

        response = prompt_openai(self.client, system_prompt, user_prompt, self.model_config)
        full_parsed_response = parse_joint_output(response)

        return [v for k,v in full_parsed_response.items() if k.endswith("_answer")], full_parsed_response

class DeepseekNaiveCoTJointModel(JointModel):
    """Model using Deepseek API to propose a list of distractors with the simple possible prompt"""

    def __init__(self, model_config: dict[str, str]):
        """
        Args:
            model_config: Dictionary with OpenAI model configuration (e.g., model name, temperature).
        """
        self.model_config = model_config

    def generate_distractors(
        self,
        context: dict[str, str],
        num_distractors: int
    ) -> tuple[list[str],dict[str,str]]:
        """
        Proposes a new list of distractors based on the given problem and reasoning.
        Returns (misconception, parsed_response_dict).
        """
        prompts = prompt_joint_naive(context, num_distractors)

        system_prompt = prompts.get("system")
        user_prompt = prompts.get("user")

        if not system_prompt or not user_prompt:
            raise ValueError(
                f"Prompt function did not produce valid 'system' and 'user' prompts "
                f"for context{context}"
            )

        
        reasoning,response = prompt_deepseek(system_prompt, user_prompt, self.model_config)
        full_parsed_response = parse_joint_output(response)

        return [v for k,v in full_parsed_response.items() if k.endswith("_answer")], {
            **full_parsed_response,
            "raw_reasoning": reasoning
        }


class OpenRouterNaiveCoTJointModel(JointModel):
    """Model using OpenRouter API to propose a list of distractors with chain-of-thought reasoning"""

    def __init__(self, client: OpenAI, model_config: dict[str, str]):
        """
        Args:
            client: OpenAI client instance (configured for OpenRouter).
            model_config: Dictionary with OpenRouter model configuration (api_key, model, temperature, etc.)
        """
        self.client = client
        self.model_config = model_config

    def generate_distractors(
        self,
        context: dict[str, str],
        num_distractors: int
    ) -> tuple[list[str],dict[str,str]]:
        """
        Proposes a new list of distractors based on the given problem and reasoning.
        Returns (list of distractors, parsed_response_dict with reasoning).
        """
        prompts = prompt_joint_naive(context, num_distractors)

        system_prompt = prompts.get("system")
        user_prompt = prompts.get("user")

        if not system_prompt or not user_prompt:
            raise ValueError(
                f"Prompt function did not produce valid 'system' and 'user' prompts "
                f"for context{context}"
            )

        reasoning, response = prompt_openrouter(self.client, system_prompt, user_prompt, self.model_config)
        full_parsed_response = parse_joint_output(response)

        return [v for k,v in full_parsed_response.items() if k.endswith("_answer")], {
            **full_parsed_response,
            "raw_reasoning": reasoning
        }