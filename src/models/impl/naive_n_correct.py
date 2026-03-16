import re
from openai import OpenAI
from src.models.joint import JointModel
from src.models.misconception import MisconceptionModel
from src.models.simulation import SimulateModel
from src.prompt_util import prompt_deepseek, prompt_openai


def prompt_joint_naive(context: dict[str,str], n: int) -> dict[str,str]:
    question = context["Problem"]["Question"]
    answer = context["Problem"]["Answer"]

    return {
        "system": f"""You will be given a math question along with the correct answer. Please generate {n} incorrect distractor answers for the question to be used as multiple-choice options in a multiple-choice exam.
[Template]
Distractor1:
...
Distractor{n}:
""",
    "user": f"""Question: {question}
Answer:{answer}"""
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

class DeepseekNaiveCorrectJointModel(JointModel):
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