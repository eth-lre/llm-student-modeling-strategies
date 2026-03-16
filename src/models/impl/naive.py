import re
from openai import OpenAI
from src.models.joint import JointModel
from src.models.simulation import SimulateModel
from src.prompt_util import prompt_deepseek, prompt_openai, prompt_openrouter


def prompt_joint_naive(context: dict[str,str], n: int) -> dict[str,str]:
    question = context["Problem"]["Question"]

    return {
        "system": f"""You will be given a math question. Please generate {n} incorrect distractor answers for the question to be used as multiple-choice options in a multiple-choice exam.
[Template]
Distractor1:
...
Distractor{n}:
""",
    "user": f"""Question: {question}"""
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

def prompt_m_sim_naive(context: dict[str,str], misconception: str) -> dict[str,str]:
    question = context["Problem"]["Question"]

    return {
        "system": f"""You will be given a math question and specific student error. Please generate the incorrect answer that a student would give on the current question if they made the specified error. 
At the end, give the student's final concise answer preceded with 'Incorrect Student Answer:'""",
    "user": f"""Question: {question}
Student Error: {misconception}"""
    }

def prompt_m_sim_naive_direct(context: dict[str,str], misconception: str) -> dict[str,str]:
    question = context["Problem"]["Question"]

    return {
        "system": f"""You will be given a math question and specific student error. Please generate the incorrect answer that a student would give on the current question if they made the specified error. 
Onlyoutput the student's final concise answer preceded with 'Incorrect Student Answer:'""",
    "user": f"""Question: {question}
Student Error: {misconception}"""
    }


class OpenAINaiveJointModel(JointModel):
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

class OpenAINaiveSimulateModel(SimulateModel):
    def __init__(self, client: OpenAI, model_config: dict[str, str]):
        self.client = client
        self.model_config = model_config

    def simulate(self, context: dict[str, str], misconception: str) -> tuple[str,dict[str,str]]:
        prompts = prompt_m_sim_naive(context, misconception)
        system_prompt = prompts.get("system", "")
        user_prompt = prompts.get("user", "")
        response = prompt_openai(self.client, system_prompt, user_prompt, self.model_config)
        text = (response or "").strip()

        # Prefer explicit labelled output
        m = re.search(r"Incorrect Student Answer:\s*(.*?)(?=(?:\n[AA-Za-z][^\n:]{0,40}:)|\Z)", text, re.DOTALL)
        if m:
            answer = m.group(1).strip()
        else:
            # fallback heuristics: try single-line "Answer:" or return whole text
            m2 = re.search(r"(?:Answer|Incorrect Student Answer|Student Answer)[:\-]\s*(.*)", text)
            answer = (m2.group(1).strip() if m2 else text).strip()

        return answer, {"raw": response, "answer": answer}


class DeepseekNaiveJointModel(JointModel):
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
    
class DeepseekNaiveSimulateModel(SimulateModel):
    def __init__(self, model_config: dict[str, str]):
        self.model_config = model_config

    def simulate(self, context: dict[str, str], misconception: str) -> tuple[str,dict[str,str]]:
        prompts = prompt_m_sim_naive(context, misconception)
        system_prompt = prompts.get("system", "")
        user_prompt = prompts.get("user", "")
        reasoning,response = prompt_deepseek(system_prompt, user_prompt, self.model_config)
        text = (response or "").strip()

        # Prefer explicit labelled output
        m = re.search(r"Incorrect Student Answer:\s*(.*?)(?=(?:\n[AA-Za-z][^\n:]{0,40}:)|\Z)", text, re.DOTALL)
        if m:
            answer = m.group(1).strip()
        else:
            # fallback heuristics: try single-line "Answer:" or return whole text
            m2 = re.search(r"(?:Answer|Incorrect Student Answer|Student Answer)[:\-]\s*(.*)", text)
            answer = (m2.group(1).strip() if m2 else text).strip()

        return answer, {"raw": response, "answer": answer, "raw_reasoning": reasoning}
    
class DeepseekNaiveDirectSimulateModel(SimulateModel):
    def __init__(self, model_config: dict[str, str]):
        self.model_config = model_config

    def simulate(self, context: dict[str, str], misconception: str) -> tuple[str,dict[str,str]]:
        prompts = prompt_m_sim_naive(context, misconception)
        system_prompt = prompts.get("system", "")
        user_prompt = prompts.get("user", "")
        reasoning,response = prompt_deepseek(system_prompt, user_prompt, self.model_config)
        text = (response or "").strip()

        # Prefer explicit labelled output
        m = re.search(r"Incorrect Student Answer:\s*(.*?)(?=(?:\n[AA-Za-z][^\n:]{0,40}:)|\Z)", text, re.DOTALL)
        if m:
            answer = m.group(1).strip()
        else:
            # fallback heuristics: try single-line "Answer:" or return whole text
            m2 = re.search(r"(?:Answer|Incorrect Student Answer|Student Answer)[:\-]\s*(.*)", text)
            answer = (m2.group(1).strip() if m2 else text).strip()

        return answer, {"raw": response, "answer": answer, "raw_reasoning": reasoning}


class OpenRouterNaiveJointModel(JointModel):
    """Model using OpenRouter API to propose a list of distractors with the simple possible prompt"""

    def __init__(self, client: OpenAI, model_config: dict[str, str]):
        """
        Args:
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


class OpenRouterNaiveSimulateModel(SimulateModel):
    """Model using OpenRouter API to simulate student misconceptions"""

    def __init__(self, model_config: dict[str, str]):
        """
        Args:
            model_config: Dictionary with OpenRouter model configuration (api_key, model, temperature, etc.)
        """
        self.model_config = model_config

    def simulate(self, context: dict[str, str], misconception: str) -> tuple[str,dict[str,str]]:
        prompts = prompt_m_sim_naive(context, misconception)
        system_prompt = prompts.get("system", "")
        user_prompt = prompts.get("user", "")
        reasoning, response = prompt_openrouter(system_prompt, user_prompt, self.model_config)
        text = (response or "").strip()

        # Prefer explicit labelled output
        m = re.search(r"Incorrect Student Answer:\s*(.*?)(?=(?:\n[AA-Za-z][^\n:]{0,40}:)|\Z)", text, re.DOTALL)
        if m:
            answer = m.group(1).strip()
        else:
            # fallback heuristics: try single-line "Answer:" or return whole text
            m2 = re.search(r"(?:Answer|Incorrect Student Answer|Student Answer)[:\-]\s*(.*)", text)
            answer = (m2.group(1).strip() if m2 else text).strip()

        return answer, {"raw": response, "answer": answer, "raw_reasoning": reasoning}
