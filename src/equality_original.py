import re
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from openai import OpenAI

from src.prompt_util import prompt_openai

class EqualityChecker(ABC):
    """
    Abstract base class for equality checkers.
    Subclasses must implement is_equal(problem, answer_a, answer_b) -> bool.
    """
    @abstractmethod
    def is_equal(self, problem: str, answer_a: str, answer_b: str) -> bool:
        raise NotImplementedError()


def judge_answer_equality_llm(client: OpenAI, model_config: dict, problem: str, answer_a: str, answer_b: str) -> bool:
        system_prompt = """You are an AI assistant tasked with judging whether two answer choices to a middle school multiple choice math problem are semantically the same as one another, regardless of their factual accuracy, completeness or correctness. Your answer should be solely based on whether the answers are semantically equivalent or not, like in the examples below:

- Output 'same' if the answers represent the same concept or value, even if formatted differently, written in words vs numbers, or include minor extra words like "only" or "I agree with".
- Output 'different' if the answers are semantically different from one another, or if one answer is empty and the other is not.
- Do not solve the problem; Do not evalute factual correctness of the answers; Only compare the provided answers with one another, regardless of their factual accuracy.
- Ignore formatting differences, LaTeX wrappers, or capitalization
- 10, 10.0, and \(10\) are considered the same in most cases except if the question asks for a specific format. For example, 4.5 x 10^3 is not equivalent to 4500 if the question asks for scientific format.
- Note that two incorrect answers can be the same if they are incorrect in the same way

Format your output exactly as:
<judgement> [same/different] </judgement>

Examples:

<math problem> <math problem> Problem P: \( 4-5+6= \) Problem Q: \( 4+5 \times 6= \) Which calculation should you do first in each problem? </math problem> </math problem>
<answer_1> Problem P: \( 4-5 \) Problem Q: \( 4+5 \) </answer_1>
<answer_2> In Problem P, do 4 minus 5 first. In Problem Q, do 4 plus 5 first. </answer_2>
<judgement> same </judgement>

<math problem> 50.09 ÷ 0.1 = </math problem>
<answer_1> 500.09 </answer_1>
<answer_2> 500.9 </answer_2>
<judgement> different </judgement>

<math problem> Convert this fraction to a percentage 4/5 </math problem>
<answer_1> 45% </answer_1>
<answer_2> 45 </answer_2>
<judgement> same </judgement>

<math problem> What is 120% of 50? </math problem>
<answer_1> 10 </answer_1>
<answer_2> 60 </answer_2>
<judgement> different </judgement>

<math problem> Tom and Katie are arguing about parallelograms. Tom says this shape is a parallelogram ![A four sided shape. All sides are equal, opposite angles are equal. There are no right angles.]() Katie says this shape is a parallelogram ![A four sided shape. Opposite sides are equal, all angles are right angles.]() Who is correct? </math problem>
<answer_1> Only Tom </answer_1>
<answer_2> I agree with Tom. </answer_2>
<judgement> same </judgement>

<math problem> Complete this statement:
\( 5 \) litres \( = \) ________\( \mathrm{cm}^{3} \) </math problem>
<answer_1> \( 5 \) </answer_1>
<answer_2> 5 </answer_2>
<judgement> same </judgement>
"""
        user_prompt = f"""<math problem> {problem} </math problem>
<answer_1> {answer_a} </answer_1>
<answer_2> {answer_b} </answer_2>
"""
        response = prompt_openai(client=client, system_prompt=system_prompt, user_prompt=user_prompt, model_config=model_config)
    
        judgements = re.findall(r"<judgement>\s*(.*?)\s*</judgement>", response)
        if len(judgements) > 1:
            print(f"Warning, got multiple judgments when comparing answers, will pick the first one! {response}")
        
        judgement = judgements[0].strip().lower()
        if judgement not in {"same", "different"}:
            print(f"Warning, got unexpected judgment {judgement}, will resort to False! {response}")
        if judgement == "same":
            return True
        return False

class SemanticEqualityChecker(EqualityChecker):
    """
    Semantic checker: uses LLM-based judgement if not trivially equivalent
    """
    def __init__(self, client: OpenAI, model_config: dict[str,str]):
        self.client = client
        self.model_config = model_config
        self.log: list[dict[str, Any]] = []
        self.memoization: dict[tuple, bool] = {}


    def is_equal(self, problem: str, answer_a: str, answer_b: str) -> bool:
        cached = self.memoization.get((problem, answer_a, answer_b),
                                      self.memoization.get((problem, answer_b, answer_a), None))
        if cached is not None:
            return cached
            
        judgement = self.judge_answer_equality(problem, answer_a, answer_b)
        self.log.append({**judgement, "problem": problem, "answer_a": answer_a, "answer_b": answer_b})
        is_match = judgement.get("match", False)
        self.memoization[(problem, answer_a, answer_b)] = is_match
        return is_match

    def judge_answer_equality(self, problem: str, answer_a: Optional[str], answer_b: Optional[str]) -> Dict[str, Any]:
        """
        Judge whether two answers A and B are semantically equivalent given problem using the strategy:
        0. if one of the answers is none/empty and the other is not => no match
        1. exact matches => equal
        2. ask llm to judge (via _judge_answer_equality_llm)
        """
        if bool(len((answer_a or "").strip())) ^ bool(len((answer_b or "").strip())):
            # one of the answers is empty, the other is not
            return {"match": False, "reason": "empty"}

        if answer_a.lower().strip() == answer_b.lower().strip():
            return {"match": True, "reason": "exact_match"}

        # fallback to LLM judgement
        try:
            if judge_answer_equality_llm(self.client, self.model_config, problem, answer_a or "", answer_b or ""):
                return {"match": True, "reason": "llm_match"}
        except Exception as e:
            # If LLM fails, log and continue to return no_match
            print(e)
            return {"match": False, "reason": f"llm_error:{e}"}

        return {"match": False, "reason": "no_match"}


class SemanticDoubleEqualityChecker(SemanticEqualityChecker):
    """
    Semantic checker: uses LLM-based judgement, double check with a more competent model in case the trivial model reports equivalence (bc it's not always given)
    """
    def __init__(self, client: OpenAI, model_config: dict[str,str], expert_model_config: dict[str,str]):
        super().__init__(client, model_config)
        self.expert_model_config = expert_model_config

    def judge_answer_equality(self, problem: str, answer_a: Optional[str], answer_b: Optional[str]) -> Dict[str, Any]:
        """
        Judge whether two answers A and B are semantically equivalent given problem using the strategy:
        0. if one of the answers is none/empty and the other is not => no match
        1. exact matches => equal
        2. ask llm to judge (via _judge_answer_equality_llm)
        """
        if bool(len((answer_a or "").strip())) ^ bool(len((answer_b or "").strip())):
            # one of the answers is empty, the other is not
            return {"match": False, "reason": "empty"}

        if answer_a.lower().strip() == answer_b.lower().strip():
            return {"match": True, "reason": "exact_match"}

        # fallback to LLM judgement
        try:
            if judge_answer_equality_llm(self.client, self.model_config, problem, answer_a or "", answer_b or ""):
                if judge_answer_equality_llm(self.client, self.expert_model_config, problem, answer_a or "", answer_b or ""):
                    return {"match": True, "reason": "llm_match"}
        except Exception as e:
            # If LLM fails, log and continue to return no_match
            print(e)
            return {"match": False, "reason": f"llm_error:{e}"}

        return {"match": False, "reason": "no_match"}


class NumericalEqualityChecker(EqualityChecker):
    """
    Numerical checker: extracts the first whole number (integer) from each answer string
    and compares integer equality. Returns False if no integer found in either answer.
    """

    integer_regex = re.compile(r"[-+]?\d+")

    def first_int(self, s: str):
        if not isinstance(s, str):
            return None
        m = self.integer_regex.search(s)
        if not m:
            return None
        try:
            return int(m.group(0))
        except Exception:
            return None

    def is_equal(self, problem: str, answer_a: str, answer_b: str) -> bool:
        a_int = self.first_int(answer_a or "")
        b_int = self.first_int(answer_b or "")

        if a_int is None or b_int is None:
            judgement = {"match": False, "reason": "no_int_found"}
        else:
            match = (a_int == b_int)
            judgement = {"match": match, "reason": "int_match" if match else "int_mismatch", "a_int": a_int, "b_int": b_int}
            
        is_match = judgement["match"]
        return is_match