import re
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pickle

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


MATH_EQUALITY_SYSTEM_PROMPT = """You are an AI assistant tasked with judging whether two answer choices to a middle-school multiple-choice math problem are semantically the same as one another. You must not solve the problem and not evaluate factual correctness — only compare the two answers with one another relative to the problem’s formatting requirements.

Your output must follow this exact structure:

<format> [TRUE/FALSE] </format>
<equivalent> [TRUE/FALSE] </equivalent>

Meaning of <format>
Output TRUE if the problem explicitly requires a specific numeric format, such as:
- rounding to a given number of decimal places or significant digits
- expressing the answer in scientific notation
- expressing the answer as a simplified fraction
- expressing the answer in terms of a constant (e.g., “in terms of π”)
- any other explicitly stated formatting requirement

Ignore unit requirements (e.g., “in cm” does NOT count as a specific format).

Output FALSE if the problem does not explicitly specify a numeric format.

Meaning of <equivalent>
Determine whether answer_1 and answer_2 represent the same value or concept under the rules:

If <format> is FALSE (no required numeric format):
Two answers are equivalent if:
- they have the same mathematical value (e.g., 3.1 == 31/10)
- they differ only in non-semantic aspects (e.g., LaTeX wrappers, capitalization, filler words)

Examples of equivalent under <format> = FALSE:
- 10, 10.0, \(10\)
- 3.1, 31/10
- fourteen, 14
- Only Bob, Bob

If <format> is TRUE (specific format required):
Two answers are equivalent only if:
1. They represent the same mathematical value, AND
2. They are both expressed in the required format.

This means:
- 3.14 vs. 3.140 (when rounding to 2 decimal places required) → not equivalent
- 4π vs. 12.56 (when “in terms of π” required) → not equivalent
- 3.1 vs. 31/10 (when “round to one decimal place” required) → not equivalent

Ignore trivial formatting wrappers (e.g., 31/10 == \(31/10\)).

General Rules
- Do not solve the problem.
- Do not judge correctness of the answer_1 and answer_2.
- Only compare answer_1 with answer_2.
- answer_1 and answer_2 can be equivalent regardless of whether they are correct or not.
"""


SCIENCE_EQUALITY_SYSTEM_PROMPT = """You are an AI assistant tasked with judging whether two answer choices to a middle-school multiple-choice science question are semantically the same as one another. You must not answer the question and not evaluate factual correctness — only compare the two answers with one another relative to what the question is asking.

Your output must follow this exact structure:

<equivalent> [TRUE/FALSE] </equivalent>

Meaning of <equivalent>
Two answers are equivalent if they refer to exactly the same scientific concept, entity, term, or value, differing only in:
- capitalization, punctuation, pluralization, articles, or filler words
- trivial formatting wrappers (e.g., LaTeX, extra whitespace)
- abbreviation vs. its expansion (e.g., DNA vs. deoxyribonucleic acid)
- well-established synonyms denoting the exact same referent (e.g., heat vs. thermal energy)
- rewording that preserves the exact same referent without adding or removing entities

Examples of equivalent:
- photosynthesis = Photosynthesis
- planet = a planet
- mirrors = a mirror
- DNA = deoxyribonucleic acid
- heat = thermal energy
- products = the products

Two answers are NOT equivalent if:
- they name different scientific concepts, even if closely related (e.g., photosynthesis vs. respiration, mirrors vs. lenses, circadian rhythm vs. circannual rhythm)
- one is a subset or superset of the other, or introduces or removes entities (e.g., "protons and neutrons" vs. "protons")
- they give different specific values or quantities (e.g., "less than half" vs. "5%", "type 1" vs. "type 2 diabetes")
- they specify different levels of generality that change the referent (e.g., "bacteria" vs. "E. coli")

Examples of NOT equivalent:
- mirrors != lenses
- protons and neutrons != protons
- carbon != nitrogen
- heat != ultraviolet

General Rules
- Do not answer the question.
- Do not judge correctness of answer_1 and answer_2.
- Only compare answer_1 with answer_2.
- answer_1 and answer_2 can be equivalent regardless of whether they are correct or not.
"""


def _judge_answer_equality_llm(client: OpenAI, model_config: dict, system_prompt: str, problem: str, answer_a: str, answer_b: str) -> bool:
    user_prompt = f"""<problem> {problem} </problem>
<answer_1> {answer_a} </answer_1>
<answer_2> {answer_b} </answer_2>
"""
    response = prompt_openai(client=client, system_prompt=system_prompt, user_prompt=user_prompt, model_config=model_config)

    equivalences = re.findall(r"<equivalent>\s*(.*?)\s*</equivalent>", response)
    if len(equivalences) > 1:
        print(f"Warning, got multiple judgments when comparing answers, will pick the first one! {response}")

    equivalence = equivalences[0].strip().lower()
    if equivalence not in {"true", "false"}:
        print(f"Warning, got unexpected judgment {equivalence}, will resort to False! {response}")
    return equivalence == "true"


def judge_math_answer_equality_llm(client: OpenAI, model_config: dict, problem: str, answer_a: str, answer_b: str) -> bool:
    return _judge_answer_equality_llm(client, model_config, MATH_EQUALITY_SYSTEM_PROMPT, problem, answer_a, answer_b)


def judge_science_answer_equality_llm(client: OpenAI, model_config: dict, problem: str, answer_a: str, answer_b: str) -> bool:
    return _judge_answer_equality_llm(client, model_config, SCIENCE_EQUALITY_SYSTEM_PROMPT, problem, answer_a, answer_b)


class _SemanticEqualityCheckerBase(EqualityChecker):
    """
    Base class for semantic equality checkers: falls through exact-match shortcuts,
    then defers to an LLM judge with a subject-specific system prompt.
    Subclasses must set SYSTEM_PROMPT.
    """
    SYSTEM_PROMPT: str = ""

    def __init__(self, client: OpenAI, model_config: dict[str, str]):
        self.client = client
        self.model_config = model_config
        self.log: list[dict[str, Any]] = []
        self.memoization: dict[tuple, bool] = {}

    def is_equal(self, problem: str, answer_a: str, answer_b: str) -> bool:
        cached = self.memoization.get((problem, answer_a, answer_b),
                                      self.memoization.get((problem, answer_b, answer_a), None))
        if cached is not None:
            return cached

        equivalence = self.judge_answer_equality(problem, answer_a, answer_b)
        self.log.append({**equivalence, "problem": problem, "answer_a": answer_a, "answer_b": answer_b})
        is_match = equivalence.get("match", False)
        self.memoization[(problem, answer_a, answer_b)] = is_match
        return is_match

    def judge_answer_equality(self, problem: str, answer_a: Optional[str], answer_b: Optional[str]) -> Dict[str, Any]:
        """
        0. if one answer is empty and the other isn't => no match
        1. exact (case-insensitive) matches => equal
        2. otherwise ask the LLM judge
        """
        if bool(len((answer_a or "").strip())) ^ bool(len((answer_b or "").strip())):
            return {"match": False, "reason": "empty"}

        if answer_a.lower().strip() == answer_b.lower().strip():
            return {"match": True, "reason": "exact_match"}

        # Any API failure propagates: we never fabricate (and then memoize) a False on error,
        # which would silently corrupt the cache. A broken run must fail loudly, not poison.
        if _judge_answer_equality_llm(self.client, self.model_config, self.SYSTEM_PROMPT, problem, answer_a or "", answer_b or ""):
            return {"match": True, "reason": "llm_match"}

        return {"match": False, "reason": "no_match"}

    def save(self, path: str):
        """Save model_config, log, and memoization to a file."""
        with open(path, "wb") as f:
            pickle.dump({
                "model_config": self.model_config,
                "log": self.log,
                "memoization": self.memoization
            }, f)

    @classmethod
    def load(cls, client: OpenAI, path: str):
        """Load model_config, log, and memoization from a file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(client, data["model_config"])
        obj.log = data["log"]
        obj.memoization = data["memoization"]
        return obj

class MathSemanticEqualityChecker(_SemanticEqualityCheckerBase):
    SYSTEM_PROMPT = MATH_EQUALITY_SYSTEM_PROMPT

class ScienceSemanticEqualityChecker(_SemanticEqualityCheckerBase):
    SYSTEM_PROMPT = SCIENCE_EQUALITY_SYSTEM_PROMPT

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
            equivalence = {"match": False, "reason": "no_int_found"}
        else:
            match = (a_int == b_int)
            equivalence = {"match": match, "reason": "int_match" if match else "int_mismatch", "a_int": a_int, "b_int": b_int}

        is_match = equivalence["match"]
        return is_match
