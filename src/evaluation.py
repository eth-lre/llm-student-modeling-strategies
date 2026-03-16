import numpy as np
import scipy.stats as st
from src.equality import EqualityChecker

def proportional_match(equality_checker: EqualityChecker, problem: str, responses: list[str], groundtruths: list[str]) -> float:
    """ Proportional match metric: the proportion of groundtruths covered by the responses """
    overlap = sum(equality_checker.is_equal(problem, r, g) for r in responses for g in groundtruths)
    return overlap / len(groundtruths)

def exact_match(equality_checker: EqualityChecker, problem: str, responses: list[str], groundtruths: list[str]) -> bool:
    """ Exact match metric: true if all groundtruths are covered by the responses """
    return all(any(equality_checker.is_equal(problem, r, g) for r in responses) for g in groundtruths)

def partial_match(equality_checker: EqualityChecker, problem: str, responses: list[str], groundtruths: list[str]) -> bool:
    """ Partial match metric: true if at least one groundtruth is covered by the responses """
    return any(equality_checker.is_equal(problem, r, g) for r in responses for g in groundtruths)

def number_correct(equality_checker: EqualityChecker, problem: str, responses: list[str], correct: str) -> int:
    """ Number of distractors suggested instead of correct answers """
    return sum([equality_checker.is_equal(problem, r, correct) for r in responses])

def number_repetitions(equality_checker: EqualityChecker, problem: str, responses: list[str]) -> int:
    """ Number of distractors suggested that are equivalent to any other previously generated distractor """
    return sum([equality_checker.is_equal(problem, r, o) for i,r in enumerate(responses) for o in responses[i+1:]])

def get_mean_and_ci(scores: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """ Mean and confidence interval using t-distribution """
    mean = np.mean(scores)
    sem = st.sem(scores)
    n = len(scores)

    h = sem * st.t.ppf((1 + confidence) / 2, n - 1)
    return mean, h