from abc import ABC, abstractmethod

class MisconceptionModel(ABC):
    """
        Abstract base class for generating misconceptions.
        Implementations should define how to propose a misconception
        given the exam context, problem, and previously tried misconceptions.
    """

    @abstractmethod
    def propose(
        self, context: dict[str, str],
        tried_misconceptions: set[str], num_misconceptions: int
    ) -> tuple[list[str],dict[str,str]]:
        """
            Return a list of newly proposed misconceptions and metadata about them
        """
        pass