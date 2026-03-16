from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class JointModel(ABC):
    """
        Abstract base class for generating distractors jointly/end-to-end.
        Implementations should define how to propose a fixed number of distractors
        given the context (problem etc)
    """

    @abstractmethod
    def generate_distractors(
        self, context: dict[str, str], num_distractors: int
    ) -> tuple[list[str],dict[str,str]]:
        """
            Return a newly proposed list of distractors of length num_distractors and all metadata about the generation
        """
        pass