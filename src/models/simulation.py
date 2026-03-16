from abc import ABC, abstractmethod

class SimulateModel(ABC):
    """
        Abstract base class for simulating a student's incorrect answer
        given a specific misconception.
    """
    @abstractmethod
    def simulate(
        self,
        context: dict[str, str],
        misconception: str,
    ) -> tuple[str,dict[str,str]]:
        """
            Return the simulated final answer produced
            under the given misconception.
        """
        pass