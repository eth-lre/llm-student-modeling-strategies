from src.models.misconception import MisconceptionModel

class GroundTruthMisconceptionModel(MisconceptionModel):
    """ Pulls misconceptions from a groundtruth dataset """
    def __init__(self):
        super().__init__()

    def propose(
        self, context: dict[str, str],
        tried_misconceptions: set[str], num_misconceptions: int
    ) -> tuple[list[str],dict[str,str]]:
        problem_meta = context["Problem"]
        available_misconceptions = [(i,context["Problem"][f"Distractor{i+1}_Error"]) for i in range(problem_meta["NumDistractors"])]
        
        misconceptions = []
        metadata = {"ids": []}
        for i,misconception in available_misconceptions:
            if misconception not in tried_misconceptions:
                misconceptions.append(misconception)
                metadata["ids"].append(f"Distractor{i+1}_Error")
                if len(misconceptions) == num_misconceptions:
                    return misconceptions, metadata
        raise ValueError(f"Asked for more ground truth misconceptions than are provided (provided = {available_misconceptions}, tried = {tried_misconceptions})")