from collections import defaultdict
from typing import Any
import os
import re
import json
import random
from abc import ABC, abstractmethod

import pandas as pd

class ADataset(ABC):
    """Abstract base class for datasets."""

    @abstractmethod
    def __len__(self) -> int:
        """ Return the number of items in the dataset. """
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> dict[str, Any]:
        """ Return the item at the given index. """
        pass

    @abstractmethod
    def save(self, path: str):
        """ Saves this dataset to the specified location """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'ADataset':
        """ Loads a dataset from a specified location """
        pass


class EEDIDataset(ADataset):
    """ Dataset of eedi MCQs. Persists the order of questions and also the in-context examples per question """
    def __init__(self):
        super().__init__()

    def __init__(self, data_folder: str = None, n_limit: int = None, questions_df: pd.DataFrame = None, questionids: list[str] = None):
        super().__init__()

        self.image_regex = r"(.*?)!\[([\s\S]*?)\]\(([\s\S]*?)\)"
        self.answer_choices = ["A", "B", "C", "D"]

        self.data_folder = data_folder
        self.n_limit = n_limit

        solvable_metadata_path = os.path.join(data_folder, "problem_solvable_by_questionid_corrected.json")
        if os.path.exists(solvable_metadata_path):
            with open(solvable_metadata_path, "r") as f:
                self.problem_solvable_by_questionid = json.load(f)
        else:
            self.problem_solvable_by_questionid = defaultdict(lambda k: True)

        if questions_df is None:
            misconception_mapping_df = pd.read_csv(os.path.join(data_folder, "misconception_mapping.csv"))
            questions_df = pd.read_csv(os.path.join(data_folder, "train.csv"), dtype={"QuestionId": str})

            # denormalize the questions dataset to include all high level errors and step-by-step correct solutions
            answer_choices = ["A", "B", "C", "D"]
            for i in answer_choices:
                questions_df = questions_df.merge(misconception_mapping_df, left_on=f"Misconception{i}Id", right_on="MisconceptionId", how="left")
                questions_df = questions_df.drop(columns=["MisconceptionId"])
                questions_df = questions_df.rename(columns={"MisconceptionName": f"Misconception{i}"})
        

            # remove all questions with visuals or that do not have full annotations
            questions_df["AllDistrBasedOnMiscon"] = questions_df.apply(lambda mcq: (pd.isna(mcq["MisconceptionA"]) + pd.isna(mcq["MisconceptionB"]) + pd.isna(mcq["MisconceptionC"]) + pd.isna(mcq["MisconceptionD"]) == 1), axis=1)
            questions_df["ContainsImage"] = questions_df["QuestionText"].str.contains(self.image_regex) | questions_df["AnswerAText"].str.contains(self.image_regex) | questions_df["AnswerBText"].str.contains(self.image_regex) \
                | questions_df["AnswerCText"].str.contains(self.image_regex) | questions_df["AnswerDText"].str.contains(self.image_regex)
            questions_df = questions_df[~questions_df["ContainsImage"] & questions_df["AllDistrBasedOnMiscon"]]

            self.questions_df = questions_df
        else:
            self.questions_df = questions_df

        correct_rts_path = os.path.join(data_folder, "correct_rts.csv")
        if os.path.exists(correct_rts_path):
            correct_rts_df = pd.read_csv(correct_rts_path, dtype={"QuestionId": str})
            self.questions_df_extd = questions_df.merge(correct_rts_df, how="left", on="QuestionId")
        else:
            self.questions_df_extd = self.questions_df

        if questionids is None:
            self.questionids = list(self.questions_df_extd["QuestionId"])
            if n_limit is not None:
                self.questionids = list(random.sample(list(self.questions_df_extd["QuestionId"]), k=n_limit))
        else:
            self.questionids = questionids
    
    def _get_num_reasoning_steps(self, reasoning: str) -> int:
        return max(map(int, re.findall(r'(?m)^(\d+)\.', f"1. \n{reasoning}")))
    
    def _get_problem(self, questionid: str) -> dict[str, str]:
        """ Loads a problem by its questionid """
        row = self.questions_df_extd[self.questions_df_extd["QuestionId"] == questionid].iloc[0]
        correct_answer_id = row["CorrectAnswer"]
        
        return {
            "QuestionId": questionid,
            "Question": row["QuestionText"],
            "Topic": row["SubjectName"],
            "Concept": row["ConstructName"],
            "Answer": row[f"Answer{correct_answer_id}Text"],
            "NumReasoningSteps": self._get_num_reasoning_steps(row["Reasoning"]),
            "NumDistractors": 3,
            "Solvable": self.problem_solvable_by_questionid[questionid],
            **{
                f"Distractor{i+1}_Error": row[f"Misconception{answer_id}"]
                for i,answer_id in enumerate([choice for choice in self.answer_choices if choice != correct_answer_id])
            },
            **{
                f"Distractor{i+1}": row[f"Answer{answer_id}Text"]
                for i,answer_id in enumerate([choice for choice in self.answer_choices if choice != correct_answer_id])
            },
        }
    
    def _get_choices(self, questionid: str) -> dict[str,str]:
        """ Extract the true correct answer and distractor answers from a question specified by its id """
        row = self.questions_df_extd[self.questions_df_extd["QuestionId"] == questionid].iloc[0]
        correct_answer_id = row["CorrectAnswer"]

        correct_answer_text = ""
        distractors = []
        for answer_id in self.answer_choices:
            answer = row[f"Answer{answer_id}Text"]
            if answer_id == correct_answer_id:
                correct_answer_text = answer
            else:
                distractors.append(answer)
        return {
            "CorrectAnswer": correct_answer_text,
            "Distractors": distractors
        }
    
    def __len__(self) -> int:
        return len(self.questionids)
    
    def __getitem__(self, index: int) -> dict[str, Any]:
        questionid = self.questionids[index]
        return {
            "Problem": self._get_problem(questionid),
            "Choices": self._get_choices(questionid),
            "NumDistractors": 3,
        }
    
    def save(self, path: str):
        with open(path, "w+") as f:
            json.dump({
                "data_folder": self.data_folder,
                "n_limit": self.n_limit,
                "questions_df": self.questions_df.to_json(orient="records"),
                "questionids": self.questionids
            }, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "r+") as f:
            d = json.load(f)
            return cls(
                data_folder = d["data_folder"],
                n_limit = d["n_limit"],
                questions_df = pd.read_json(d["questions_df"], orient="records", dtype={"QuestionId": str}),
                questionids = d["questionids"],
            )
        

def get_or_create_dataset(data_folder: str, n_limit: int|None = 100) -> ADataset:
    dataset_path = os.path.join(data_folder, f"dataset-{n_limit}.json")
    dataset_ctor = {
        "eedi_data": EEDIDataset
    }.get(data_folder, None)

    if dataset_ctor is None:
        raise ValueError(f"Unknown constructor for data of folder: {data_folder}")
    
    if os.path.exists(dataset_path):
        print(f"Loading existing dataset: {data_folder}")
        dataset = dataset_ctor.load(dataset_path)
        assert dataset.n_limit == n_limit
    else:
        print(f"Creating new dataset: {data_folder}")
        dataset = dataset_ctor(data_folder, n_limit)
        dataset.save(dataset_path)
    return dataset