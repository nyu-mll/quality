import os
from abc import abstractmethod
from dataclasses import dataclass, field

import numpy as np
import datasets
import transformers

from lrqa.utils.io_utils import read_json, read_jsonl


class Task:

    @property
    @abstractmethod
    def num_choices(self) -> int:
        raise NotImplementedError()

    @property
    def drop_columns(self) -> list:
        """Returns list of columns to drop when tokenizing
        (Not really necessary, just reduces clutter in the batch objects)

        Don't include any of:
            label
            context
            query
            option_*

        :return: list columns to drop
        """
        return []

    @abstractmethod
    def standardize_examples(self, examples) -> dict:
        """Called by (batched) dataset method to convert data to standard format
        Output is a dict of lists, with the following types
            - context: str
            - query: str
            - label: int
            - option_[0..NUM_CHOICES]: str

        Ultimately, examples will be formatted as:
            context + query + option
        or
            context + [sep] + query + option

        with NO SPACES, so adjust accordingly (e.g. prepending space to query/options)

        :return: dict of lists
        """
        raise NotImplementedError()

    @abstractmethod
    def get_datasets(self) -> dict:
        """Returns dict (or dict-like) of datasets, with keys:
            train
            validation
            test

        :return: dict[str, Dataset]
        """
        raise NotImplementedError()

    # noinspection PyMethodMayBeStatic
    def compute_metrics(self, p: transformers.EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=-1)
        
        if preds.ndim < 3:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}
        else:
            label_ids = p.label_ids
            total = 0
            num_correct = 0
            for idx, ex_labels in enumerate(label_ids):
                ex_labels[ex_labels == -100] = 1
                total += 1
                if (ex_labels == preds[idx]).all():
                    num_correct += 1
            return {'accuracy': num_correct / total}


class CosmosQATask(Task):
    @property
    def num_choices(self) -> int:
        return 4

    @property
    def drop_columns(self) -> list:
        return ["question", "answer0", "answer1", "answer2", "answer3"]

    @classmethod
    def standardize_examples(cls, examples):
        result = {
            "context": examples["context"],
            "query": prepend_space(examples["question"]),
        }
        for i in range(4):
            result[f"option_{i}"] = prepend_space(examples[f"answer{i}"])
        return result

    def get_datasets(self) -> dict:
        return datasets.load_dataset("cosmos_qa")
    
    
class RaceTask(Task):
    def get_datasets(self) -> dict:
        return datasets.load_dataset("race", "all")
    
    @classmethod
    def standardize_examples(cls, examples):
        result = {
            "context": examples["article"],
            "query": prepend_space(examples["question"]),
        }
        for i in range(4):
            result[f"option_{i}"] = prepend_space([ex_options[i] for ex_options in examples["options"]])
        label_mappings = {"A": 0, "B": 1, "C": 2, "D": 3}
        result["label"] = [label_mappings[ex_answer] for ex_answer in examples["answer"]]
        return result
    
    @property
    def drop_columns(self) -> list:
        return ["question", "article", "options", "answer"]
    
    @property
    def num_choices(self) -> int:
        return 4


class CustomJSONLTask(Task):
    def __init__(self, base_path, num_choices, drop_columns=None):
        self.base_path = base_path
        self._drop_columns = drop_columns if drop_columns else []
        self._num_choices = num_choices

    @property
    def drop_columns(self) -> list:
        return self._drop_columns

    @property
    def num_choices(self) -> int:
        return self._num_choices

    @classmethod
    def standardize_examples(cls, examples):
        # jsonl data should already be preformatted to have keys
        #    context
        #    query
        #    label
        #    option_*
        return examples

    def get_datasets(self) -> dict:
        phases = ["train", "validation", "test"]
        dataset_dict = {}
        for phase in phases:
            phase_path = os.path.join(self.base_path, f"{phase}.jsonl")
            if not os.path.exists(phase_path):
                continue
            dataset_dict[phase] = datasets.load_dataset(
                "json",
                data_files=phase_path,
            )["train"]  # <- yes this is weird
        return dataset_dict

    @classmethod
    def create_from_path(cls, base_path):
        config = read_json(os.path.join(base_path, "config.json"))
        return cls(
            base_path=base_path,
            num_choices=config["num_choices"],
            drop_columns=config.get("drop_columns", []),
        )

def prepend_space(list_of_strings: list) -> list:
    return [" " + x for x in list_of_strings]


@dataclass
class TaskArguments:
    task_name: str = field(
        metadata={"help": "Task name (e.g. CosmosQA, CustomJSONLTask)"}
    )
    task_base_path: str = field(
        metadata={"help": "Path to data from CustomJSONLTask"},
        default=None,
    )


def get_task(task_args: TaskArguments):
    if task_args.task_name == "custom":
        return CustomJSONLTask.create_from_path(base_path=task_args.task_base_path)
    task_dict = {
        "cosmosqa": CosmosQATask,
        "race": RaceTask,
    }
    return task_dict[task_args.task_name]()
