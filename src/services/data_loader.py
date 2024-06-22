"""
"""

from datasets import load_dataset
from transformers import (PreTrainedTokenizerFast,
                          DefaultDataCollator)

from copy import deepcopy


class DataLoader:
    """
    """

    def __init__(
        self,
        file_path: str = None,
        max_length: int = 1024,
        stride: int = 128,
        tokenizer: PreTrainedTokenizerFast = None
    ):
        self.file_path = file_path
        self.max_length = max_length
        self.stride = stride
        self.tokenizer = tokenizer
        self.dataset = None

    def load_json(self):
        """
        """

        self.dataset = load_dataset(
            path=self.file_path
        )

    def remove_plausible_answers(self):
        """
        """

        data = deepcopy(self.dataset)
        data["train"] = data["train"].filter(
            lambda x: len(x["answers"]["text"]) == 1
        )
        data["validation"] = data["validation"].filter(
            lambda x: len(x["answers"]["text"]) == 1
        )
        return data

    def get_dataset(self):
        """
        """

        return self.dataset

    def preprocess_data(self, data):
        """
        """

        if self.tokenizer == None:
            raise ValueError("Tokenizer is not found!!!")

        questions = [q.strip() for q in data["question"]]
        inputs = self.tokenizer(
            questions,
            data["context"],
            max_length=self.max_length,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = data["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)
                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    def apply_processing(self, data, data_group):
        """
        """

        group_data = data[data_group].map(
            self.preprocess_data,
            batched=True,
            remove_columns=data[data_group].column_names
        )
        return group_data

    def data_collactor_config(self):
        """
        """

        data_collator = DefaultDataCollator()
        return data_collator
