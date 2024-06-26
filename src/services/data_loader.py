"""
This module is used for loading data.
"""

from typing import Any
from datasets import (load_dataset,
                      Dataset)
from transformers import (DefaultDataCollator,
                          PreTrainedTokenizerFast,
                          BatchEncoding)


class DataLoader:
    """
    A class to handle loading, processing, and collating datasets for NLP tasks, particularly
    question-answering tasks.

    Attributes:
        file_path (str): The path to the dataset file.
        max_length (int): The maximum length of tokenized sequences.
        stride (int): The stride size for overlapping tokenized sequences.
        tokenizer (PreTrainedTokenizerFast): The tokenizer to be used for processing the data.
        dataset (Dataset): The dataset object loaded from the file.
    """

    def __init__(
        self,
        file_path: str = None,
        max_length: int = 1024,
        stride: int = 128
    ):
        """
        Initializes the DataLoader with the specified file path, tokenizer, max length, and stride.

        Args:
            file_path (str): The path to the dataset file.
            max_length (int): The maximum length of tokenized sequences.
            stride (int): The stride size for overlapping tokenized sequences.
            tokenizer (PreTrainedTokenizerFast): The tokenizer to be used for processing the data.
        """
        self.file_path = file_path
        self.max_length = max_length
        self.stride = stride

    def load_json(self) -> Dataset:
        """
        Loads a dataset from a JSON file specified by the file path.
        """
        dataset = load_dataset(path=self.file_path)
        print(dataset)
        dataset["train"] = dataset["train"].filter(
            lambda x: len(x["answers"]["text"]) == 1)
        dataset["validation"] = dataset["validation"].filter(
            lambda x: len(x["answers"]["text"]) == 1)
        return dataset

    def preprocess_data(
        self,
        data: Dataset,
        tokenizer: PreTrainedTokenizerFast
    ) -> BatchEncoding:
        """
        Preprocesses the dataset for training a question-answering model 
        by tokenizing the questions and contexts,
        and creating start and end positions for the answers.

        Args:
            data (Dataset): The dataset to be preprocessed.

        Returns:
            Dict: A dictionary containing tokenized inputs and 
            corresponding start and end positions for answers.
        """
        if tokenizer is None:
            raise ValueError("Tokenizer is not found!!!")

        questions = [q.strip() for q in data["question"]]
        inputs = tokenizer(
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

    def apply_processing(
        self,
        data: Dataset,
        data_group: str,
        tokenizer: PreTrainedTokenizerFast

    ) -> Any:
        """
        Applies preprocessing to a specified group (train/validation/test) within the dataset.

        Args:
            data (Dataset): The dataset to be processed.
            data_group (str): The group within the dataset to be processed 
            (e.g., 'train', 'validation').

        Returns:
            Dataset: The preprocessed dataset group.
        """
        group_data = data[data_group].map(
            self.preprocess_data,
            batched=True,
            fn_kwargs={'tokenizer': tokenizer},
            remove_columns=data[data_group].column_names
        )
        return group_data

    def data_collator_config(self) -> DefaultDataCollator:
        """
        Configures and returns a data collator for the dataset.

        Returns:
            DefaultDataCollator: The data collator object.
        """
        data_collator = DefaultDataCollator()
        return data_collator

    def preprocess_validated_dataset(
        self,
        examples: Any,
        tokenizer: PreTrainedTokenizerFast
    ) -> BatchEncoding:
        """
        Preprocesses the validation dataset for a question-answering task.

        Args:
            examples (Any): A batch of examples containing 'question', 'context', and 'id' fields.
            tokenizer (PreTrainedTokenizerFast): The tokenizer to be used for processing.

        Returns:
            BatchEncoding: The tokenized inputs ready for model inference, including overflow
                        token mappings, offset mappings, and example IDs.
        """
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=self.max_length,
            truncation="only_second",
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs
