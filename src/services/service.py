"""
This model is to ultize all the services
"""

import argparse
from typing import Any
from datasets import (Dataset,
                      DatasetDict)

from transformers import (AutoModelForQuestionAnswering,
                          PreTrainedTokenizerFast,
                          TrainingArguments,
                          DefaultDataCollator,
                          Trainer)

from src.services.model import ModelConfig
from src.services.inference import InferenceEngine
from src.services.data_loader import DataLoader


class Service:
    """
    A class to manage the configuration and execution of model training and inference services.
    """

    def __init__(
        self,
        args: argparse = None
    ):
        """
        Initializes the Service object.
        """
        self.model_loader = ModelConfig(
            base_model_path=args.path,
            use_local=args.local,
            output_dir=args.dir,
            eval_strategy=args.strategy,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.grad_accum_steps,
            eval_accumulation_steps=args.eval_accum_steps,
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            save_total_limit=args.save_total_limit,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            load_best_model_at_end=args.load_best_model,
            push_to_hub=args.push_to_hub,
            metric_for_best_model=args.metric_for_best_model,
            greater_is_better=args.greater_is_better,
            fp16=args.fp16,
            logging_dir=args.logger,
            logging_steps=args.log_step
        )
        self.inference_engine = InferenceEngine(
            context=None,
            question=None
        )
        self.data_loader = DataLoader(
            file_path=args.database,
            max_length=args.max_length,
            stride=args.stride
        )

    def get_tokenizer(self) -> PreTrainedTokenizerFast:
        """
        Retrieves and configures the tokenizer.

        Returns:
            PreTrainedTokenizerFast:
                The configured tokenizer with padding token added if not already present.
        """
        tokenizer = self.model_loader.tokenizer_config()
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        return tokenizer

    def get_model(
        self,
        tokenizer: PreTrainedTokenizerFast = None
    ) -> AutoModelForQuestionAnswering:
        """
        Retrieves and configures the model.

        Args:
            tokenizer (PreTrainedTokenizerFast): The tokenizer to be used with the model.

        Returns:
            AutoModelForQuestionAnswering: The configured model with resized token embeddings.
        """
        model = self.model_loader.model_config()
        model.resize_token_embeddings(len(tokenizer))
        return model

    def get_training_args(self) -> TrainingArguments:
        """
        Retrieves the training arguments.

        Returns:
            TrainingArguments: The training arguments configured for the model.
        """
        training_args = self.model_loader.hyper_parameters_config()
        return training_args

    def get_dataset(self) -> DatasetDict:
        """
        Modifies the dataset to remove plausible answers.

        Returns:
            Dataset: The modified dataset.
        """
        self.data_loader.load_json()
        dataset_modified = self.data_loader.remove_plausible_answers()
        return dataset_modified

    def get_trainer(
        self,
        model: AutoModelForQuestionAnswering = None,
        tokenizer: PreTrainedTokenizerFast = None,
        training_args: TrainingArguments = None,
        train_dataset: Dataset = None,
        validation_dataset: Dataset = None,
        data_collator: DefaultDataCollator = None
    ) -> Trainer:
        """
        Configures and retrieves the trainer.

        Args:
            model (AutoModelForQuestionAnswering): The model to be trained.
            tokenizer (PreTrainedTokenizerFast): The tokenizer to be used.
            training_args (TrainingArguments): The training arguments.
            train_dataset (Dataset): The training dataset.
            validation_dataset (Dataset): The validation dataset.
            data_collator (DefaultDataCollator): The data collator.

        Returns:
            Trainer: The configured trainer.
        """
        trainer = self.model_loader.trainer_config(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            data_collator=data_collator
        )
        return trainer

    def set_dataset_to_train(
        self,
        dataset_modified: Dataset = None
    ) -> Any:
        """
        Preprocesses and sets the datasets for training and validation.

        Args:
            dataset_modified (Dataset): The modified dataset.

        Returns:
            Tuple[Dataset, Dataset]: The preprocessed training and validation datasets.
        """
        train_dataset = self.data_loader.apply_processing(
            data=dataset_modified,
            data_group="train"
        )
        val_dataset = self.data_loader.apply_processing(
            data=dataset_modified,
            data_group="validation"
        )
        return train_dataset, val_dataset

    def set_trainer_to_train(
        self,
        train_dataset: Any,
        validation_dataset: Any
    ) -> Trainer:
        """
        Configures and retrieves the trainer for training.

        Args:
            train_dataset (Dataset): The training dataset.
            validation_dataset (Dataset): The validation dataset.

        Returns:
            Trainer: The configured trainer ready for training.
        """
        data_collator = self.data_loader.data_collator_config()
        tokenizer = self.get_tokenizer()
        model = self.get_model(
            tokenizer=tokenizer
        )
        training_args = self.get_training_args()
        trainer = self.get_trainer(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            data_collator=data_collator
        )
        return trainer
