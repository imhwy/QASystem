"""
This module model is used to tr
"""

from typing import Optional
from transformers import (
    PreTrainedTokenizerFast,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
)


class ModelConfig:
    """
    A class to handle configuration of a question-answering model, including its tokenizer,
    model, hyperparameters, and trainer setup.
    """

    def __init__(
        self,
        base_model_path: str = None,
        use_local: bool = None,
        tokenizer_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        eval_strategy: Optional[str] = None,
        per_device_train_batch_size: Optional[int] = None,
        per_device_eval_batch_size: Optional[int] = None,
        gradient_accumulation_steps: Optional[int] = None,
        eval_accumulation_steps: Optional[int] = None,
        learning_rate: Optional[int] = None,
        num_train_epochs: Optional[int] = None,
        save_total_limit: Optional[int] = None,
        save_steps: Optional[int] = None,
        eval_steps: Optional[int] = None,
        fp16: Optional[bool] = None,
        load_best_model_at_end: Optional[bool] = None,
        push_to_hub: Optional[bool] = None,
        metric_for_best_model: Optional[str] = None,
        greater_is_better: Optional[bool] = None,
        logging_dir: Optional[str] = None,
        logging_steps: Optional[int] = None,
    ):
        """
        Initializes the ModelConfig with the specified parameters.

        Args:
            base_model_path (str): The path to the base model.
            use_local (bool): Whether to use local files only.
            tokenizer_path (Optional[str]): The path to the tokenizer.
            output_dir (Optional[str]): The directory to save the output.
            eval_strategy (Optional[str]): The evaluation strategy to use.
            per_device_train_batch_size (Optional[int]): Batch size for training.
            per_device_eval_batch_size (Optional[int]): Batch size for evaluation.
            gradient_accumulation_steps (Optional[int]): Gradient accumulation steps.
            eval_accumulation_steps (Optional[int]): Evaluation accumulation steps.
            learning_rate (Optional[int]): The learning rate.
            num_train_epochs (Optional[int]): Number of training epochs.
            save_total_limit (Optional[int]): Limit on the total number of checkpoints to save.
            save_steps (Optional[int]): Save checkpoint every X steps.
            eval_steps (Optional[int]): Evaluate every X steps.
            load_best_model_at_end (Optional[bool]): Whether to load the best model at the end of training.
            push_to_hub (Optional[bool]): Whether to push the model to the Hugging Face Hub.
        """
        self.base_model_path = base_model_path
        self.use_local = use_local
        self.tokenizer_path = tokenizer_path
        self.output_dir = output_dir
        self.strategy = eval_strategy
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.eval_accumulation_steps = eval_accumulation_steps
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.save_total_limit = save_total_limit
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.fb16 = fp16
        self.load_best_model_at_end = load_best_model_at_end
        self.push_to_hub = push_to_hub
        self.fp16 = fp16
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        self.logging_dir = logging_dir
        self.logging_steps = logging_steps

    def tokenizer_config(self):
        """
        Configures and returns a tokenizer for the model.

        Returns:
            PreTrainedTokenizerFast: The configured tokenizer.
        """
        if self.tokenizer_path is not None:
            tokenizer = PreTrainedTokenizerFast.from_pretrained(
                self.tokenizer_path,
                local_files_only=self.use_local
            )
            print("loading tokenizer avalable!!!")
            return tokenizer
        print("No tokenizer found!!!")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            self.base_model_path,
            local_files_only=self.use_local
        )
        print("loading tokenizer from base model!!!")
        return tokenizer

    def model_config(self):
        """
        Configures and returns a question-answering model.

        Returns:
            AutoModelForQuestionAnswering: The configured model.
        """
        model = AutoModelForQuestionAnswering.from_pretrained(
            self.base_model_path,
            local_files_only=self.use_local
        )
        return model

    def hyper_parameters_config(self):
        """
        Configures and returns the hyperparameters for training the model.

        Returns:
            TrainingArguments: The configured training arguments.
        """
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy=self.strategy,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            eval_accumulation_steps=self.eval_accumulation_steps,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_train_epochs,
            save_total_limit=self.save_total_limit,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            load_best_model_at_end=self.load_best_model_at_end,
            push_to_hub=self.push_to_hub,
            fp16=self.fp16,
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=self.greater_is_better,
            logging_dir=self.logging_dir,
            logging_steps=self.logging_steps,
            report_to="none"
        )
        return training_args

    def trainer_config(self, model, tokenizer, training_args, train_dataset, validation_dataset, data_collator) -> Trainer:
        """
        Configures and returns a Trainer for the model.

        Args:
            model (PreTrainedModel): The model to be trained.
            tokenizer (PreTrainedTokenizerFast): The tokenizer to be used.
            training_args (TrainingArguments): The training arguments.
            train_dataset (Dataset): The training dataset.
            validation_dataset (Dataset): The validation dataset.
            data_collator (DataCollator): The data collator.

        Returns:
            Trainer: The configured trainer.
        """
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        return trainer
