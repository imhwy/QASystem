import torch
from typing import Optional

from transformers import (PreTrainedTokenizerFast,
                          AutoModelForQuestionAnswering,
                          TrainingArguments,
                          Trainer,
                          DefaultDataCollator)


class ModelConfig:
    """
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
        load_best_model_at_end: Optional[bool] = None,
        push_to_hub: Optional[bool] = None
    ):

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
        self.load_best_model_at_end = load_best_model_at_end
        self.push_to_hub = push_to_hub
        self.metric_for_best_model = 'eval_loss'  # use eval_loss to select best model
        self.greater_is_better = False         # the smaller the eval_loss the better
        self.logging_dir = './logs'          # directory for storing logs
        self.logging_steps = 10

    def tokenizer_config(self):
        """
        """
        print(self.base_model_path)
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            self.base_model_path,
            local_files_only=self.use_local
        )
        print("load")
        return tokenizer

    def model_config(self):
        """
        """
        model = AutoModelForQuestionAnswering.from_pretrained(
            self.base_model_path,
            local_files_only=self.use_local
        )
        return model

    def hyper_parameters_config(self):
        """
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
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=self.greater_is_better,
            logging_dir=self.logging_dir,
            logging_steps=self.logging_steps,
        )
        return training_args

    def trainer_config(self, model, tokenizer, training_args, train_dataset, validation_dataset, data_collator):
        """
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
