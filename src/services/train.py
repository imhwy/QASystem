"""
"""

import argparse

from data_loader import DataLoader
from model import ModelConfig


def main():
    parser = argparse.ArgumentParser(description="Setting hyper-parameter!!!")
    parser.add_argument('-m', '--name', type=str, default='',
                        help='Name of fine-tuned model')
    parser.add_argument('-v', '--database', type=str,
                        default='', help='The path to database')
    parser.add_argument('-p', '--path', type=str,
                        default='bartpho-syllable', help='The path to fine-tuned model')
    parser.add_argument('-l', '--local', type=bool,
                        default=False, help='Using local model')
    parser.add_argument('-d', '--dir', type=str, default='fine-tuned',
                        help='The path to save fine-tuned model')
    parser.add_argument('-s', '--strategy', type=str,
                        default='steps', help='Evaluation strategy')
    parser.add_argument('-b', '--train_batch_size', type=int,
                        default=4, help='per device training batch size')
    parser.add_argument('-e', '--eval_batch_size', type=int,
                        default=4, help='Per device evaluation batch size')
    parser.add_argument('-g', '--grad_accum_steps', type=int,
                        default=2, help='Gradient accumulation steps')
    parser.add_argument('-a', '--eval_accum_steps', type=int,
                        default=2, help='Evaluation accumulation steps')
    parser.add_argument('-r', '--learning_rate', type=float,
                        default=1e-5, help='Learning rate')
    parser.add_argument('-n', '--epochs', type=int,
                        default=4, help='Number of training epochs')
    parser.add_argument('-st', '--save_total_limit', type=int,
                        default=1, help='Number of total limit')
    parser.add_argument('-t', '--save_steps', type=int, default=2000,
                        help='Save checkpoint every X updates steps')
    parser.add_argument('-es', '--eval_steps', type=int,
                        default=2000, help='Evaluate every X updates steps')
    parser.add_argument('-k', '--load_best_model', type=bool,
                        default=True, help='Load the best model at the end')
    parser.add_argument('-u', '--push_to_hub', type=bool,
                        default=False, help='Push model checkpoints to hub')

    args = parser.parse_args()

    print(args.path)

    model_loader = ModelConfig(
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
        push_to_hub=args.push_to_hub
    )

    tokenizer = model_loader.tokenizer_config()
    model = model_loader.model_config()
    training_args = model_loader.hyper_parameters_config()
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

    data_loader = DataLoader(
        file_path=args.database,
        tokenizer=tokenizer
    )
    dataset = data_loader.load_json()
    dataset_modified = data_loader.remove_plausible_answers()
    train_dataset = data_loader.apply_processing(
        data=dataset_modified,
        data_group="train"
    )
    val_dataset = data_loader.apply_processing(
        data=dataset_modified,
        data_group="validation"
    )
    data_collator = data_loader.data_collactor_config()

    trainer = model_loader.trainer_config(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=train_dataset,
        validation_dataset=val_dataset,
        data_collator=data_collator
    )
    trainer.train()
    trainer.save_model(args.dỉr)


if __name__ == "__main__":
    main()
