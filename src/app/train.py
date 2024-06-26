"""
Main script to initialize and train a model using the provided service configuration.
"""

import argparse

from src.services.service import Service


def main():
    """
    Main function to parse arguments, initialize the service, and start training.

        The function performs the following steps:
        1. Parses command-line arguments for model and training configurations.
        2. Initializes the Service object with the parsed arguments.
        3. Preprocesses the dataset and prepares it for training and validation.
        4. Configures the Trainer and starts the training process.
    """
    parser = argparse.ArgumentParser(description="Setting hyper-parameter!!!")
    parser.add_argument('-m', '--max_length', type=int,
                        default=1024, help='Max length of tokenized sequence')
    parser.add_argument('-st', '--stride', type=int,
                        default=128, help='Stride size for overlapping tokenized sequence')
    parser.add_argument('-db', '--database', type=str,
                        default='data/UITVisqAD2.0', help='The path to database')
    parser.add_argument('-p', '--path', type=str,
                        default='bartpho-syllable', help='The path to fine-tuned model')
    parser.add_argument('-l', '--local', type=bool,
                        default=True, help='Using local model')
    parser.add_argument('-d', '--dir', type=str, default='fine-tuned',
                        help='The path to save fine-tuned model')
    parser.add_argument('-s', '--strategy', type=str,
                        default='steps', help='Evaluation strategy')
    parser.add_argument('-t', '--train_batch_size', type=int,
                        default=4, help='per device training batch size')
    parser.add_argument('-e', '--eval_batch_size', type=int,
                        default=4, help='Per device evaluation batch size')
    parser.add_argument('-g', '--grad_accum_steps', type=int,
                        default=2, help='Gradient accumulation steps')
    parser.add_argument('-a', '--eval_accum_steps', type=int,
                        default=2, help='Evaluation accumulation steps')
    parser.add_argument('-lr', '--learning_rate', type=float,
                        default=1e-5, help='Learning rate')
    parser.add_argument('-ep', '--epochs', type=int,
                        default=4, help='Number of training epochs')
    parser.add_argument('-stl', '--save_total_limit', type=int,
                        default=1, help='Number of total limit')
    parser.add_argument('-sts', '--save_steps', type=int, default=2000,
                        help='Save checkpoint every X updates steps')
    parser.add_argument('-es', '--eval_steps', type=int,
                        default=2000, help='Evaluate every X updates steps')
    parser.add_argument('-lbm', '--load_best_model', type=bool,
                        default=True, help='Load the best model at the end')
    parser.add_argument('-uth', '--push_to_hub', type=bool,
                        default=False, help='Push model checkpoints to hub')
    parser.add_argument('-mfbm', '--metric_for_best_model', type=str,
                        default='eval_loss', help='Metric for best model')
    parser.add_argument('-gb', '--greater_is_better', type=bool,
                        default=False, help='Greater is better')
    parser.add_argument('-f', '--fp16', type=bool, default=False,
                        help='Enable fp16')
    parser.add_argument('-lg', '--logger', type=str,
                        default='./logs', help='Logger path')
    parser.add_argument('-ls', '--log_step', type=int,
                        default=10, help='Logger level')

    args = parser.parse_args()

    service = Service(
        args=args
    )

    dataset = service.get_dataset()
    train_dataset, val_dataset = service.set_dataset_to_train(dataset)
    trainer = service.set_trainer_to_train(
        train_dataset=train_dataset,
        validation_dataset=val_dataset
    )
    trainer.train()


if __name__ == "__main__":
    main()
