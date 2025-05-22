import os
import argparse
import torch
import torch.nn as nn
import transformers
import wandb
from accelerate import Accelerator
from torch.utils.data import DataLoader
import utils
import pickle as pkl
from loguru import logger
from utils import set_logger, train_loop, ClassificationModel, TokenizerCollator


def main(args):
    accelerator = Accelerator(mixed_precision="bf16")

    identifier = f"{args.dataset_name.replace('/', '_')}_{args.model_name.replace('/', '_')}_{args.token_selection}_token_scratch"
    output_dir = os.path.join(args.result_path, identifier)

    if os.path.exists(output_dir):
        if accelerator.is_main_process:
            print("Output directory already exists. Exiting.")
        accelerator.end_training()
        return

    if accelerator.is_main_process:
        logfile_path = os.path.join(args.log_dir, f"{identifier}.log")
        set_logger(logfile_path=logfile_path)
        logger.info(f"Logging to {logfile_path}")

    # Load the dataset.
    dataset = utils.dataset_load(args.dataset_name)
    train_dataset, test_dataset = dataset["train"], dataset["test"]
    if accelerator.is_main_process:
        logger.info(f"Prepared dataset: {args.dataset_name}, train size: {len(train_dataset)}, test size: {len(test_dataset)}")

    # Initialize the tokenizer and model.
    tokenizer, model = utils.tokenizer_model_load(args.model_name, torch_dtype=torch.bfloat16, pretrained=False)
    if accelerator.is_main_process:
        logger.info(f"Loaded tokenizer and model: {args.model_name}")

    if 'Qwen' in args.model_name or 'Llama' in args.model_name:
        model.lm_head = nn.Identity()

    collate_fn = TokenizerCollator(tokenizer)

    num_labels = len(set(sample["labels"] for sample in train_dataset))

    model = ClassificationModel(model, num_labels=num_labels, token_selection=args.token_selection)

    if accelerator.is_main_process:
        for name, param in model.named_parameters():
            if param.requires_grad:
                    logger.info(f"Trainable parameter: {name}")

    # Optimizer and Scheduler.
    optimizer = transformers.Adafactor(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=1e-5,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * args.n_steps),
        num_training_steps=args.n_steps,
    )

    # DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate_fn, num_workers=args.num_workers)
    eval_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=collate_fn, num_workers=args.num_workers)

    model, optimizer, scheduler, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader, eval_dataloader
    )

    if accelerator.is_main_process:
        wandb.init(
            name=identifier,
            config=vars(args),
        )

    # Run training.
    if accelerator.is_main_process:
        logger.info("Starting training...")
    train_curve, eval_curve = train_loop(model, train_dataloader, eval_dataloader, optimizer, scheduler, accelerator, args)

    # Save the final model checkpoint only on the main process.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving the model checkpoint and train/val curves to {output_dir}")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, safe_serialization=True)
        with open(os.path.join(output_dir, "train_curve.pkl"), "wb") as f:
            pkl.dump(train_curve, f)
        with open(os.path.join(output_dir, "eval_curve.pkl"), "wb") as f:
            pkl.dump(eval_curve, f)
        wandb.finish()

    accelerator.end_training()


if __name__ == "__main__":
    @logger.catch
    def run():
        parser = argparse.ArgumentParser(description="LoRA Fine-Tuning")

        # Command-line arguments.
        parser.add_argument("--dataset_name", type=str, default="rotten_tomatoes",
                            help="Name of the dataset to be used")
        parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B",
                            help="Name of the pre-trained model")
        parser.add_argument("--batch_size", type=int, default=1,
                            help="Batch size for training and evaluation")
        parser.add_argument("--n_steps", type=int, default=10000,
                            help="Total number of training steps")
        parser.add_argument("--eval_steps", type=int, default=500,
                            help="Interval (in steps) for evaluation")
        parser.add_argument("--logging_steps", type=int, default=100,
                            help="Interval (in steps) for logging training loss")
        parser.add_argument("--learning_rate", type=float, default=1e-4,
                            help="Learning rate for the optimizer")
        parser.add_argument("--num_workers", type=int, default=4,
                            help="Number of worker threads for data loading")
        parser.add_argument("--token_selection", type=str, default="mean",
                            help="Token selection method: 'last' or 'mean'")
        parser.add_argument("--result_path", type=str, default="./scratch",
                            help="Path to save the best model checkpoint and train/val curves")
        parser.add_argument("--log_dir", type=str, default="./scratch_logs",
                            help="Directory to save log files")
        parser.add_argument("--wandb_entity", type=str, required=True,
                            help="Wandb entity name")
        parser.add_argument("--wandb_project", type=str, default="NTPS",
                            help="Wandb project name")
        args = parser.parse_args()
        main(args)

    run()
