import os
import argparse
import torch
import torch.nn as nn
import transformers
from tqdm import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader
from sklearn import metrics
import utils
import pickle as pkl
from loguru import logger
from utils import set_logger, ClassificationModel, train_loop
import wandb
from transformers.modeling_outputs import SequenceClassifierOutput
from accelerate.utils import find_executable_batch_size

def tokenize_and_record(examples, tokenizer):
    # Tokenize without padding/truncation, record per-example lengths
    toks = tokenizer(examples["text"], truncation=True)
    return {
        "input_ids": toks["input_ids"],
        "attention_mask": toks["attention_mask"],
        "length": [len(ids) for ids in toks["input_ids"]],
        "labels": examples["labels"],
    }

def tokenize_and_pad(examples, tokenizer, max_length):
    toks = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    toks["labels"] = torch.tensor(examples["labels"], dtype=torch.long)
    return toks

def main(args):
    accelerator = Accelerator(mixed_precision="bf16")

    identifier = (f"{args.dataset_name.replace('/', '_')}_{args.model_name.replace('/', '_')}_"
                  f"{args.token_selection}_token_linear_training")
    output_dir = os.path.join(args.result_path, identifier)

    if os.path.exists(os.path.join(output_dir, "classifier_head.pt")):
        if accelerator.is_main_process:
            print("Output directory already exists. Exiting.")
        accelerator.end_training()
        return

    if accelerator.is_main_process:
        logfile_path = os.path.join(args.log_dir, f"{identifier}.log")
        set_logger(logfile_path=logfile_path)
        logger.info(f"Logging to {logfile_path}")

    # Load the dataset.
    raw = utils.dataset_load(args.dataset_name)

    # Initialize the tokenizer and model.
    tokenizer, base_model = utils.tokenizer_model_load(args.model_name, torch_dtype=torch.bfloat16)
    if 'Qwen' in args.model_name or 'Llama' in args.model_name:
        base_model.lm_head = nn.Identity()
    base_model.config.use_cache = False  # Avoid extra memory usage since we are not doing generation
    for p in base_model.parameters():
        p.requires_grad = False
    if accelerator.is_main_process:
        logger.info(f"Loaded tokenizer and model: {args.model_name}")

    train_dataset = raw['train'].map(
        lambda ex: tokenize_and_record(ex, tokenizer),
        batched=True, remove_columns=raw['train'].column_names, keep_in_memory=True)
    test_dataset = raw['test'].map(
        lambda ex: tokenize_and_record(ex, tokenizer),
        batched=True, remove_columns=raw['test'].column_names, keep_in_memory=True)
    all_lens = train_dataset['length'] + test_dataset['length']
    max_length = max(all_lens)
    train_dataset = raw["train"].map(
        lambda ex: tokenize_and_pad(ex, tokenizer, max_length),
        batched=True,
        remove_columns=raw["train"].column_names
    )
    test_dataset = raw["test"].map(
        lambda ex: tokenize_and_pad(ex, tokenizer, max_length),
        batched=True,
        remove_columns=raw["test"].column_names
    )
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    num_labels = len(set(sample["labels"] for sample in train_dataset))
    if accelerator.is_main_process:
        logger.info(f"Prepared dataset: {args.dataset_name}, train size: {len(train_dataset)}, test size: {len(test_dataset)}")
        logger.info(f"Dataset max length: {max_length}")

    @find_executable_batch_size(starting_batch_size=args.batch_size)
    def inner_training_loop(batch_size):
        nonlocal accelerator
        accelerator.free_memory()

        model = ClassificationModel(base_model, num_labels=num_labels, token_selection=args.token_selection, linear_probe=True)
        if accelerator.is_main_process:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    logger.info(f"Trainable parameter: {name}")

        # Optimizer and Scheduler.
        optimizer = torch.optim.AdamW(params=model.classifier.parameters(), lr=args.learning_rate)
        scheduler = None

        # DataLoaders
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        eval_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        model_p, opt_p, sched_p, train_p, eval_p = accelerator.prepare(
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
        train_curve, eval_curve = train_loop(model_p, train_p, eval_p, opt_p, sched_p, accelerator, args)
        if accelerator.is_main_process:
            wandb.finish()
        return train_curve, eval_curve, model_p

    train_curve, eval_curve, model_p = inner_training_loop()

    # Save the final model checkpoint only on the main process.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving the model checkpoint and train/val curves to {output_dir}")
        with open(os.path.join(output_dir, "train_curve.pkl"), "wb") as f:
            pkl.dump(train_curve, f)
        with open(os.path.join(output_dir, "eval_curve.pkl"), "wb") as f:
            pkl.dump(eval_curve, f)
        unwrapped_model = accelerator.unwrap_model(model_p)
        unwrapped_model.save_pretrained(output_dir, safe_serialization=True)
    wandb.finish()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear training")

    # Command-line arguments.
    parser.add_argument("--dataset_name", type=str, default="rotten_tomatoes",
                        help="Name of the dataset to be used")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B",
                        help="Name of the pre-trained model")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training and evaluation")
    parser.add_argument("--n_epochs", type=int, default=50,
                        help="Total number of training steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for the optimizer")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of worker threads for data loading")
    parser.add_argument("--token_selection", type=str, default="mean",
                        help="Token selection method: 'last' or 'mean'")
    parser.add_argument("--result_path", type=str, default="./linear_training",
                        help="Path to save the best model checkpoint and train/val curves")
    parser.add_argument("--log_dir", type=str, default="./linear_training/linear_training_logs",
                        help="Directory to save log files")
    parser.add_argument("--logging_steps", type=int, default=1,
                        help="Interval (in steps) for logging training loss")
    parser.add_argument("--wandb_entity", type=str, required=True,
                        help="Wandb entity name")
    parser.add_argument("--wandb_project", type=str, default="NTPS",
                        help="Wandb project name")

    args = parser.parse_args()
    main(args)
