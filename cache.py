import numpy as np
import os
import pickle
from typing import Dict, Tuple
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
from utils import dataset_load, tokenizer_model_load, set_logger
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta
from loguru import logger
from peft import PeftModel

os.environ["TOKENIZERS_PARALLELISM"] = "false" # Disable parallelism in tokenizer as we tokenize on the fly

def get_cache_filename(dataset_name, model_name, split, save_dir):

    return (
        f"{save_dir}/{dataset_name.replace('/', '_')}_"
        f"{split}_{model_name.replace('/', '_')}_alignment_cache.pkl"
    )


def initialize_dimensions(dataset, tokenizer, model):

    # Dataset dimensions
    n_samples = len(dataset)
    n_classes = len(set(sample["labels"] for sample in dataset))

    # Model dimensions
    with torch.no_grad():
        sample_input = tokenizer(
            dataset[0]["text"],
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(model.device)
        hidden_states = model(**sample_input, output_hidden_states=True).hidden_states

    n_layers = len(hidden_states)
    hidden_dim = hidden_states[0].size(-1)

    return n_samples, n_classes, n_layers, hidden_dim


def initialize_caches(n_samples, n_layers, hidden_dim, n_classes, device):
    # Helper for creating cache dictionaries
    def init_cache_dict(shape: Tuple) -> Dict[int, torch.Tensor]:
        return {layer: torch.zeros(shape, dtype=torch.float32, device=device)
                for layer in range(n_layers)}

    # Initialize all caches
    caches = {
        'meanX_meanXT_dict': init_cache_dict((hidden_dim, hidden_dim)),
        'meanX_YT_dict': init_cache_dict((hidden_dim, n_classes)),
        'lag0_cum_dict': init_cache_dict((hidden_dim, hidden_dim)),
        'lag1_cum_dict': init_cache_dict((hidden_dim, hidden_dim)),
    }

    return caches


def create_lag_matrices(seq_length, device):
    L1_cum = torch.tensor(
        np.triu(np.ones((seq_length, seq_length - 1))),
        dtype=torch.float32, device=device
    )
    L2 = torch.tensor(
        np.vstack([np.zeros((1, seq_length - 1)), np.eye(seq_length - 1)]),
        dtype=torch.float32, device=device
    )
    return L1_cum, L2


def update_sentence_representations(
        caches: Dict[str, Dict[int, torch.Tensor]],
        hidden_states,
        batch_size,
        attention_mask,
        token_counts,
        batch_indices,
        one_hot_labels,
        n_samples,
        n_layers):

    # Calculate mean and last token representations
    mean_tokens = (hidden_states * attention_mask.unsqueeze(0).unsqueeze(-1)).sum(dim=2) / token_counts.unsqueeze(
        0).unsqueeze(-1).to(torch.float32)

    # Calculate batch statistics
    meanX_meanXT_batch = mean_tokens.transpose(1, 2) @ mean_tokens / n_samples # torch.bmm
    meanX_YT_batch = mean_tokens.transpose(1, 2) @ one_hot_labels.unsqueeze(0).expand(n_layers, -1, -1) / n_samples

    # Update caches for each layer
    for layer in range(n_layers):
        caches['meanX_meanXT_dict'][layer] += meanX_meanXT_batch[layer]
        caches['meanX_YT_dict'][layer] += meanX_YT_batch[layer]


def update_lagging_matrices(
        caches: Dict[str, Dict[int, torch.Tensor]],
        hidden_states,
        word_embeddings,
        batch_size,
        attention_mask,
        n_samples,
        device):

    for i in range(batch_size):
        seq_length = attention_mask[i].sum().item()
        if seq_length <= 1:
            continue

        L1_cum, L2 = create_lag_matrices(seq_length, device)
        Hs = hidden_states[:, i, :seq_length, :].to(torch.float32) # [layer, l, d]

        # all-layer batched matmuls in one continuous chain
        cov0c = Hs.transpose(1, 2).matmul(L1_cum).matmul(L1_cum.T).matmul(Hs)
        cov1c = Hs.transpose(1, 2).matmul(L1_cum).matmul(L2.T).matmul(Hs)

        for l in range(hidden_states.shape[0]):
            caches['lag0_cum_dict'][l] += cov0c[l] * 1.0 / n_samples
            caches['lag1_cum_dict'][l] += cov1c[l] * 1.0 / n_samples


def process_batch(
        batch: Dict[str, torch.Tensor],
        tokenizer,
        model,
        caches: Dict[str, Dict[int, torch.Tensor]],
        n_samples,
        n_classes,
        n_layers,
        accelerator):

    # Prepare inputs
    tokens = tokenizer(batch["text"], truncation=True, padding=True, return_tensors="pt").to(accelerator.device)
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    batch_indices = batch['idx']

    with torch.no_grad():
        # Forward pass
        outputs = model(input_ids, output_hidden_states=True, attention_mask=attention_mask)
        if accelerator.is_main_process:
            logger.info("Forward pass completed")

        hidden_states = torch.stack(outputs.hidden_states, dim=0)
        word_embeddings = hidden_states[0]  # First layer is word embeddings

        # One-hot encode labels
        one_hot_labels = F.one_hot(batch['labels'].long(), num_classes=n_classes).float()

        # Get batch dimensions
        batch_size, seq_len = input_ids.shape
        token_counts = attention_mask.sum(dim=1).long()

        # Update representations
        update_sentence_representations(
            caches, hidden_states, batch_size, attention_mask, token_counts,
            batch_indices, one_hot_labels, n_samples, n_layers
        )
        if accelerator.is_main_process:
            logger.info("Updated sentence representations")

        # Process each sample for lagging matrices
        update_lagging_matrices(
            caches, hidden_states, word_embeddings, batch_size, attention_mask, n_samples, accelerator.device
        )
        if accelerator.is_main_process:
            logger.info("Updated lagging matrices")


def reduce_distributed_tensors(caches: Dict[str, Dict[int, torch.Tensor]], accelerator):
    # only in a multi‐GPU job does `.state.distributed_type` become non‐None
    if accelerator.state.distributed_type is None:
        return

    for cache_dict in caches.values():
        for key, tensor in cache_dict.items():
            cache_dict[key] = accelerator.reduce(tensor, reduction="sum").detach().cpu()


def save_cache(caches: Dict[str, Dict[int, torch.Tensor]], filename):

    with open(filename, 'wb') as f:
        pickle.dump((
            caches['meanX_meanXT_dict'], caches['meanX_YT_dict'],
            caches['lag0_cum_dict'], caches['lag1_cum_dict'],
        ), f)


def alignment_cache_accelerate(
        dataset_name: str,
        model_name: str,
        split: str,
        batch_size: int = 1,
        dtype: str = 'float32',
        num_workers: int = 1,
        lora_ckpt: str = None,
        save_dir: str = '.cache/',
        log_dir: str = './cache_logs'):
    # Initialize distributed processing
    handler = InitProcessGroupKwargs(timeout=timedelta(hours=2))
    accelerator = Accelerator(kwargs_handlers=[handler])

    # Set up logger
    if accelerator.is_main_process:
        logfile_path = os.path.join(log_dir,
            f"{dataset_name.replace('/', '_')}_{split}_{model_name.replace('/', '_')}_alignment_cache.log")
        set_logger(logfile_path=logfile_path)
        logger.info(f"Logging to {logfile_path}")

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Get cache filename
    cache_filename = get_cache_filename(dataset_name, model_name, split, save_dir)

    # Check if cache already exists
    if os.path.exists(cache_filename):
        if accelerator.is_main_process:
            logger.info("Alignment cache already exists, skipping computation.")
        return

    # Load dataset and prepare dataloader
    dataset = dataset_load(dataset_name)[split]
    dataset = dataset.map(lambda example, i: {'idx': i}, with_indices=True)
    if accelerator.is_main_process:
        logger.info(f"Finished loading dataset")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    if accelerator.is_main_process:
        logger.info(f"Prepared dataloader with batch size {batch_size} and {num_workers} workers")

    # Load model and tokenizer
    if dtype == 'float16':
        dtype = torch.float16
    elif dtype == 'bfloat16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    tokenizer, model = tokenizer_model_load(model_name, torch_dtype=dtype)
    if lora_ckpt:
        model = PeftModel.from_pretrained(model, os.path.join(lora_ckpt,
                                                              f"{dataset_name.replace('/', '_')}_{model_name.replace('/', '_')}_layer0_mean_token_LoRA"))
        model = model.merge_and_unload()
        if accelerator.is_main_process:
            logger.info(f"Loaded LoRA checkpoint from {lora_ckpt}")
    model.eval()

    # Prepare for distributed execution
    model, dataloader = accelerator.prepare(model, dataloader)
    if accelerator.is_main_process:
        logger.info(f"Prepared model {model_name} and tokenizer with dtype {dtype}")

    # Initialize dimensions and caches
    n_samples, n_classes, n_layers, hidden_dim = initialize_dimensions(dataset, tokenizer, model)
    if accelerator.is_main_process:
        logger.info(f"{dataset_name} has {n_samples} samples, {n_classes} classes")
        logger.info(f"{model_name} has {n_layers} layers and hidden dimension {hidden_dim}")
    caches = initialize_caches(n_samples, n_layers, hidden_dim, n_classes, accelerator.device)
    if accelerator.is_main_process:
        logger.info("Initialized cache dictionaries")

    # Process all batches
    if accelerator.is_main_process:
        logger.info("Starting to process batches")
    for batch in tqdm(dataloader, desc="Processing batches"):
        process_batch(batch, tokenizer, model, caches, n_samples, n_classes, n_layers, accelerator)

    # Reduce and save results
    if accelerator.is_main_process:
        logger.info("Reducing distributed tensors")
    reduce_distributed_tensors(caches, accelerator)

    if accelerator.is_main_process:
        logger.info("Saving cache to file")
        save_cache(caches, cache_filename)


if __name__ == "__main__":
    @logger.catch()
    def run():
        import argparse


        def parse_args():
            parser = argparse.ArgumentParser(description="Generate alignment cache for a dataset and model.")
            parser.add_argument("--dataset_name", type=str, default="emotion",
                                help="Name of the dataset to process")
            parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B",
                                help="Model identifier to use for processing")
            parser.add_argument("--batch_size", type=int, default=32,
                                help="Batch size for processing")
            parser.add_argument("--split", type=str, default='train',
                                help="Dataset split to process (e.g., train, test)")
            parser.add_argument("--num_workers", type=int, default=4,
                                help="Number of worker processes for data loading")
            parser.add_argument("--save_dir", type=str, default='.lora_cache/',
                                help="Directory to save cache files")
            parser.add_argument("--dtype", type=str, default='float32', help="Data type.")
            parser.add_argument("--log_dir", type=str, default='./lora_cache_logs',
                                help="Directory to save log files")
            parser.add_argument("--lora_ckpt", type=str, default="lora",
                                help="Path to LoRA checkpoint to load")

            return parser.parse_args()


        args = parse_args()

        alignment_cache_accelerate(
            args.dataset_name,
            args.model_name,
            split=args.split,
            batch_size=args.batch_size,
            dtype=args.dtype,
            lora_ckpt=args.lora_ckpt,
            num_workers=args.num_workers,
            save_dir=args.save_dir
        )

    run()
