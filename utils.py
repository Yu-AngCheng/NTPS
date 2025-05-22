import torch
import wandb
from sklearn import metrics
from torch import nn as nn
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer, AutoConfig)
import numpy as np
import random
from datasets import (
    get_dataset_split_names,
    load_dataset,
    DatasetDict
)
from loguru import logger
import os
import sys

from transformers.modeling_outputs import SequenceClassifierOutput


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def dataset_load(name: str):
    if name == "wiki_toxic":
        name = "OxAISH-AL-LLM/wiki_toxic"
    elif name == "toxigen":
        name = "toxigen/toxigen-data"
    elif name == "bias_in_bios":
        name = "LabHC/bias_in_bios"
    elif name == "emotion":
        name = "dair-ai/emotion"
    elif name == "polarity":
        name = "fancyzhx/amazon_polarity"
    elif name == "snli":
        name = "stanfordnlp/snli"
    elif name == "sst2":
        name = "stanfordnlp/sst2"
    elif name == "medical":
        name = "medical_questions_pairs"
    elif name == "common_sense":
        name = "tau/commonsense_qa"
    print(f"Loading {name}")
    splits = get_dataset_split_names(name)
    print("\t-splits:", splits)
    data = DatasetDict()
    for split in splits:
        data[split] = load_dataset(name, split=split)
    for split in splits:
        if "label" in data[split].column_names:
            data[split] = data[split].rename_column("label", "labels")
        if name == "OxAISH-AL-LLM/wiki_toxic":
            assert "comment_text" in data[split].column_names
            data[split] = data[split].rename_column("comment_text", "text")
        elif name == "toxigen/toxigen-data":
            assert "toxicity_human" in data[split].column_names
            data[split] = data[split].rename_column("toxicity_human", "labels")
        elif name == "LabHC/bias_in_bios":
            data[split] = data[split].rename_column("hard_text", "text")
            data[split] = data[split].rename_column("profession", "labels")
        elif name == "fancyzhx/amazon_polarity":
            data[split] = data[split].rename_column("content", "text")
        elif name == "stanfordnlp/snli":

            def preprocess(example):
                for i, v in enumerate(example["hypothesis"]):
                    example["premise"][i] += " " + v
                    return example

            data[split] = data[split].map(preprocess, batched=True)
            data[split] = data[split].rename_column("premise", "text")
        elif name == "stanfordnlp/sst2":
            data[split] = data[split].rename_column("sentence", "text")
        elif name == "medical_questions_pairs":

            def preprocess(example):
                for i, v in enumerate(example["question_2"]):
                    example["question_1"][i] += " " + v
                    return example

            data[split] = data[split].map(preprocess, batched=True)
            data[split] = data[split].rename_column("question_1", "text")
        elif name == "DeveloperOats/DBPedia_Classes":
            data[split] = data[split].rename_column("l1", "labels")
            data[split] = data[split].class_encode_column("labels")
        elif name == "Bhuvaneshwari/intent_classification":
            data[split] = data[split].rename_column("intent", "labels")
            data[split] = data[split].class_encode_column("labels")
        elif name == "valurank/Topic_Classification":
            data[split] = data[split].rename_column("article_text", "text")
            data[split] = data[split].rename_column("topic", "labels")
            data[split] = data[split].class_encode_column("labels")
        elif name == "valurank/Topic_Classification":
            data[split] = data[split].rename_column("article_text", "text")
            data[split] = data[split].rename_column("topic", "labels")
            data[split] = data[split].class_encode_column("labels")
        elif name == "marksverdhei/clickbait_title_classification":
            data[split] = data[split].rename_column("title", "text")
            data[split] = data[split].rename_column("clickbait", "labels")
        elif name == "PriyaPatel/Bias_identification":
            data[split] = data[split].rename_column("context", "text")
            data[split] = data[split].rename_column("bias_type", "labels")
        elif name == "ucirvine/sms_spam":
            data[split] = data[split].rename_column("sms", "text")

        elif name == "tau/commonsense_qa":
            # function to be used if commonsense dataset is chosen, combines question and answer choices
            # makes the answers and choices corresponds to numbers instead of letters
            def preprocess(example):
                # combine question and chouces
                question = example["question"]
                print(f'{example["choices"]}')
                choices = example["choices"]["text"]
                # Keep the answer choices zero-indexed (0-4) for better alignment
                text = question + " " + " ".join([f"({i}) {choice}" for i, choice in enumerate(choices)])
                # Convert answerKey ('A'-'E') into numeric label (0-4)
                print(f'answer key: {example["answerKey"]}')
                print(f'ID: {example["id"]}')
                if split != "test":
                    label = ord(example["answerKey"]) - ord("A")
                else:
                    # dummy lable so that the code can run, replacing "test" with "validation" as "test" does not include
                    # the answerKey
                    label = 0
                    # return them
                return {"text": text, "labels": label}

            data[split] = data[split].map(preprocess)

        data[split] = data[split].filter(lambda row: row["labels"] >= 0)
        assert "text" in data[split].column_names
        print(f"\t-{split}: {data[split].shape}")
    # cases where need to make the test set the validation set
    if name == "stanfordnlp/sst2":
        data["test"] = data["validation"]
        del data["validation"]
    elif name == "tau/commonsense_qa":
        data["test"] = data["validation"]
        del data["validation"]
    if "test" not in data:
        data = data["train"].train_test_split(test_size=0.3, shuffle=True, seed=42)
    print("\t-columns:", data[split].column_names)
    return data


def tokenizer_model_load(model_name, torch_dtype=torch.float32, pretrained=True):
    if 'apple' in model_name:
        # according to the paper: https://arxiv.org/pdf/2404.14619 and
        # the example https://huggingface.co/apple/OpenELM/blob/main/generate_openelm.py
        # the tokenizer is llama2 tokenizer and the max context length is 2048 across all openelm models
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True)
        assert tokenizer.eos_token is not None
        tokenizer.pad_token = tokenizer.eos_token
        if pretrained:
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch_dtype)
        else:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        tokenizer.model_max_length = model.config.max_context_length
        return tokenizer, model
    elif 'llama' in model_name:
        # according to the doc: https://huggingface.co/docs/transformers/en/model_doc/llama3
        # and https://huggingface.co/docs/transformers/en/model_doc/llama2
        # the dtype should be float16 for llama models
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        assert tokenizer.eos_token is not None
        tokenizer.pad_token = tokenizer.eos_token
        if pretrained:
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch_dtype)
        else:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        tokenizer.model_max_length = model.config.max_position_embeddings
        return tokenizer, model
    elif 'Qwen' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if pretrained:
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch_dtype)
        else:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        return tokenizer, model
    else:
        raise ValueError(f"Model {model_name} not supported. Please use a model from apple, llama or Qwen.")


def set_logger(console_level="INFO", logfile_level="DEBUG", logfile_path: str = None):
    """
    Setup the logger.
    """
    logger.remove()
    logger.add(sys.stdout, level=console_level)
    if logfile_path:
        os.makedirs(os.path.dirname(logfile_path), exist_ok=True)
        logger.add(logfile_path, level=logfile_level, mode="w")

    # Redirect unhandled exceptions to logger
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.opt(exception=(exc_type, exc_value, exc_traceback)).critical("Unhandled exception")

    sys.excepthook = handle_exception


def evaluate(model, dataloader, accelerator):
    model.eval()
    all_logits, all_labels = [], []
    for batch in tqdm(dataloader, desc="Evaluating"):
        with torch.no_grad():
            outputs = model(**batch)

        all_logits.append(accelerator.gather_for_metrics(outputs.logits).cpu())
        all_labels.append(accelerator.gather_for_metrics(batch["labels"]).cpu())

    if accelerator.is_main_process:
        logits  = torch.cat(all_logits).numpy()
        labels  = torch.cat(all_labels).numpy()
        preds   = logits.argmax(-1)
        return {
            "accuracy": (preds == labels).mean(),
            "balanced_accuracy": metrics.balanced_accuracy_score(labels, preds),
        }
    return {}


def train_loop(model, train_dataloader, eval_dataloader, optimizer, scheduler, accelerator, args):
    train_curve, eval_curve = [], {}

    model.train()
    global_step = 0
    epoch = 0

    max_steps     = getattr(args, "n_steps",   None)
    logging_steps = getattr(args, "logging_steps", None)
    eval_steps    = getattr(args, "eval_steps",    None)
    max_epochs    = getattr(args, "n_epochs", None)
    assert max_steps is not None or max_epochs is not None, "Either n_steps or n_epochs must be specified"

    while True:
        if max_steps is not None:
            if global_step >= max_steps:
                break
        if max_epochs is not None:
            if epoch >= max_epochs:
                break

        epoch += 1
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch}")
        for batch in tqdm(train_dataloader, desc="Training"):
            model.train()
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            # Make a detached copy for logging
            loss_detached = loss.detach()
            loss_global = accelerator.reduce(loss_detached, reduction="mean")

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

            global_step += 1

            if accelerator.is_main_process:
                train_curve.append(loss_global.item())
                if logging_steps is not None and (global_step % logging_steps == 0):
                    wandb.log({"train_loss": loss_global.item(), "step": global_step})
                    logger.info(f"Step {global_step}: Train loss: {loss_global.item()}")

            if eval_steps is not None and global_step % args.eval_steps == 0:
                eval_metrics = evaluate(model, eval_dataloader, accelerator)
                if accelerator.is_main_process:
                    wandb.log({**eval_metrics, "step": global_step})
                    if eval_curve == {}:
                        eval_curve = {k: [v] for k, v in eval_metrics.items()}
                    else:
                        for k, v in eval_metrics.items():
                            eval_curve[k].append(v)
                    logger.info(f"Step {global_step}: Eval metrics: {eval_metrics}")

            if max_steps is not None and global_step >= max_steps:
                break

    eval_metrics = evaluate(model, eval_dataloader, accelerator)
    if accelerator.is_main_process:
        if eval_curve == {}:
            eval_curve = {k: [v] for k, v in eval_metrics.items()}
        else:
            for k, v in eval_metrics.items():
                eval_curve[k].append(v)
        logger.info(f"Eval metrics: {eval_metrics}")
        wandb.log(eval_metrics)

    return train_curve, eval_curve


class ClassificationModel(nn.Module):
    def __init__(self,
                 encoder,
                 num_labels,
                 criterion = torch.nn.CrossEntropyLoss(),
                 dropout: float = 0.05,
                 linear_probe = False,
                 token_selection: str = "last"):
        super().__init__()

        self.encoder = encoder
        self.criterion = criterion
        hidden_size = self.encoder.config.hidden_size if hasattr(self.encoder.config, "hidden_size") else self.encoder.config.model_dim

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        self.token_selection = token_selection
        self.linear_probe = linear_probe

    def save_pretrained(self, save_directory: str, safe_serialization=True):
        os.makedirs(save_directory, exist_ok=True)
        if self.linear_probe is False:
            self.encoder.save_pretrained(save_directory, safe_serialization=safe_serialization)
        torch.save(self.classifier.state_dict(), os.path.join(save_directory, "classifier_head.pt"))

    def forward(self, input_ids, attention_mask, labels=None):
        if self.linear_probe is True:
            with torch.no_grad():
                seq_hid = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                ).hidden_states[-1]
        else:
            seq_hid = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            ).hidden_states[-1]

        if self.token_selection == "last":
            last_token_idx = attention_mask.sum(dim=1) - 1  # subtract 1 for zero-based indexing, shape: (B,)
            batch_indices = torch.arange(seq_hid.size(0), device=seq_hid.device)  # (B,)
            token = seq_hid[batch_indices, last_token_idx]  # (B, D)
        elif self.token_selection == "mean":
            # (B, L, D) * (B, L, 1)
            masked_seq_hid = seq_hid * attention_mask.unsqueeze(-1)  # mask invalid tokens
            val_token_num = attention_mask.sum(dim=1, keepdim=True)  # (B, 1)
            token = masked_seq_hid.sum(dim=1) / val_token_num.clamp(min=1)  # (B, D)

        token = self.dropout(token)
        logits = self.classifier(token)

        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


class TokenizerCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, examples):
        texts  = [ex["text"]   for ex in examples]
        labels = [ex["labels"] for ex in examples]
        tok = self.tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
        tok["labels"] = torch.tensor(labels)
        return tok
