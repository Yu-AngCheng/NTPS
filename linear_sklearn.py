import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh, svd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from tqdm import tqdm
import pickle
import llm_research
import os
import utils
from accelerate import Accelerator
import wandb
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import log_loss, accuracy_score, mean_squared_error
from sklearn. preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from torch.utils.data import Dataset, DataLoader
import joblib


def linear_probe_logistic(dataset_name: str,
                          model_name: str,
                          cache_dir: str = ".cache/",
                          result_dir: str = "./linear_probe") -> None:
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = utils.dataset_load(dataset_name)["train"]

    # Load tokenizer and model
    tokenizer, model =utils.tokenizer_model_load(model_name)
    model.to(device)
    model.eval()

    # Get number of classes (C)
    n = len(dataset)
    C = len(set(sample["labels"] for sample in dataset))

    # Prepare the labels
    labels = torch.tensor(dataset['labels'], dtype=torch.long, device=device)
    one_hot_labels = F.one_hot(labels, num_classes=C).float()

    # Get the number of layers and hidden dimension from a dummy forward pass
    with torch.no_grad():
        sample_text = dataset[0]["text"]
        toks = tokenizer(sample_text, truncation=True, padding=True, return_tensors="pt").to(device)
        hidden_states_temp = model(**toks, output_hidden_states=True).hidden_states

    num_layers = len(hidden_states_temp)
    hidden_dim = hidden_states_temp[0].size(-1)

    hidden_states_cache_path = f"{cache_dir}/{dataset_name.replace('/', '_')}_train_{model_name.replace('/', '_')}_alignment_cache.pkl"
    with open(hidden_states_cache_path, "rb") as f:
        hidden_states_cache = pickle.load(f)
        hidden_states_cache = hidden_states_cache[-2]

    results = {'logistic': {}, 'linear': {}}
    for i in range(num_layers):

        X = hidden_states_cache[i].cpu().to(torch.float32).numpy()
        for clf_type  in ['linear', 'logistic']:
            # create a pipeline with standard scaler and logistic regression
            if clf_type  == 'linear':
                model = make_pipeline(StandardScaler(), LinearRegression())
                y = one_hot_labels.cpu().numpy()
            else:
                model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver='saga', verbose=1, tol=0.01))
                y = labels.cpu().numpy()

            model.fit(X, y)
            y_pred = model.predict(X)
            acc = accuracy_score(y, y_pred) if clf_type  == 'logistic' else np.nan
            loss = log_loss(y, model.predict_proba(X)) if clf_type  == 'logistic' else np.nan
            mse = mean_squared_error(y, y_pred) if clf_type  == 'linear' else np.nan
            print(f"Layer {i}, {clf_type} model: Accuracy: {acc:.2f}, Loss: {loss:.4f}, MSE: {mse:.4f}")
            results[clf_type][i] = {
                "accuracy": acc,
                "loss": loss,
                "mse": mse
            }

            if result_dir:
                os.makedirs(result_dir, exist_ok=True)
                output_path = os.path.join(result_dir, f"{dataset_name.replace('/', '_')}_train_{model_name.replace('/', '_')}/")
                save_name = f"layer{i}_{clf_type}.joblib"
                save_path = os.path.join(output_path, save_name) if output_path else save_name
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                joblib.dump(model, save_path)
                print(f"saved {clf_type} model to {save_path}")

    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
        output_path = os.path.join(result_dir, f"{dataset_name.replace('/', '_')}_train_{model_name.replace('/', '_')}/")
        summary_path = os.path.join(output_path, "linear_probe.pkl") if output_path else "linear_probe.pkl"
        with open(summary_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Saved summary metrics to {summary_path}")


if __name__ == "__main__":
    import argparse

    def parse_args():
        parser = argparse.ArgumentParser(description="Compute linear_probe for a given dataset and model.")
        parser.add_argument("--dataset_name", type=str, default="climatebert/climate_sentiment", help="Name of the dataset to use.")
        parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B", help="Name of the model to use.")
        return parser.parse_args()

    args = parse_args()

    linear_probe_logistic(args.dataset_name, args.model_name)
