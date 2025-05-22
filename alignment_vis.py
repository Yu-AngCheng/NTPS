import numpy as np
from scipy.sparse.linalg import eigsh
import pickle
from scipy.linalg import eigh
from tqdm import tqdm
import os
import torch
import torch.nn.functional as F
import llm_research
import utils


def alignment_vis(model_name, dataset_name='emotion', i_layer=0):
    # Load cached alignment data
    data_path = f".cache/{dataset_name.replace('/', '_')}_train_{model_name.replace('/', '_')}_alignment_cache.pkl"
    with open(data_path, 'rb') as f:
        (meanX_meanXT_dict, meanX_YT_dict,
         lag0_cum_dict, lag1_cum_dict,
         mean_cache, last_cache) = pickle.load(f)
    del mean_cache, last_cache
    num_layers = len(meanX_meanXT_dict)
    assert i_layer < num_layers, f"Layer {i_layer} exceeds the number of layers {num_layers} in the model."

    M_t = meanX_meanXT_dict[i_layer]
    X_t = meanX_YT_dict[i_layer]
    C0_t = lag0_cum_dict[i_layer]
    C1_t = lag1_cum_dict[i_layer]

    M = M_t.detach().cpu().to(torch.float32).numpy()
    X = X_t.detach().cpu().to(torch.float32).numpy()
    C0 = C0_t.detach().cpu().to(torch.float32).numpy()
    C1 = C1_t.detach().cpu().to(torch.float32).numpy()

    # Compute generalized eigenvectors
    eigvals_U, U = eigsh(X @ X.T, k=2, M=M, which='LA')
    eigvals_V, V = eigsh(C1 @ C1.T, k=2, M=C0, which='LA')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = utils.tokenizer_model_load(model_name)
    model.to(device)
    model.eval()

    categories = {
        'Joy': ['lovely', 'enjoyable', 'amazing', 'delighted'],
        'Sad': ['horrible', 'disappointing', 'unhappy', 'lonely'],
        'Be': ['am', 'is', 'are', 'was', 'were', ],
    }

    # Prepare projections
    labels = ['U', 'V']
    projected = {
        label: {cat: [] for cat in categories}
        for label in labels
    }
    
    # Project each word
    for cat, word_list in categories.items():
        for w in word_list:
            toks = tokenizer(w, return_tensors='pt', padding=True, truncation=True).to(device)
            out = model(toks['input_ids'], attention_mask=toks['attention_mask'], output_hidden_states=True)
            h = out.hidden_states[i_layer].detach().cpu().numpy()
            h_vec = h.mean(axis=1)
            projected['U'][cat].append(h_vec @ U)
            projected['V'][cat].append(h_vec @ V)

    # Save the projections
    save_path = f"alignment_vis_{model_name.replace('/', '_')}_{dataset_name}_{i_layer}.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(projected, f)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Alignment Visualization")
    parser.add_argument('--model_name', type=str, default="apple/OpenELM-450M", help='Name of the model')
    parser.add_argument('--dataset_name', type=str, default='emotion', help='Name of the dataset')
    parser.add_argument('--i_layer', type=int, default=0, help='Layer index to visualize')

    args = parser.parse_args()
    alignment_vis(args.model_name, args.dataset_name, args.i_layer)