import numpy as np
from scipy.sparse.linalg import eigsh
import pickle
from scipy.linalg import eigh
from tqdm import tqdm
import os
import torch

def nullspace_eig_generalized(A: np.ndarray, B: np.ndarray, tol: float = 1e-8):
    eigvals_B, eigvecs_B = eigh(B)
    idx = eigvals_B > tol
    U_r = eigvecs_B[:, idx]
    S_r = np.diag(eigvals_B[idx])
    W = U_r @ np.linalg.inv(np.sqrt(S_r))
    A_std = W.T @ A @ W
    eigs, y = eigh(A_std)
    X = W @ y

    return eigs, X

def alignment_score(E_sentXYT, E_sentX_sentXT, Cov_shift, Cov, sparse_k=100, fraction_k=np.arange(0.05, 1.0, 0.05)):
    """
    Compute alignment metrics using both sparse and dense eigensolvers.

    Parameters:
        E_sentXYT (np.ndarray): Cross-covariance matrix between the input and output embeddings.
        E_sentX_sentXT (np.ndarray): Auto-covariance matrix of the input sentence embeddings.
        Cov_shift (np.ndarray): Shifted Auto-covariance matrix of the input token embeddings.
        Cov (np.ndarray): Covariance matrix of the input token embeddings.
        sparse_k (int): Number of eigenvectors to compute with the sparse solver.
        fraction_k (np.ndarray): Fractions of eigenvectors (from 0 to 1) to compute with the dense solver.

    Returns:
        tuple: (alignment_list_sparse, alignment_list_dense)
            - alignment_list_sparse: List of alignment metrics computed with the sparse eigensolver.
            - alignment_list_dense: List of alignment metrics computed with the dense eigensolver.
    """
    alignment_list_sparse = []
    alignment_list_dense = []

    # convert everything
    E_sentXYT      = E_sentXYT.detach().cpu().to(torch.float32).numpy()
    E_sentX_sentXT = E_sentX_sentXT.detach().cpu().to(torch.float32).numpy()
    Cov_shift      = Cov_shift.detach().cpu().to(torch.float32).numpy()
    Cov            = Cov.detach().cpu().to(torch.float32).numpy()

    if sparse_k > 0:
        # --- Sparse eigensolver: compute top sparse_k eigenvectors ---
        eigvals_U, U = eigsh(E_sentXYT @ E_sentXYT.T, k=sparse_k, M=E_sentX_sentXT, which='LA')
        eigvals_V, V = eigsh(Cov_shift @ Cov_shift.T, k=sparse_k, M=Cov, which='LA')

        for k in np.arange(1, sparse_k):
            U_k = U[:, :k]
            V_k = V[:, :k]
            P_k = V_k @ np.linalg.pinv(V_k)
            alignment_k = np.linalg.norm(P_k @ U_k, 'fro') ** 2 / np.linalg.norm(U_k, 'fro') ** 2
            alignment_list_sparse.append(alignment_k)
    else:
        print("Sparse eigensolver not used. sparse_k is set to 0.")
        # --- Sparse eigensolver: compute top sparse_k eigenvectors ---
        # sparse_k = min(np.linalg.matrix_rank(E_sentX_sentXT), np.linalg.matrix_rank(Cov))
        # eigvals_U, U = eigsh(E_sentXYT @ E_sentXYT.T, k=sparse_k, M=E_sentX_sentXT, which='LA')
        # eigvals_V, V = eigsh(Cov_shift @ Cov_shift.T, k=sparse_k, M=Cov, which='LA')
        #
        # for k in np.arange(1, sparse_k):
        #     U_k = U[:, :k]
        #     V_k = V[:, :k]
        #     P_k = V_k @ np.linalg.pinv(V_k)
        #     alignment_k = np.linalg.norm(P_k @ U_k, 'fro') ** 2 / np.linalg.norm(U_k, 'fro') ** 2
        #     alignment_list_sparse.append(alignment_k)

    # --- Dense eigensolver: compute eigenvectors for the full spectrum ---
    eigvals_U, U = nullspace_eig_generalized(A=E_sentXYT @ E_sentXYT.T, B=E_sentX_sentXT)
    eigvals_V, V = nullspace_eig_generalized(A=Cov_shift @ Cov_shift.T, B=Cov)


    for k in fraction_k:
        U_k = U[:, :int(k * U.shape[1])] # shape (d, k1)
        V_k = V[:, :int(k * V.shape[1])] # shape (d, k2)
        P_k = V_k @ np.linalg.pinv(V_k)
        alignment_k = np.linalg.norm(P_k @ U_k, 'fro') ** 2 / np.linalg.norm(U_k, 'fro') ** 2
        alignment_list_dense.append(alignment_k)

    if sparse_k > 0:
        return alignment_list_sparse, alignment_list_dense
    else:
        return None, alignment_list_dense


def process_dataset_model(args):
    """
    Process a single dataset-model pair:
      - Load pre-calculated alignment data.
      - Skip processing if the alignment score file already exists.
      - Compute alignment scores for both 'meansent' and 'lastsent' data.
      - Save the alignment scores.
    """
    dataset_name = args.dataset_name
    model_name = args.model_name
    sparse_k = args.sparse_k
    fraction_k = args.fraction_k
    print(f"Processing {dataset_name} and {model_name}")
    print(f"Using sparse_k: {sparse_k}, fraction_k: {fraction_k}")

    data_path = f"{args.data_path}/{dataset_name.replace('/', '_')}_train_{model_name.replace('/', '_')}_alignment_cache_hack.pkl"
    os.makedirs(f"{args.save_path}", exist_ok=True)
    save_path = f"{args.save_path}/{dataset_name.replace('/', '_')}_{model_name.replace('/', '_')}_alignment_scores.pkl"

    try:
        # meanX_meanXT_dict, meanX_YT_dict, lag0_cum_dict, lag1_cum_dict = torch.load(data_path, map_location=torch.device('cpu'))
        with open(data_path, 'rb') as f:
            (meanX_meanXT_dict, meanX_YT_dict,
             lag0_cum_dict, lag1_cum_dict) = pickle.load(f)
    except FileNotFoundError:
        print(f"File not found for {dataset_name} and {model_name}")
        return

    if os.path.exists(save_path):
        print(f"Alignment scores already computed for {dataset_name} and {model_name}")
        return

    num_layers = len(meanX_meanXT_dict)
    alignment_mean_cum = [None] * num_layers

    for i in tqdm(range(num_layers)):
        alignment_mean_cum[i] = alignment_score(meanX_YT_dict[i], meanX_meanXT_dict[i], lag1_cum_dict[i], lag0_cum_dict[i],
                                          sparse_k=sparse_k, fraction_k=fraction_k)
    with open(save_path, "wb") as f:
        pickle.dump({
            "alignment_mean_cum": alignment_mean_cum,
        }, f)
    print(f"Processed {dataset_name} and {model_name}")


if __name__ == "__main__":
    import argparse


    def parse_args():
        parser = argparse.ArgumentParser(description="Calculate alignment scores for a dataset-model pair.")
        parser.add_argument("--dataset_name", type=str, default="legacy-datasets/banking77", help="Name of the dataset.")
        parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B", help="Model identifier.")
        parser.add_argument("--sparse_k", type=int, default=0, help="Number of eigenvectors for sparse solver.")
        parser.add_argument("--fraction_k", type=float, nargs='+', default=np.arange(0.05, 1.0, 0.05),
                            help="Fractions of eigenvectors for dense solver.")
        parser.add_argument("--data_path", type=str, default=".cache", help="Path to the data directory.")
        parser.add_argument("--save_path", type=str, default="NTPS", help="Path to save the results.")

        return parser.parse_args()


    args = parse_args()
    print(f"Dataset: {args.dataset_name}, Model: {args.model_name}, Sparse k: {args.sparse_k}, Fraction k: {args.fraction_k}")
    process_dataset_model(args)