# eegconformer/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset


def euclidean_align(X: np.ndarray) -> np.ndarray:
    """
    X: (N, C, T) float32 — epochs for one subject
    Returns X_aligned: (N, C, T) float32, whitened by R_mean^(-1/2)
    """
    N, C, T = X.shape
    covs = np.einsum("nct,ndt->ncd", X, X) / T
    R_mean = covs.mean(axis=0)

    eigvals, eigvecs = np.linalg.eigh(R_mean)
    eigvals = np.maximum(eigvals, 1e-10)
    R_inv_sqrt = eigvecs @ np.diag(eigvals ** -0.5) @ eigvecs.T

    return np.einsum("cd,ndt->nct", R_inv_sqrt, X).astype(np.float32)


class EEGDataset(Dataset):
    """
    Loads preprocessed EEG epochs and applies per-subject Euclidean Alignment.

    Args:
        epochs_path: path to .npy file, shape (N, 56, 260)
        labels_path: path to .npy file, shape (N,) — None for test set
        users_path:  path to .npy file, shape (N,) int subject IDs
        subject_ids: list of subject IDs to include — None means all
        augment: if True, apply Gaussian noise + random time shift
    """

    def __init__(self, epochs_path, labels_path, users_path,
                 subject_ids=None, augment=False):
        epochs = np.load(epochs_path)
        users = np.load(users_path)
        labels = np.load(labels_path) if labels_path is not None else None

        if subject_ids is not None:
            mask = np.isin(users, subject_ids)
            epochs = epochs[mask]
            users = users[mask]
            if labels is not None:
                labels = labels[mask]

        aligned = np.zeros_like(epochs)
        for uid in np.unique(users):
            idx = users == uid
            aligned[idx] = euclidean_align(epochs[idx])

        self.X = aligned
        self.y = labels
        self.users = users
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].copy()
        if self.augment:
            x += np.random.randn(*x.shape).astype(np.float32) * 0.1
            shift = np.random.randint(-20, 21)
            x = np.roll(x, shift, axis=-1)
        x = torch.from_numpy(x)
        if self.y is not None:
            return x, torch.tensor(self.y[i], dtype=torch.float32)
        return x
