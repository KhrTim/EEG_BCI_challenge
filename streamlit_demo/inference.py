# streamlit_demo/inference.py
import io
import os
import sys

import numpy as np
import torch

_EEGCONFORMER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eegconformer")
if _EEGCONFORMER not in sys.path:
    sys.path.insert(0, _EEGCONFORMER)
from model import EEGConformer  # noqa: E402

# Patch centers in ms for the 13 transformer time patches.
# Patch i covers input samples [i*15 : i*15+75]; center = (i*15 + 37.5) / 200 * 1000 ms.
PATCH_CENTERS_MS = [(i * 15 + 37.5) / 200.0 * 1000.0 for i in range(13)]


def _infer_arch(state: dict) -> dict:
    indices = [int(k.split(".")[1]) for k in state if k.startswith("transformer.")]
    num_layers = max(indices) + 1 if indices else 4
    emb_size = state["patch_embed.temporal.0.weight"].shape[0]
    return {"num_layers": num_layers, "emb_size": emb_size}


def load_model(checkpoint_file, device: str = "cpu") -> EEGConformer:
    """
    checkpoint_file: object with .read() → bytes, or a file path string.
    Loads an EEGConformer checkpoint saved by eegconformer/train.py (raw state_dict).
    Returns the model in eval mode on device. Architecture is inferred from the checkpoint.
    """
    if hasattr(checkpoint_file, "read"):
        buf = io.BytesIO(checkpoint_file.read())
        state = torch.load(buf, map_location=device, weights_only=True)
    else:
        state = torch.load(checkpoint_file, map_location=device, weights_only=True)

    arch = _infer_arch(state)
    model = EEGConformer(in_channels=56, **arch)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def run_inference(model: EEGConformer, epoch: np.ndarray, device: str = "cpu"):
    """
    epoch: (56, 260) numpy array — one EA-aligned EEG epoch.
    Returns:
        prob:  float in [0, 1] — probability of ERROR class (sigmoid of logit).
        attn:  (13,) numpy float32 — normalized mean attention per time patch,
               averaged over layers, heads, and query positions. Sums to 1.
    """
    if epoch.shape != (56, 260):
        raise ValueError(f"epoch must be shape (56, 260), got {epoch.shape}")
    x = torch.from_numpy(epoch.astype(np.float32)).unsqueeze(0).to(device)  # (1, 56, 260)
    with torch.no_grad():
        logit, attn_list = model(x, return_attn=True)

    prob = float(torch.sigmoid(logit.squeeze()).item())

    # attn_list: list of num_layers tensors, each (1, 8, 13, 13)
    # Stack → (L, 1, 8, 13, 13), mean over (layers=0, batch=1, heads=2, queries=3) → (13,)
    attn_stack = torch.stack(attn_list, dim=0)
    attn_mean = attn_stack.mean(dim=(0, 1, 2, 3))
    attn_norm = (attn_mean / (attn_mean.sum() + 1e-8)).cpu().numpy()
    return prob, attn_norm


def run_saliency(model: EEGConformer, epoch: np.ndarray, device: str = "cpu") -> np.ndarray:
    """
    Gradient-based channel importance for one epoch.
    Returns: (56,) float32 — mean |d logit / d input| over time, normalized to sum=1.
    """
    if epoch.shape != (56, 260):
        raise ValueError(f"epoch must be shape (56, 260), got {epoch.shape}")
    x = torch.from_numpy(epoch.astype(np.float32)).unsqueeze(0).to(device)
    x.requires_grad_(True)
    logit = model(x)
    logit.squeeze().backward()
    # grad: (1, 56, 260) → abs → mean over time → (56,)
    importance = x.grad.abs().squeeze(0).mean(dim=1).detach().cpu().numpy()
    importance = importance / (importance.sum() + 1e-8)
    return importance
