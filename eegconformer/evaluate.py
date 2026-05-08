# eegconformer/evaluate.py
import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from dataset import EEGDataset
from model import EEGConformer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

WINNER_FOLD_CV_AUC = 0.7294
WINNER_LB_AUC = 0.8458


def load_model(ckpt_path, device):
    model = EEGConformer(in_channels=56).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def predict(model, loader, device):
    preds = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)
            logits = model(x).squeeze(1).cpu().numpy()
            preds.extend(logits)
    return np.array(preds)


def evaluate_cv(device="cuda"):
    """Report mean AUC across all fold checkpoints on their respective val sets."""
    epochs_path = os.path.join(DATA_DIR, "epochs.npy")
    labels_path = os.path.join(DATA_DIR, "labels.npy")
    users_path = os.path.join(DATA_DIR, "users.npy")

    all_users = np.unique(np.load(users_path))
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "fold_*.pt")))
    if not ckpts:
        print("No checkpoints found. Run train.py first.")
        return

    from sklearn.model_selection import KFold
    import re
    aucs = []
    for ckpt in ckpts:
        m = re.search(r"rep(\d+)_fold(\d+)", ckpt)
        rep, fold = int(m.group(1)), int(m.group(2))
        kf = KFold(n_splits=4, shuffle=True, random_state=rep * 42)
        splits = list(kf.split(all_users))
        _, val_idx = splits[fold]
        val_subjects = all_users[val_idx].tolist()

        ds = EEGDataset(epochs_path, labels_path, users_path,
                        subject_ids=val_subjects, augment=False)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

        model = load_model(ckpt, device)
        preds = predict(model, loader, device)
        auc = roc_auc_score(ds.y, preds)
        aucs.append(auc)
        print(f"  {os.path.basename(ckpt)}: val_auc={auc:.4f}")

    mean_auc = np.mean(aucs)
    print(f"\n=== EEGConformer CV AUC: {mean_auc:.4f} ===")
    print(f"    Winner fold-CV AUC:  {WINNER_FOLD_CV_AUC:.4f}")
    delta = mean_auc - WINNER_FOLD_CV_AUC
    print(f"    Delta: {delta:+.4f} ({'BEATS' if delta > 0 else 'BELOW'} baseline)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    evaluate_cv(args.device)
