# eegconformer/train.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from dataset import EEGDataset
from model import EEGConformer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def build_loaders(epochs_path, labels_path, users_path,
                  train_subjects, val_subjects, batch_size=64, device="cuda"):
    train_ds = EEGDataset(epochs_path, labels_path, users_path,
                          subject_ids=train_subjects, augment=True)
    val_ds = EEGDataset(epochs_path, labels_path, users_path,
                        subject_ids=val_subjects, augment=False)
    num_workers = 0 if device == "cpu" else 4
    pin_memory = device != "cpu"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader


def train_one_fold(epochs_path, labels_path, users_path,
                   train_subjects, val_subjects, checkpoint_path,
                   max_epochs=100, patience=20, batch_size=64,
                   device="cuda", lr=3e-4, weight_decay=1e-4):
    train_loader, val_loader = build_loaders(
        epochs_path, labels_path, users_path,
        train_subjects, val_subjects, batch_size, device
    )

    all_labels = train_loader.dataset.y
    n_pos = all_labels.sum()
    n_neg = len(all_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)

    model = EEGConformer(in_channels=56).to(device)
    if torch.cuda.device_count() > 1 and device != "cpu":
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    warmup_epochs = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(max_epochs - warmup_epochs, 1)
    )

    best_auc = 0.0
    patience_counter = 0

    for epoch in range(max_epochs):
        if epoch < warmup_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] = lr * (epoch + 1) / warmup_epochs

        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if epoch >= warmup_epochs:
            scheduler.step()

        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x).squeeze(1).cpu().numpy()
                all_logits.extend(logits)
                all_labels.extend(y.numpy())

        try:
            val_auc = roc_auc_score(all_labels, all_logits)
        except ValueError:
            # Only one class present in val set — fallback to 0.5
            val_auc = 0.5
        print(f"  Epoch {epoch+1:3d} | val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

    return best_auc


def run_cv(n_folds=4, n_repeats=3, batch_size=64, device="cuda"):
    epochs_path = os.path.join(DATA_DIR, "epochs.npy")
    labels_path = os.path.join(DATA_DIR, "labels.npy")
    users_path = os.path.join(DATA_DIR, "users.npy")

    all_users = np.unique(np.load(users_path))
    os.makedirs(CKPT_DIR, exist_ok=True)

    all_aucs = []
    for rep in range(n_repeats):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=rep * 42)
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_users)):
            train_subjects = all_users[train_idx].tolist()
            val_subjects = all_users[val_idx].tolist()
            ckpt = os.path.join(CKPT_DIR, f"fold_rep{rep}_fold{fold_idx}.pt")
            print(f"\n=== Rep {rep+1}/{n_repeats}, Fold {fold_idx+1}/{n_folds} ===")
            print(f"  Train: {train_subjects}")
            print(f"  Val:   {val_subjects}")
            auc = train_one_fold(
                epochs_path, labels_path, users_path,
                train_subjects, val_subjects, ckpt,
                batch_size=batch_size, device=device
            )
            all_aucs.append(auc)
            print(f"  Best val AUC: {auc:.4f}")

    mean_auc = np.mean(all_aucs)
    std_auc = np.std(all_aucs)
    print(f"\n=== CV Result: {mean_auc:.4f} ± {std_auc:.4f} ===")
    print(f"Baseline (winner no-leak): 0.7294 fold-CV, 0.8458 LB")
    return mean_auc, std_auc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    run_cv(args.folds, args.repeats, args.batch_size, args.device)
