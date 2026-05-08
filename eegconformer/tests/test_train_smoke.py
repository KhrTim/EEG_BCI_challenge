# eegconformer/tests/test_train_smoke.py
import numpy as np
import torch
import tempfile, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train import train_one_fold, build_loaders


def test_train_one_fold_smoke():
    """Training on synthetic data must not crash and return valid AUC."""
    rng = np.random.default_rng(99)
    n = 100
    epochs = rng.standard_normal((n, 56, 260)).astype(np.float32)
    labels = rng.integers(0, 2, n).astype(np.int32)
    users = np.array([1] * 50 + [2] * 50, dtype=np.int32)

    with tempfile.TemporaryDirectory() as tmp:
        ep = os.path.join(tmp, "ep.npy"); np.save(ep, epochs)
        lb = os.path.join(tmp, "lb.npy"); np.save(lb, labels)
        us = os.path.join(tmp, "us.npy"); np.save(us, users)

        ckpt = os.path.join(tmp, "model.pt")
        val_auc = train_one_fold(
            epochs_path=ep, labels_path=lb, users_path=us,
            train_subjects=[1], val_subjects=[2],
            checkpoint_path=ckpt,
            max_epochs=2, patience=5, batch_size=16, device="cpu"
        )
        assert 0.0 <= val_auc <= 1.0
        assert os.path.exists(ckpt)
