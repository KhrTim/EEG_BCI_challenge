# eegconformer/tests/test_dataset.py
import numpy as np
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset import EEGDataset


def _make_fake_data(tmpdir, n=200, n_subjects=4):
    rng = np.random.default_rng(1)
    epochs = rng.standard_normal((n, 56, 260)).astype(np.float32)
    labels = rng.integers(0, 2, n).astype(np.int32)
    users = np.repeat(np.arange(1, n_subjects + 1), n // n_subjects).astype(np.int32)
    ep = os.path.join(tmpdir, "epochs.npy")
    lb = os.path.join(tmpdir, "labels.npy")
    us = os.path.join(tmpdir, "users.npy")
    np.save(ep, epochs); np.save(lb, labels); np.save(us, users)
    return ep, lb, us


def test_dataset_length_and_shape():
    with tempfile.TemporaryDirectory() as tmp:
        ep, lb, us = _make_fake_data(tmp, n=200, n_subjects=4)
        ds = EEGDataset(ep, lb, us)
        assert len(ds) == 200
        x, y = ds[0]
        assert x.shape == (56, 260)
        assert y.shape == ()


def test_dataset_subject_filter():
    with tempfile.TemporaryDirectory() as tmp:
        ep, lb, us = _make_fake_data(tmp, n=200, n_subjects=4)
        ds = EEGDataset(ep, lb, us, subject_ids=[1, 2])
        assert len(ds) == 100
