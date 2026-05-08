# eegconformer/tests/test_ea.py
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset import euclidean_align


def test_ea_produces_near_identity_mean_cov():
    rng = np.random.default_rng(42)
    A = rng.standard_normal((56, 56))
    cov_true = A @ A.T + np.eye(56) * 0.1
    L = np.linalg.cholesky(cov_true)
    white = rng.standard_normal((100, 56, 260))
    X = np.einsum("cd,ndt->nct", L, white)

    X_aligned = euclidean_align(X)

    covs = np.einsum("nct,ndt->ncd", X_aligned, X_aligned) / 260
    mean_cov = covs.mean(axis=0)
    np.testing.assert_allclose(mean_cov, np.eye(56), atol=0.15)


def test_ea_preserves_shape():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 56, 260)).astype(np.float32)
    X_aligned = euclidean_align(X)
    assert X_aligned.shape == (50, 56, 260)
