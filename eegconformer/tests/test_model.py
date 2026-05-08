# eegconformer/tests/test_model.py
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import EEGConformer


def test_forward_output_shape():
    model = EEGConformer(in_channels=56)
    x = torch.randn(4, 56, 260)
    out = model(x)
    assert out.shape == (4, 1), f"Expected (4, 1), got {out.shape}"


def test_forward_no_nan():
    model = EEGConformer(in_channels=56)
    x = torch.randn(4, 56, 260)
    out = model(x)
    assert not torch.isnan(out).any()


def test_patch_embedding_shape():
    from model import PatchEmbedding
    pe = PatchEmbedding(in_channels=56)
    x = torch.randn(4, 56, 260)
    out = pe(x)
    emb = pe.temporal[0].out_channels
    assert out.shape == (4, 13, emb), f"Expected (4, 13, {emb}), got {out.shape}"
