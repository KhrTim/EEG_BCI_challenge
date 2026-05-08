# eegconformer/tests/test_model_attn.py
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import EEGConformer


def test_return_attn_shapes():
    model = EEGConformer(in_channels=56, num_layers=4)
    model.eval()
    x = torch.randn(2, 56, 260)
    logit, attn_list = model(x, return_attn=True)
    T = EEGConformer.T_PRIME
    assert logit.shape == (2, 1), f"logit shape {logit.shape}"
    assert len(attn_list) == 4, f"expected 4 layers, got {len(attn_list)}"
    for i, a in enumerate(attn_list):
        assert a.shape == (2, 8, T, T), f"layer {i} attn shape {a.shape}"


def test_return_attn_false_unchanged():
    model = EEGConformer(in_channels=56, num_layers=4)
    model.eval()
    x = torch.randn(2, 56, 260)
    out = model(x)
    assert out.shape == (2, 1)


def test_checkpoint_compat():
    """nn.ModuleList keeps identical state_dict keys to nn.Sequential."""
    model = EEGConformer(in_channels=56, num_layers=4)
    sd = model.state_dict()
    assert "transformer.0.attn.in_proj_weight" in sd
    assert "transformer.3.attn.in_proj_weight" in sd
    model2 = EEGConformer(in_channels=56, num_layers=4)
    model2.load_state_dict(sd)  # must not raise
