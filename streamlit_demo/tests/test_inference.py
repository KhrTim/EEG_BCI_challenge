# streamlit_demo/tests/test_inference.py
import io, os, sys
import numpy as np
import torch
import pytest

_DEMO = os.path.join(os.path.dirname(__file__), "..")
_EEGCONFORMER = os.path.join(_DEMO, "..", "eegconformer")
sys.path.insert(0, _DEMO)
sys.path.insert(0, _EEGCONFORMER)

from inference import load_model, run_inference
from model import EEGConformer


def _save_fake_checkpoint() -> bytes:
    model = EEGConformer(in_channels=56, num_layers=4)
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    return buf.read()


class MockFile:
    def __init__(self, content: bytes, name: str = "ckpt.pt"):
        self.name = name
        self._content = content

    def read(self):
        return self._content


@pytest.fixture
def checkpoint_file():
    return MockFile(_save_fake_checkpoint())


def test_load_model_returns_eegconformer(checkpoint_file):
    net = load_model(checkpoint_file)
    assert isinstance(net, EEGConformer)


def test_load_model_eval_mode(checkpoint_file):
    net = load_model(checkpoint_file)
    assert not net.training


def test_run_inference_output_types(checkpoint_file):
    net = load_model(checkpoint_file)
    epoch = np.random.randn(56, 260).astype(np.float32)
    prob, attn = run_inference(net, epoch)
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0
    assert isinstance(attn, np.ndarray)
    assert attn.shape == (13,)


def test_run_inference_attn_sums_to_one(checkpoint_file):
    net = load_model(checkpoint_file)
    epoch = np.random.randn(56, 260).astype(np.float32)
    _, attn = run_inference(net, epoch)
    assert abs(attn.sum() - 1.0) < 1e-5
