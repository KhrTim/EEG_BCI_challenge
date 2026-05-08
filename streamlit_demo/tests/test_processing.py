# streamlit_demo/tests/test_processing.py
import io, os, sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from processing import bandpass, epoch_csv_files, apply_ea_to_sessions, build_label_dict

FREQ = 200.0
EPOCH_SAMPLES = 260
N_CHANNELS = 56
N_COLS = N_CHANNELS + 3  # col0=id, cols1-56=eeg, col57=misc, col58=trigger


class MockFile:
    def __init__(self, content: bytes, name: str):
        self.name = name
        self._content = content

    def read(self):
        return self._content


def make_fake_csv(user=2, session=1, n_fb=3) -> MockFile:
    n_rows = n_fb * 300 + EPOCH_SAMPLES
    data = np.zeros((n_rows, N_COLS), dtype=np.float32)
    data[:, 1 : N_CHANNELS + 1] = np.random.randn(n_rows, N_CHANNELS).astype(np.float32)
    for i in range(n_fb):
        data[i * 300, -1] = 1  # FeedBackEvent trigger
    cols = ["Time"] + [f"CH{c:02d}" for c in range(1, N_CHANNELS + 1)] + ["EOG", "FeedBackEvent"]
    df = pd.DataFrame(data, columns=cols)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return MockFile(buf.getvalue(), f"Data_S{user:02d}_Sess{session:02d}.csv")


def test_bandpass_shape():
    sig = np.random.randn(1000, N_CHANNELS).astype(np.float32)
    out = bandpass(sig)
    assert out.shape == sig.shape
    assert out.dtype == np.float32


def test_bandpass_attenuates_dc():
    sig = np.ones((1000, N_CHANNELS), dtype=np.float32)
    out = bandpass(sig)
    assert np.abs(out[500:]).mean() < 0.01


def test_epoch_csv_files_shapes():
    f = make_fake_csv(user=2, session=1, n_fb=3)
    sessions, subjects, ch_names = epoch_csv_files([f])
    assert (2, 1) in sessions
    assert sessions[(2, 1)]["epochs"].shape == (3, N_CHANNELS, EPOCH_SAMPLES)
    assert len(sessions[(2, 1)]["fb_ids"]) == 3
    assert subjects == [2]
    assert len(ch_names) == N_CHANNELS


def test_epoch_csv_files_fb_id_format():
    f = make_fake_csv(user=7, session=2, n_fb=2)
    sessions, _, _ = epoch_csv_files([f])
    ids = sessions[(7, 2)]["fb_ids"]
    assert ids[0] == "S07_Sess02_FB001"
    assert ids[1] == "S07_Sess02_FB002"


def test_apply_ea_preserves_shape():
    f = make_fake_csv(user=2, session=1, n_fb=4)
    sessions, _, _ = epoch_csv_files([f])
    aligned = apply_ea_to_sessions(sessions)
    assert aligned[(2, 1)]["epochs"].shape == (4, N_CHANNELS, EPOCH_SAMPLES)


def test_build_label_dict_parses_csv():
    rows = "IdFeedBack,Prediction\nS02_Sess01_FB001,1\nS02_Sess01_FB002,0\n"
    f = MockFile(rows.encode(), "TrainLabels.csv")
    d = build_label_dict(f)
    assert d["S02_Sess01_FB001"] == 1
    assert d["S02_Sess01_FB002"] == 0


def test_build_label_dict_none():
    assert build_label_dict(None) is None
