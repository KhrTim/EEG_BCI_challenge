# streamlit_demo/processing.py
import io
import re
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter

FREQ = 200.0
EPOCH_SAMPLES = 260  # 1.3 s × 200 Hz


def bandpass(sig: np.ndarray, fs: float = FREQ) -> np.ndarray:
    """sig: (T, C) float32 → bandpass-filtered (T, C) float32, 1–40 Hz."""
    B, A = butter(5, np.array([1.0, 40.0]) / (fs / 2), btype="bandpass")
    return lfilter(B, A, sig, axis=0).astype(np.float32)


def _euclidean_align(X: np.ndarray) -> np.ndarray:
    """X: (N, C, T) → EA-whitened (N, C, T). Identical to eegconformer/dataset.py."""
    N, C, T = X.shape
    covs = np.einsum("nct,ndt->ncd", X, X) / T
    R_mean = covs.mean(axis=0)
    eigvals, eigvecs = np.linalg.eigh(R_mean)
    eigvals = np.maximum(eigvals, 1e-10)
    R_inv_sqrt = eigvecs @ np.diag(eigvals ** -0.5) @ eigvecs.T
    return np.einsum("cd,ndt->nct", R_inv_sqrt, X).astype(np.float32)


def epoch_csv_files(uploaded_files):
    """
    Parse and epoch a list of CSV files.

    Each file must have .name (e.g. "Data_S02_Sess01.csv") and .read() → bytes.
    EEG signal layout: col 0 ignored, cols 1:-2 are 56 EEG channels, col -1 is trigger.

    Returns:
        sessions:  dict[(user, session)] = {"epochs": (N,56,260) float32, "fb_ids": list[str]}
        subjects:  sorted list of unique user IDs (int)
        ch_names:  list of 56 channel name strings (from first file's CSV header)
    """
    reg = re.compile(r"\d+")
    sessions: dict = {}
    ch_names: list = []

    for f in uploaded_files:
        name = getattr(f, "name", "")
        nums = reg.findall(name)
        if len(nums) < 2:
            continue
        user, session = int(nums[0]), int(nums[1])

        raw = f.read()
        buf = io.BytesIO(raw) if isinstance(raw, bytes) else io.StringIO(raw)
        df = pd.read_csv(buf)

        file_ch = list(df.columns[1:-2])
        if not ch_names:
            ch_names = file_ch
        elif len(file_ch) != len(ch_names):
            raise ValueError(
                f"{name}: expected {len(ch_names)} EEG channels, got {len(file_ch)}"
            )

        sig = df.values.astype(np.float32)
        eeg = sig[:, 1:-2]    # (T, 56)
        trigger = sig[:, -1]  # (T,)

        eeg_f = bandpass(eeg)
        idx_fb = np.where(trigger == 1)[0]

        epochs, fb_ids = [], []
        for fb_num, idx in enumerate(idx_fb):
            ep = eeg_f[idx: idx + EPOCH_SAMPLES, :]
            if ep.shape[0] < EPOCH_SAMPLES:
                continue
            epochs.append(ep.T)  # (56, 260)
            fb_ids.append(f"S{user:02d}_Sess{session:02d}_FB{fb_num + 1:03d}")

        if epochs:
            sessions[(user, session)] = {
                "epochs": np.array(epochs, dtype=np.float32),
                "fb_ids": fb_ids,
            }

    if not ch_names:
        ch_names = [f"CH{i:02d}" for i in range(1, 57)]

    subjects = sorted({k[0] for k in sessions})
    return sessions, subjects, ch_names


def apply_ea_to_sessions(sessions: dict) -> dict:
    """
    Apply Euclidean Alignment per subject (pooling across all sessions for that subject).
    Returns new dict with aligned epochs; fb_ids unchanged.
    """
    user_to_keys: dict = {}
    for (user, session) in sessions:
        user_to_keys.setdefault(user, []).append((user, session))

    aligned: dict = {}
    for user, keys in user_to_keys.items():
        all_epochs = np.concatenate([sessions[k]["epochs"] for k in keys], axis=0)
        ea_all = _euclidean_align(all_epochs)
        offset = 0
        for k in keys:
            n = len(sessions[k]["epochs"])
            aligned[k] = {
                "epochs": ea_all[offset: offset + n],
                "fb_ids": sessions[k]["fb_ids"],
            }
            offset += n
    return aligned


def build_label_dict(labels_file):
    """
    labels_file: object with .read() → bytes, or None.
    Returns {fb_id: int} from TrainLabels.csv, or None if no file.
    """
    if labels_file is None:
        return None
    raw = labels_file.read()
    buf = io.BytesIO(raw) if isinstance(raw, bytes) else io.StringIO(raw)
    df = pd.read_csv(buf)
    return dict(zip(df["IdFeedBack"], df["Prediction"].astype(int)))
