# eegconformer/preproc.py
import numpy as np
import pandas as pd
import glob
import re
import os
from scipy.signal import butter, lfilter

DATA_ROOT = "/Users/timur/timur_dev/kaggle/inria-bci-challenge"
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
FREQ = 200.0
EPOCH_SAMPLES = int(1.3 * FREQ)  # 260


def bandpass(sig, band, fs):
    B, A = butter(5, np.array(band) / (fs / 2), btype="bandpass")
    return lfilter(B, A, sig, axis=0)


def process_split(split):
    folder = os.path.join(DATA_ROOT, split)
    files = sorted(glob.glob(os.path.join(folder, "Data_*.csv")))
    reg = re.compile(r"\d+")

    X, users, fb_ids = [], [], []
    for f in files:
        nums = reg.findall(os.path.basename(f))
        user, session = int(nums[0]), int(nums[1])
        sig = pd.read_csv(f).values.astype(np.float32)
        eeg = sig[:, 1:-2]        # 56 channels
        trigger = sig[:, -1]

        eeg_f = bandpass(eeg, [1.0, 40.0], FREQ).astype(np.float32)
        idx_fb = np.where(trigger == 1)[0]

        for fb_num, idx in enumerate(idx_fb):
            epoch = eeg_f[idx: idx + EPOCH_SAMPLES, :]
            if epoch.shape[0] < EPOCH_SAMPLES:
                continue
            X.append(epoch.T)  # (56, 260)
            users.append(user)
            fb_ids.append(f"S{user:02d}_Sess{session:02d}_FB{fb_num+1:03d}")

    return np.array(X, dtype=np.float32), np.array(users, dtype=np.int32), fb_ids


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Processing train...")
    X_train, users_train, ids_train = process_split("train")
    labels_csv = pd.read_csv(os.path.join(DATA_ROOT, "TrainLabels.csv"))
    label_map = dict(zip(labels_csv["IdFeedBack"], labels_csv["Prediction"].astype(np.int32)))
    labels_train = np.array([label_map.get(fid, -1) for fid in ids_train], dtype=np.int32)

    valid = labels_train >= 0
    X_train, users_train, labels_train = X_train[valid], users_train[valid], labels_train[valid]

    np.save(os.path.join(OUT_DIR, "epochs.npy"), X_train)
    np.save(os.path.join(OUT_DIR, "labels.npy"), labels_train)
    np.save(os.path.join(OUT_DIR, "users.npy"), users_train)
    print(f"Train: {X_train.shape}, labels: {np.bincount(labels_train)}")

    print("Processing test...")
    X_test, users_test, ids_test = process_split("test")
    np.save(os.path.join(OUT_DIR, "test_epochs.npy"), X_test)
    np.save(os.path.join(OUT_DIR, "test_users.npy"), users_test)
    np.save(os.path.join(OUT_DIR, "test_ids.npy"), np.array(ids_test))
    print(f"Test: {X_test.shape}")


if __name__ == "__main__":
    main()
