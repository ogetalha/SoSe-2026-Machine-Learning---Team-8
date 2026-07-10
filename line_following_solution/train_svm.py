#!/usr/bin/env python3
"""
Train an OpenCV SVM on real extracted CARLA frames.
Run once:  python docs/train_svm.py
Outputs:   models/svm_model.xml
           models/svm_mean.npy
           models/svm_std.npy
"""
import cv2
import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
FRAMES_DIR  = Path("extracted_images")
MODEL_DIR   = Path("models")
MAX_FRAMES  = 4000   # sample from your 46k — enough to train well
N_POS       = 30     # green pixels sampled per frame
N_NEG       = 90     # background pixels sampled per frame

# Used ONLY for auto-labelling ground truth (not for the SVM itself)
# CARLA green line: BGR(0,255,0) → HSV(60,255,255)
LOWER_GREEN = np.array([40, 150, 150])
UPPER_GREEN = np.array([80, 255, 255])


def extract_samples(img, rng):
    """Sample labelled HSV pixels from one frame."""
    h = img.shape[0]
    roi = img[int(h * 0.6):, :]                          # bottom 40%
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    green_mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN).astype(bool)
    pos_idx = np.argwhere(green_mask)
    neg_idx = np.argwhere(~green_mask)

    if len(pos_idx) == 0:
        return None, None                                 # no line visible

    pos_idx = pos_idx[rng.choice(len(pos_idx), min(N_POS, len(pos_idx)), replace=False)]
    neg_idx = neg_idx[rng.choice(len(neg_idx), min(N_NEG, len(neg_idx)), replace=False)]

    X_pos = hsv[pos_idx[:, 0], pos_idx[:, 1]].astype(np.float32)
    X_neg = hsv[neg_idx[:, 0], neg_idx[:, 1]].astype(np.float32)

    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([
        np.full(len(X_pos),  1, dtype=np.int32),   # green line  = +1
        np.full(len(X_neg), -1, dtype=np.int32),   # background  = -1
    ])
    return X, y


def main():
    rng = np.random.default_rng(42)
    MODEL_DIR.mkdir(exist_ok=True)

    # ── Load frame paths ──────────────────────────────────────────────────────
    frames = sorted(FRAMES_DIR.glob("*.png")) + sorted(FRAMES_DIR.glob("*.jpg"))
    print(f"Found {len(frames)} frames — using up to {MAX_FRAMES}")

    if len(frames) > MAX_FRAMES:
        idx = rng.choice(len(frames), MAX_FRAMES, replace=False)
        frames = [frames[i] for i in sorted(idx)]

    # ── Build dataset ─────────────────────────────────────────────────────────
    X_list, y_list = [], []
    skipped = 0

    for i, fpath in enumerate(frames):
        img = cv2.imread(str(fpath))
        if img is None:
            skipped += 1
            continue
        X, y = extract_samples(img, rng)
        if X is None:
            skipped += 1
            continue
        X_list.append(X)
        y_list.append(y)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(frames)} frames processed …")

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    print(f"\nDataset ready: {len(X)} samples  |  "
          f"green={int((y == 1).sum())}  background={int((y == -1).sum())}  "
          f"skipped_frames={skipped}")

    # ── Normalise (save mean/std so inference can apply same scaling) ─────────
    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-6
    X_norm = (X - mean) / std

    np.save(MODEL_DIR / "svm_mean.npy", mean)
    np.save(MODEL_DIR / "svm_std.npy",  std)

    # ── Train OpenCV SVM ──────────────────────────────────────────────────────
    print("Training SVM … (this may take a minute)")
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(1.0)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 5000, 1e-6))

    train_data = cv2.ml.TrainData_create(X_norm, cv2.ml.ROW_SAMPLE, y)
    svm.train(train_data)
    svm.save(str(MODEL_DIR / "svm_model.xml"))

    print(f"\nDone! Saved to {MODEL_DIR}/")
    print("  svm_model.xml")
    print("  svm_mean.npy")
    print("  svm_std.npy")


if __name__ == "__main__":
    main()