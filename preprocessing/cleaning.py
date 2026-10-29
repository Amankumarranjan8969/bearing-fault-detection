"""Recording-level data-quality control. Never train on dirty signals."""
import hashlib
import numpy as np


def validate_recording(signals, fs, expected_fs=None, min_seconds=1.0, min_std=1e-8):
    """signals: dict channel -> 1-D array. Returns (ok, reasons)."""
    reasons = []
    if expected_fs and abs(fs - expected_fs) > 1e-6:
        reasons.append(f"sampling rate {fs} != expected {expected_fs}")
    lengths = {k: len(v) for k, v in signals.items()}
    if len(set(lengths.values())) > 1:
        reasons.append(f"channel length mismatch (synchronisation): {lengths}")
    for k, v in signals.items():
        v = np.asarray(v, float)
        if not np.isfinite(v).all():
            reasons.append(f"{k}: NaN/Inf")
        elif v.std() < min_std:
            reasons.append(f"{k}: flat/constant signal")
        if len(v) < min_seconds * fs:
            reasons.append(f"{k}: shorter than {min_seconds}s")
    return len(reasons) == 0, reasons


def find_duplicates(arrays):
    """Indices of recordings whose samples are byte-identical to an earlier one."""
    seen, dup = {}, []
    for i, a in enumerate(arrays):
        h = hashlib.md5(np.asarray(a, np.float32).tobytes()).hexdigest()
        if h in seen:
            dup.append((i, seen[h]))
        else:
            seen[h] = i
    return dup
