"""Loaders for the three real datasets. Each returns recordings in the same format as synthetic.py:
{'signals': {channel: 1-D array}, 'meta': {...}}. Check file layouts against your local copies."""
import re
from pathlib import Path
import numpy as np
from scipy.io import loadmat

PADERBORN_FS = 64000


def _paderborn_label(code):
    if code.startswith("K0"):
        return "healthy"
    if code.startswith("KA"):
        return "outer_race"
    if code.startswith("KI"):
        return "inner_race"
    if code.startswith("KB"):
        return "ball"
    return None


def load_paderborn(root, max_files=None):
    """root: folder with extracted Paderborn .mat files, e.g. N15_M07_F04_KA01_1.mat"""
    pat = re.compile(r"N(\d+)_M(\d+)_F(\d+)_(K[A-Z0-9]+)_(\d+)")
    recs = []
    for i, p in enumerate(sorted(Path(root).rglob("*.mat"))):
        if max_files and len(recs) >= max_files:
            break
        m = pat.search(p.stem)
        if not m:
            continue
        label = _paderborn_label(m.group(4))
        if label is None:
            continue
        try:
            mat = loadmat(p, struct_as_record=False, squeeze_me=True)
            rec = mat[[k for k in mat if not k.startswith("__")][0]]
            ch = {y.Name: np.asarray(y.Data, float).ravel() for y in rec.Y}
        except Exception:
            continue
        pick = lambda pre: next((v for k, v in ch.items() if k.startswith(pre)), None)
        slow = lambda pre: float(np.mean(pick(pre))) if pick(pre) is not None else float("nan")
        if pick("vibration") is None or len(pick("vibration")) < 32768:
            continue
        recs.append({"signals": {"vibration": pick("vibration"), "current1": pick("phase_current_1"),
                                 "current2": pick("phase_current_2")},
                     "meta": {"recording_id": p.stem, "bearing_id": m.group(4), "dataset": "paderborn",
                              "label": label, "fs": PADERBORN_FS, "rpm": int(m.group(1)) * 100.0,
                              "torque": int(m.group(2)) / 10.0, "force": float(int(m.group(3)) * 100),
                              "temperature": slow("temp")}})
    return recs


def load_cwru(root, catalog_csv, fs=12000):
    """catalog_csv columns: file,label,bearing_id[,rpm]  (label in healthy/inner_race/outer_race/ball).
    CWRU has no current channel; the drive-end accelerometer is used."""
    import pandas as pd
    cat = pd.read_csv(catalog_csv)
    recs = []
    for _, r in cat.iterrows():
        mat = loadmat(Path(root) / r["file"])
        de = next(k for k in mat if k.endswith("DE_time"))
        rpm_key = next((k for k in mat if k.endswith("RPM")), None)
        rpm = float(r["rpm"]) if "rpm" in cat.columns and not np.isnan(r["rpm"]) else \
            (float(np.ravel(mat[rpm_key])[0]) if rpm_key else 1797.0)
        recs.append({"signals": {"vibration": np.ravel(mat[de]).astype(float)},
                     "meta": {"recording_id": Path(r["file"]).stem, "bearing_id": r["bearing_id"],
                              "dataset": "cwru", "label": r["label"], "fs": fs, "rpm": rpm,
                              "torque": float("nan"), "force": float("nan"), "temperature": float("nan")}})
    return recs


def load_ims_run(folder, channel=0, fs=20000, rpm=2000.0):
    """NASA IMS run-to-failure: one whitespace-separated file (20480 x n_channels) per snapshot,
    filename = timestamp. Returns snapshots in time order (used for health/degradation, not classification)."""
    files = sorted(p for p in Path(folder).iterdir() if p.is_file())
    snaps = []
    for k, p in enumerate(files):
        a = np.loadtxt(p)
        snaps.append({"signals": {"vibration": a[:, channel]},
                      "meta": {"recording_id": p.name, "bearing_id": f"ims_ch{channel}", "dataset": "ims",
                               "label": "unknown", "fs": fs, "rpm": rpm, "snapshot": k}})
    return snaps
