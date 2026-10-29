"""Recording list -> windows + feature table (+ physics targets) with bearing-level splits."""
import logging
import numpy as np
import pandas as pd
from preprocessing.cleaning import validate_recording
from preprocessing.filtering import preprocess
from preprocessing.segmentation import segment
from preprocessing.splitting import group_split, assert_no_leakage
from physics.bearing_geometry import BearingGeometry
from physics.physics_features import physics_features
from physics.characteristic_frequency import PHYS_KEYS
from .time_features import time_features
from .frequency_features import frequency_features
from .envelope_features import envelope_features
from .current_features import current_features

log = logging.getLogger(__name__)
CHANNELS = ["vibration", "current1", "current2"]
AUX_KEYS = ["rpm", "torque", "force", "temperature"]


def extract_window_features(win, fs, aux, geom, band_env=None, phys_cfg=None):
    """win: (C, L) with channel 0 = vibration. Returns (features, physics_evidence)."""
    phys_cfg = phys_cfg or {}
    f = {}
    vib = win[0]
    f.update({f"vib_{k}": v for k, v in time_features(vib).items()})
    f.update({f"vib_{k}": v for k, v in frequency_features(vib, fs).items()})
    f.update({f"vib_env_{k}": v for k, v in envelope_features(vib, fs, band_env).items()})
    for i in range(1, win.shape[0]):
        f.update({f"cur{i}_{k}": v for k, v in current_features(win[i], fs).items()})
    f.update({k: aux.get(k, np.nan) for k in AUX_KEYS})
    pf, evidence = physics_features(vib, fs, aux["rpm"], geom, band_env,
                                    phys_cfg.get("tolerance_pct", 3.0), phys_cfg.get("harmonics", 3))
    f.update({f"phys_{k}": v for k, v in pf.items()})
    return f, evidence


def feature_groups(columns):
    """Column groups used by the ablation study (A: signals, B: +operating, C: +physics)."""
    sig = [c for c in columns if c.startswith(("vib_", "cur"))]
    op = [c for c in columns if c in AUX_KEYS]
    ph = [c for c in columns if c.startswith("phys_")]
    return {"signals": sig, "operating": op, "physics": ph}


def build_dataset(recordings, cfg, geom=None, geometry_key="synthetic", verbose=True):
    """Returns (features_df, windows dict of arrays, recording metadata df)."""
    s, pcfg = cfg["signal"], cfg["physics"]
    geom = geom or BearingGeometry.from_dict(cfg["geometry"][geometry_key])
    classes = cfg["classes"]
    meta = pd.DataFrame([r["meta"] for r in recordings])
    meta["split"] = group_split(meta, "bearing_id", "label", cfg["split"]["ratios"], cfg["seed"])
    split_of = dict(zip(meta["recording_id"], meta["split"]))
    rows, X, AUX, PH = [], [], [], []
    for r in recordings:
        m = r["meta"]
        fs = m["fs"]
        ok, why = validate_recording(r["signals"], fs)
        if not ok:
            log.warning("skip %s: %s", m["recording_id"], why)
            continue
        chans = [preprocess(r["signals"][c], fs, s["band_hz"]) for c in CHANNELS if c in r["signals"]]
        segs = segment(np.stack(chans), s["window"], s["overlap"])
        aux = {k: m.get(k, np.nan) for k in AUX_KEYS}
        for j, w in enumerate(segs):
            f, _ = extract_window_features(w, fs, aux, geom, s["envelope_band_hz"], pcfg)
            rows.append({"recording_id": m["recording_id"], "bearing_id": m["bearing_id"],
                         "dataset": m["dataset"], "label": m["label"], "split": split_of[m["recording_id"]],
                         "window_idx": j, **f})
            X.append(w.astype(np.float32))
            AUX.append([aux[k] for k in AUX_KEYS])
            PH.append([f[f"phys_{k}_ratio"] for k in PHYS_KEYS])
    df = pd.DataFrame(rows)
    assert_no_leakage(df, "bearing_id", "split")
    win = {"X": np.stack(X), "aux": np.asarray(AUX, np.float32), "phys": np.asarray(PH, np.float32),
           "y": np.array([classes.index(l) if l in classes else -1 for l in df["label"]]),
           "split": df["split"].to_numpy().astype(str), "bearing": df["bearing_id"].to_numpy().astype(str),
           "classes": np.array(classes), "fs": np.float32(recordings[0]["meta"]["fs"])}
    if verbose:
        print(f"{len(df)} windows | {df['bearing_id'].nunique()} bearings | " + str(df["split"].value_counts().to_dict()))
    return df, win, meta


def meta_columns():
    return ["recording_id", "bearing_id", "dataset", "label", "split", "window_idx"]
