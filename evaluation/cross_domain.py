"""Feature-level cross-domain check: train on source, test on target (vibration features only)."""
import pandas as pd
from configs.config import load_config
from evaluation.metrics import classification_metrics
from feature_engineering.cleaning import clean_features
from ml.train_ml import train_all


def cross_domain(src_df, tgt_df, cfg, model="lightgbm"):
    cols = [c for c in src_df.columns if c.startswith("vib_") and c in tgt_df.columns]
    src, keep, _ = clean_features(src_df, cols)
    res, best = train_all(src, cfg, keep, models=(model,))
    est, sel, classes = best["model"], best["features"], cfg["classes"]
    tgt = tgt_df.dropna(subset=sel)
    y = tgt["label"].map(classes.index)
    tgt_m = classification_metrics(y, est.predict(tgt[sel]), est.predict_proba(tgt[sel]), labels=list(range(len(classes))))
    src_f1 = res[model]["test"]["f1_macro"]
    return {"source_test_f1": src_f1, "target_f1": tgt_m["f1_macro"], "performance_drop": src_f1 - tgt_m["f1_macro"]}
