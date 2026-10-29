"""ML ablation: A signals only | B + operating conditions | C + BPFO/BPFI/BSF/FTF physics features.
    python -m evaluation.ablation --features data/features/synthetic_features.csv
Results are computed, never invented. DL ablation (D-H): python -m evaluation.ablation --dl ..."""
import argparse
import pandas as pd
from configs.config import load_config
from feature_engineering.build_features import feature_groups
from feature_engineering.cleaning import clean_features
from ml.train_ml import train_all

META = ["recording_id", "bearing_id", "dataset", "label", "split", "window_idx"]


def run_ml_ablation(df, cfg, model="lightgbm"):
    g = feature_groups([c for c in df.columns if c not in META])
    sets = {"A signals": g["signals"], "B signals+operating": g["signals"] + g["operating"],
            "C signals+operating+physics": g["signals"] + g["operating"] + g["physics"]}
    rows = []
    for name, cols in sets.items():
        d, keep, _ = clean_features(df, cols)
        res, _ = train_all(d, cfg, keep, models=(model,))
        r = res[model]["test"]
        rows.append({"experiment": name, "n_features": len(keep), "accuracy": r["accuracy"],
                     "macro_f1": r["f1_macro"]})
    return pd.DataFrame(rows)


DL_EXPERIMENTS = [("D cnn", "cnn"), ("E multimodal", "multimodal"), ("F physics-informed", "pi"),
                  ("G domain adaptation", "da"), ("H physics + domain adaptation", "pi_da")]


def run_dl_ablation(source, target=None, epochs=None):
    from deep_learning.train_dl import train
    rows = []
    for label, name in DL_EXPERIMENTS:
        if name in ("da", "pi_da") and target is None:
            continue
        m = train(name, source, target, epochs=epochs)
        rows.append({"experiment": label, "source_test_macro_f1": m["source_test"]["f1_macro"],
                     "target_macro_f1": (m.get("target") or {}).get("f1_macro")})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features/synthetic_features.csv")
    ap.add_argument("--dl", action="store_true")
    ap.add_argument("--source", default="data/processed/synthetic_windows.npz")
    ap.add_argument("--target", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    if a.dl:
        out = run_dl_ablation(a.source, a.target, a.epochs)
    else:
        out = run_ml_ablation(pd.read_csv(a.features), load_config())
    print(out.to_string(index=False))
    out.to_csv("evaluation/results/ablation_dl.csv" if a.dl else "evaluation/results/ablation_ml.csv", index=False)
