"""Train SVM / RandomForest / LightGBM on the feature table with bearing-level splits.
    python -m ml.train_ml --features data/features/synthetic_features.csv [--tune]"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from configs.config import load_config
from evaluation.metrics import classification_metrics, inference_time_ms
from feature_engineering.build_features import feature_groups
from feature_engineering.cleaning import clean_features
from ml import svm, random_forest, lgbm
from ml.feature_selection import select_features
from ml.tuning import tune
from models import registry

MODELS = {"svm": (svm.build, svm.PARAMS), "random_forest": (random_forest.build, random_forest.PARAMS),
          "lightgbm": (lgbm.build, lgbm.PARAMS)}
OUT = Path("models/ml")


def train_all(df, cfg, feature_cols, do_tune=False, k=None, models=("svm", "random_forest", "lightgbm")):
    """Returns (results dict, best bundle). Selection uses VAL macro-F1; TEST is only reported."""
    classes = cfg["classes"]
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    y = lambda d: d["label"].map(classes.index).to_numpy()
    k = k or cfg["ml"]["n_features"]
    sel = select_features(tr[feature_cols], y(tr), k, cfg["seed"])
    results, best = {}, None
    for name in models:
        build, grid = MODELS[name]
        est = build(cfg["seed"])
        params = {}
        if do_tune:
            est, params, _ = tune(est, grid, tr[sel], y(tr), tr["bearing_id"], cfg["ml"]["tune_iter"],
                                  cfg["ml"]["cv_folds"], cfg["seed"])
        else:
            est.fit(tr[sel], y(tr))
        res = {}
        for tag, d in (("val", va), ("test", te)):
            if len(d):
                res[tag] = classification_metrics(y(d), est.predict(d[sel]), est.predict_proba(d[sel]),
                                                  labels=list(range(len(classes))))
        res["inference_ms"] = inference_time_ms(est.predict_proba, tr[sel].iloc[:64])
        res["params"] = {k_: str(v) for k_, v in params.items()}
        results[name] = res
        score = res["val"]["f1_macro"] if "val" in res else 0.0
        if best is None or score >= best["val_f1"]:      # ties -> later model (tree models keep importances)
            best = {"name": name, "model": est, "features": sel, "classes": classes, "val_f1": score,
                    "test_f1": res.get("test", {}).get("f1_macro"), "backend": lgbm.BACKEND, "metrics": res}
    return results, best


def save_bundle(bundle, df_train, cfg, path=OUT / "ml_bundle.joblib", dataset="synthetic"):
    """Adds the healthy-baseline health index, then stores everything the API needs."""
    from health.health_index import HealthIndex
    path = Path(path)
    hf = [c for c in cfg["health"]["features"] if c in df_train.columns]
    healthy = df_train[df_train.label == "healthy"][hf].to_numpy()
    faulty = df_train[df_train.label != "healthy"][hf].to_numpy()
    bundle["health_index"] = HealthIndex(hf).fit(healthy, faulty)
    bundle["bg_mean"] = df_train[bundle["features"]].mean().to_dict()      # for local explanations
    bundle["bg_std"] = df_train[bundle["features"]].std().to_dict()
    bundle["dataset"] = dataset
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    registry.register(f"ml-{bundle['name']}", "ml", dataset, bundle["features"], {"backend": bundle["backend"]},
                      bundle["val_f1"], bundle["test_f1"], path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features/synthetic_features.csv")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--k", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config()
    df = pd.read_csv(a.features)
    meta = ["recording_id", "bearing_id", "dataset", "label", "split", "window_idx"]
    df, cols, rep = clean_features(df, [c for c in df.columns if c not in meta])
    print("cleaning:", {k: v for k, v in rep.items() if v})
    results, best = train_all(df, cfg, cols, a.tune or cfg["ml"]["tune"], a.k)
    for n, r in results.items():
        t = r.get("test", {})
        print(f"{n:14s} val macro-F1 {r['val']['f1_macro']:.3f} | test macro-F1 {t.get('f1_macro', float('nan')):.3f} "
              f"| test acc {t.get('accuracy', float('nan')):.3f} | {r['inference_ms']:.3f} ms/sample")
    path = save_bundle(best, df[df.split == "train"], cfg)
    Path("evaluation/results").mkdir(parents=True, exist_ok=True)
    Path("evaluation/results/ml_results.json").write_text(json.dumps(results, indent=2, default=str))
    print("saved best model:", best["name"], "->", path)


if __name__ == "__main__":
    main()
