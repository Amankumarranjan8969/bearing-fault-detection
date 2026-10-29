"""Build windows (.npz), feature table (.csv) and recording metadata (.csv).
    python -m scripts.prepare_data --source synthetic
    python -m scripts.prepare_data --source synthetic --shift --name synthetic_target   # second 'domain'
    python -m scripts.prepare_data --source paderborn --root data/raw/paderborn
    python -m scripts.prepare_data --source cwru --root data/raw/cwru --catalog data/raw/cwru/catalog.csv
    (CWRU/IMS: resample with preprocessing.filtering.resample_to before domain adaptation.)"""
import argparse
from pathlib import Path
import numpy as np
from configs.config import load_config
from feature_engineering.build_features import build_dataset
from physics.bearing_geometry import BearingGeometry
from preprocessing import loaders, synthetic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "paderborn", "cwru"], default="synthetic")
    ap.add_argument("--root")
    ap.add_argument("--catalog")
    ap.add_argument("--name")
    ap.add_argument("--shift", action="store_true")
    ap.add_argument("--max-files", type=int)
    a = ap.parse_args()
    cfg = load_config()
    name = a.name or a.source
    gkey = {"synthetic": "synthetic", "paderborn": "paderborn_6203", "cwru": "cwru_6205"}[a.source]
    geom = BearingGeometry.from_dict(cfg["geometry"][gkey])
    if a.source == "synthetic":
        recs = synthetic.generate_dataset(geom, cfg["signal"]["fs"], shift=a.shift, seed=cfg["seed"], name=name)
    elif a.source == "paderborn":
        recs = loaders.load_paderborn(a.root, a.max_files)
        cfg["signal"].update(fs=loaders.PADERBORN_FS, band_hz=[10, 20000], envelope_band_hz=[2000, 20000], window=32768)
    else:
        recs = loaders.load_cwru(a.root, a.catalog)
    df, win, meta = build_dataset(recs, cfg, geom)
    Path("data/features").mkdir(parents=True, exist_ok=True)
    df.to_csv(f"data/features/{name}_features.csv", index=False)
    meta.to_csv(f"data/metadata/{name}_metadata.csv", index=False)
    np.savez_compressed(f"data/processed/{name}_windows.npz", **win)
    print(f"saved data/features/{name}_features.csv, data/processed/{name}_windows.npz")


if __name__ == "__main__":
    main()
