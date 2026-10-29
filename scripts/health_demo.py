"""NASA IMS-style degradation: snapshots -> features -> health index -> trend -> indicative time-to-threshold.
    python -m scripts.health_demo --synthetic                       # no data needed
    python -m scripts.health_demo --ims-dir data/raw/ims/2nd_test --channel 0
The health index is fitted on the first 20% of snapshots (assumed healthy). The time-to-threshold printed here is
an indicative extrapolation, NOT a validated RUL: validate on held-out run-to-failure bearings before claiming RUL."""
import argparse
import numpy as np
from configs.config import load_config
from feature_engineering.build_features import extract_window_features
from health.degradation import fit_trend, trend_label, time_to_threshold
from health.health_index import HealthIndex, smooth
from physics.bearing_geometry import BearingGeometry
from preprocessing import loaders, synthetic
from preprocessing.filtering import preprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--ims-dir")
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=20.0)
    a = ap.parse_args()
    cfg = load_config()
    if a.synthetic:
        geom = BearingGeometry.from_dict(cfg["geometry"]["synthetic"])
        fs, rpm, steps = int(cfg["signal"]["fs"]), 1800.0, 60
        sev = 0.02 + 1.5 * (np.arange(steps) / (steps - 1)) ** 3
        wins = [synthetic.make_window("outer_race", s, rpm, geom, fs, 8192, seed=i)[0]["vibration"] for i, s in enumerate(sev)]
    else:
        geom = BearingGeometry.from_dict(cfg["geometry"]["ims_zA2115"])
        snaps = loaders.load_ims_run(a.ims_dir, a.channel)
        fs, rpm = snaps[0]["meta"]["fs"], snaps[0]["meta"]["rpm"]
        wins = [s["signals"]["vibration"] for s in snaps]
    feats = [extract_window_features(preprocess(w, fs, [10, 0.45 * fs]).reshape(1, -1), fs, {"rpm": rpm}, geom,
                                     [0.1 * fs, 0.4 * fs], cfg["physics"])[0] for w in wins]
    cols = [c for c in cfg["health"]["features"] if c in feats[0]]
    F = np.array([[f[c] for c in cols] for f in feats])
    n0 = max(5, len(F) // 5)
    hi = HealthIndex(cols).fit(F[:n0], F[-max(3, len(F) // 10):])
    health = smooth(hi.score(F), 0.3)
    slope = fit_trend(health[-10:])
    print(f"{len(F)} snapshots | health start {health[0]:.0f} -> end {health[-1]:.0f} | recent trend: {trend_label(slope)}")
    print("indicative steps to threshold from the last snapshot:", time_to_threshold(float(health[-1]), slope, a.threshold))
    for i in range(0, len(health), max(1, len(health) // 12)):
        print(f"  snapshot {i:4d}  health {health[i]:5.1f}  " + "#" * int(health[i] / 4))


if __name__ == "__main__":
    main()
