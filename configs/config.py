from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG = {
    "project": "bearing-fault-detection",
    "seed": 42,
    "classes": ["healthy", "inner_race", "outer_race", "ball"],
    "signal": {
        "fs": 12000,
        "window": 8192,
        "overlap": 0.5,
        "band_hz": [10, 5500],
        "envelope_band_hz": [1500, 5000],
        "channels": ["vibration", "current1", "current2"],
    },
    "geometry": {
        "cwru_6205": {"n_balls": 9, "ball_d_mm": 7.94, "pitch_d_mm": 39.04, "contact_deg": 0.0},
        "paderborn_6203": {"n_balls": 8, "ball_d_mm": 6.75, "pitch_d_mm": 28.5, "contact_deg": 0.0},
        "ims_zA2115": {"n_balls": 16, "ball_d_mm": 8.4074, "pitch_d_mm": 71.501, "contact_deg": 15.17},
        "synthetic": {"n_balls": 8, "ball_d_mm": 6.75, "pitch_d_mm": 28.5, "contact_deg": 0.0},
    },
    "physics": {
        "tolerance_pct": 3.0,
        "harmonics": 3,
        "min_peak_ratio": 3.0,
    },
    "health": {
        "features": ["vib_rms", "vib_kurtosis", "vib_crest", "vib_env_rms", "vib_env_kurtosis"],
        "alert_thresholds": {"normal": 70, "warning": 50, "degradation": 30},
        "fault_prob_warn": 0.8,
    },
}

def load_config(path=None):
    p = Path(path) if path else ROOT / "configs" / "config.yaml"
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    return loaded
        except Exception:
            pass
    return DEFAULT_CONFIG