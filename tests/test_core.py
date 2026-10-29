import numpy as np
import pandas as pd
from configs.config import load_config
from physics.bearing_geometry import BearingGeometry
from physics.characteristic_frequency import characteristic_frequencies
from physics.physics_features import physics_features, evidence_supports
from preprocessing.segmentation import segment
from preprocessing.splitting import group_split, assert_no_leakage
from preprocessing.cleaning import validate_recording
from preprocessing.synthetic import make_signals
from preprocessing.filtering import preprocess
from health.health_index import HealthIndex

CFG = load_config()
GEOM = BearingGeometry.from_dict(CFG["geometry"]["synthetic"])


def test_cwru_characteristic_frequencies():
    cf = characteristic_frequencies(1797, BearingGeometry.from_dict(CFG["geometry"]["cwru_6205"]))
    assert abs(cf["BPFO"] - 107.36) < 0.1 and abs(cf["BPFI"] - 162.19) < 0.1 and abs(cf["FTF"] - 11.93) < 0.05


def test_segment_shape():
    assert segment(np.random.randn(3, 10000), 4096, 0.5).shape == (3, 3, 4096)


def test_group_split_has_no_leakage():
    rows = [{"bearing_id": f"{c}{b}", "label": c} for c in "abcd" for b in range(8) for _ in range(2)]
    meta = pd.DataFrame(rows)
    meta["split"] = group_split(meta)
    assert_no_leakage(meta)
    assert set(meta.split) == {"train", "val", "test"}


def test_validation_flags_bad_signals():
    ok, why = validate_recording({"vibration": np.array([1.0, np.nan, 2.0])}, fs=1)
    assert not ok and any("NaN" in w for w in why)


def test_physics_evidence_finds_outer_race_peak():
    sig, _ = make_signals("outer_race", 1800, GEOM, 12000, 8192, severity=1.0, seed=1)
    x = preprocess(sig["vibration"], 12000, CFG["signal"]["band_hz"])
    _, ev = physics_features(x, 12000, 1800, GEOM, CFG["signal"]["envelope_band_hz"])
    assert evidence_supports("outer_race", ev, 3.0)
    assert ev["BPFO"]["peak_to_median"] > ev["BPFI"]["peak_to_median"]


def test_health_index_orders_healthy_above_shifted():
    rng = np.random.default_rng(0)
    h = rng.normal(0, 1, (300, 4))
    hi = HealthIndex(list("abcd")).fit(h, h + 6)
    assert hi.score(h).mean() > 80 and hi.score(h + 6).mean() < 20
