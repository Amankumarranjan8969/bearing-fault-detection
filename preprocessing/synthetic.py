"""Physics-based synthetic bearing signals so the whole pipeline runs without downloading data.
Impulses at the characteristic frequency excite a decaying structural resonance; the current
carries a weak amplitude modulation at the same frequency. NOT a substitute for real datasets."""
import numpy as np
from scipy.signal import fftconvolve
from physics.characteristic_frequency import characteristic_frequencies, FAULT_TO_FREQ

CLASSES = ["healthy", "inner_race", "outer_race", "ball"]
DEFAULT_CONDITIONS = [(1200, 0.7, 400), (1800, 0.7, 400)]      # rpm, torque Nm, radial force N


def _impulses(n, fs, f_c, rng, jitter=0.01):
    t = np.arange(0, n / fs, 1.0 / f_c)
    t = t + rng.normal(0, jitter / f_c, len(t))                 # slip
    idx = np.clip((t * fs).astype(int), 0, n - 1)
    x = np.zeros(n)
    np.add.at(x, idx, 1.0)
    return x


def _resonance(fs, f_res, zeta=0.05, dur=0.01):
    t = np.arange(int(dur * fs)) / fs
    return np.exp(-2 * np.pi * f_res * zeta * t) * np.sin(2 * np.pi * f_res * t)


def make_signals(fault, rpm, geom, fs, n, severity=1.0, load=0.7, f_res=3000.0, noise=0.25, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    fr = rpm / 60.0
    cf = characteristic_frequencies(rpm, geom)
    vib = 0.3 * np.sin(2 * np.pi * fr * t + rng.uniform(0, 6.28)) + 0.1 * np.sin(2 * np.pi * 2 * fr * t)
    vib += rng.normal(0, noise, n)
    m_cur = 0.0
    if fault != "healthy" and severity > 0:
        f_c = cf[FAULT_TO_FREQ[fault]]
        if fault == "ball":
            imp = _impulses(n, fs, 2 * f_c, rng) * (1 + 0.5 * np.cos(2 * np.pi * cf["FTF"] * t))
        elif fault == "inner_race":
            imp = _impulses(n, fs, f_c, rng) * (1 + 0.8 * np.cos(2 * np.pi * fr * t))
        else:
            imp = _impulses(n, fs, f_c, rng)
        vib += 3.0 * severity * (0.8 + 0.4 * load) * fftconvolve(imp, _resonance(fs, f_res))[:n]
        m_cur = 0.03 * severity
        f_mod = f_c
    else:
        f_mod = 0.0
    sup = 50.0
    i1 = (1 + m_cur * np.cos(2 * np.pi * f_mod * t)) * np.sin(2 * np.pi * sup * t) + rng.normal(0, 0.02, n)
    i2 = (1 + m_cur * np.cos(2 * np.pi * f_mod * t)) * np.sin(2 * np.pi * sup * t - 2 * np.pi / 3) + rng.normal(0, 0.02, n)
    temp = 40 + 4 * load + rng.normal(0, 1.5)                    # deliberately independent of the fault
    return {"vibration": vib, "current1": i1, "current2": i2}, {"temperature": float(temp)}


def generate_dataset(geom, fs=12000, seconds=3.0, bearings_per_class=8, conditions=None,
                     shift=False, seed=42, name="synthetic"):
    """List of recordings: {'signals': {...}, 'meta': {...}}. shift=True builds a different 'domain'
    (other resonance band, more noise, weaker faults) to demonstrate domain adaptation."""
    rng = np.random.default_rng(seed + (999 if shift else 0))
    conditions = conditions or DEFAULT_CONDITIONS
    n = int(seconds * fs)
    recs = []
    for c in CLASSES:
        for b in range(bearings_per_class):
            bid = f"{name}_{c}_{b:02d}"
            sev = rng.uniform(0.5, 1.2) * (0.7 if shift else 1.0)
            f_res = rng.uniform(2500, 3800) + (600 if shift else 0)
            noise = 0.25 * (1.5 if shift else 1.0) * rng.uniform(0.9, 1.1)
            for k, (rpm, torque, force) in enumerate(conditions):
                sig, extra = make_signals(c, rpm, geom, fs, n, sev, load=torque, f_res=f_res, noise=noise,
                                          seed=int(rng.integers(1 << 30)))
                recs.append({"signals": sig, "meta": {
                    "recording_id": f"{bid}_c{k}", "bearing_id": bid, "dataset": name, "label": c,
                    "fs": fs, "rpm": rpm, "torque": torque, "force": force, **extra}})
    return recs


def make_window(fault, severity, rpm, geom, fs=12000, n=8192, seed=0):
    sig, extra = make_signals(fault, rpm, geom, fs, n, severity=severity, seed=seed)
    return sig, {"fs": fs, "rpm": rpm, "torque": 0.7, "force": 400.0, **extra}
