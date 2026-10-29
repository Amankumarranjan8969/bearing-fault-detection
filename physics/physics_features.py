"""Physics feature branch: energy at BPFO/BPFI/BSF/FTF (+harmonics) in the envelope spectrum."""
import numpy as np
from preprocessing.filtering import envelope_signal
from .characteristic_frequency import characteristic_frequencies, PHYS_KEYS, FAULT_TO_FREQ


def envelope_spectrum(x, fs, band=None):
    env = envelope_signal(x, fs, band)
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env)))) / (len(env) / 2)
    return np.fft.rfftfreq(len(env), 1 / fs), spec


def physics_features(x, fs, rpm, geom, band=None, tol_pct=3.0, harmonics=3):
    """Returns (features dict, evidence dict). Features are ML inputs; evidence feeds the dashboard/LLM."""
    freqs, spec = envelope_spectrum(x, fs, band)
    df = freqs[1] - freqs[0]
    valid = freqs >= 2.0
    total = float(np.sum(spec[valid] ** 2)) + 1e-12
    med = float(np.median(spec[valid])) + 1e-12
    cf = characteristic_frequencies(rpm, geom)
    feats, evidence = {}, {}
    for k in PHYS_KEYS:
        e = 0.0
        for h in range(1, harmonics + 1):
            c = h * cf[k]
            if c >= freqs[-1]:
                break
            half = max(tol_pct / 100 * c, 1.5 * df)
            m = (freqs >= c - half) & (freqs <= c + half)
            e += float(np.sum(spec[m] ** 2))
        feats[f"{k}_energy"] = e
        feats[f"{k}_ratio"] = e / total
        best = {"expected_hz": float(cf[k]), "observed_hz": None, "amplitude": 0.0,
                "peak_to_median": 0.0, "harmonic": 1}
        for h in (1, 2):                      # ball faults often appear at 2 x BSF
            c = h * cf[k]
            if c >= freqs[-1]:
                break
            half = max(tol_pct / 100 * c, 1.5 * df)
            m = (freqs >= c - half) & (freqs <= c + half)
            if m.any():
                i = int(np.argmax(spec[m]))
                apk = float(spec[m][i])
                if apk / med > best["peak_to_median"]:
                    best = {"expected_hz": float(c), "observed_hz": float(freqs[m][i]), "amplitude": apk,
                            "peak_to_median": apk / med, "harmonic": h}
        evidence[k] = best
    return feats, evidence


def evidence_supports(fault, evidence, min_peak_ratio=3.0):
    """True/False: does the spectrum contain a peak at the predicted fault's frequency?
    None for healthy (no fault frequency expected)."""
    key = FAULT_TO_FREQ.get(fault)
    if key is None:
        return None
    return bool(evidence[key]["peak_to_median"] >= min_peak_ratio)
