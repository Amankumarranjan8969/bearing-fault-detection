import numpy as np
from scipy.stats import kurtosis, skew

EPS = 1e-12


def time_features(x):
    x = np.asarray(x, float)
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    absmean = float(np.mean(np.abs(x)))
    return {
        "mean": float(x.mean()), "rms": rms, "var": float(x.var()), "std": float(x.std()),
        "peak": peak, "p2p": float(np.ptp(x)),
        "kurtosis": float(kurtosis(x)), "skew": float(skew(x)),
        "crest": peak / (rms + EPS), "impulse": peak / (absmean + EPS),
        "shape": rms / (absmean + EPS),
        "clearance": peak / (float(np.mean(np.sqrt(np.abs(x)))) ** 2 + EPS),
    }
