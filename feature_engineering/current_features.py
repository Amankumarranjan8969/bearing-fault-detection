import numpy as np
from scipy.stats import kurtosis
from .frequency_features import spectrum

EPS = 1e-12


def current_features(x, fs):
    x = np.asarray(x, float)
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    f, s = spectrum(x, fs)
    return {"rms": rms, "peak": peak, "var": float(x.var()), "kurtosis": float(kurtosis(x)),
            "crest": peak / (rms + EPS), "dom_freq": float(f[int(np.argmax(s[1:])) + 1]),
            "spec_energy": float(np.sum(s ** 2))}
