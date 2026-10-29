import numpy as np
from scipy.stats import kurtosis
from preprocessing.filtering import envelope_signal
from .frequency_features import spectrum, EPS


def envelope_features(x, fs, band=None):
    env = envelope_signal(x, fs, band)
    f, s = spectrum(env, fs)
    m = f >= 2.0
    f, s = f[m], s[m]
    pn = s ** 2 / (np.sum(s ** 2) + EPS)
    i = int(np.argmax(s))
    return {"rms": float(np.sqrt(np.mean(env ** 2))), "kurtosis": float(kurtosis(env)),
            "energy": float(np.sum(env ** 2)), "dom_freq": float(f[i]),
            "spec_entropy": float(-(pn * np.log(pn + EPS)).sum())}
