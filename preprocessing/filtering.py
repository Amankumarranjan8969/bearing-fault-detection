"""DC removal, detrending, band-pass filtering, resampling, envelope."""
from fractions import Fraction
import numpy as np
from scipy.signal import butter, sosfiltfilt, detrend as _detrend, hilbert, resample_poly


def remove_dc(x):
    x = np.asarray(x, float)
    return x - x.mean(axis=-1, keepdims=True)


def detrend(x):
    return _detrend(np.asarray(x, float), axis=-1, type="linear")


def bandpass(x, fs, lo, hi, order=4):
    nyq = fs / 2.0
    hi = min(hi, 0.95 * nyq)
    sos = butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, np.asarray(x, float), axis=-1)


def preprocess(x, fs, band=None):
    """Raw -> DC removed -> detrended -> band-passed."""
    x = detrend(remove_dc(x))
    if band:
        x = bandpass(x, fs, band[0], band[1])
    return x


def resample_to(x, fs_in, fs_out):
    """Bring datasets to a common sampling rate (needed for domain adaptation)."""
    if abs(fs_in - fs_out) < 1e-9:
        return np.asarray(x, float)
    f = Fraction(fs_out / fs_in).limit_denominator(1000)
    return resample_poly(np.asarray(x, float), f.numerator, f.denominator, axis=-1)


def envelope_signal(x, fs, band=None):
    """Hilbert envelope of the (optionally band-passed) signal, mean removed."""
    x = np.asarray(x, float)
    x = x - x.mean()
    if band:
        x = bandpass(x, fs, band[0], band[1])
    env = np.abs(hilbert(x))
    return env - env.mean()
