import numpy as np

EPS = 1e-12


def spectrum(x, fs):
    x = np.asarray(x, float) - np.mean(x)
    s = np.abs(np.fft.rfft(x * np.hanning(len(x)))) / (len(x) / 2)
    return np.fft.rfftfreq(len(x), 1 / fs), s


def spectral_stats(f, s, prefix=""):
    p = s ** 2
    tot = p.sum() + EPS
    pn = p / tot
    i = int(np.argmax(s[1:])) + 1
    cen = float((f * pn).sum())
    bw = float(np.sqrt(((f - cen) ** 2 * pn).sum()))
    return {f"{prefix}dom_freq": float(f[i]), f"{prefix}dom_amp": float(s[i]), f"{prefix}centroid": cen,
            f"{prefix}bandwidth": bw, f"{prefix}entropy": float(-(pn * np.log(pn + EPS)).sum()),
            f"{prefix}total_energy": float(tot), f"{prefix}freq_var": bw ** 2}


def frequency_features(x, fs, n_bands=8):
    f, s = spectrum(x, fs)
    out = spectral_stats(f, s)
    p = s ** 2
    tot = p.sum() + EPS
    edges = np.linspace(0, fs / 2, n_bands + 1)
    for b in range(n_bands):
        m = (f >= edges[b]) & (f < edges[b + 1])
        out[f"band{b}_energy"] = float(p[m].sum() / tot)
    return out
