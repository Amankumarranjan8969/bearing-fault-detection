import numpy as np


def segment(x, win, overlap=0.5):
    """x: (C, N) or (N,) -> (n_windows, C, win)."""
    x = np.atleast_2d(np.asarray(x))
    step = max(1, int(win * (1 - overlap)))
    n = (x.shape[-1] - win) // step + 1
    if n <= 0:
        return np.empty((0, x.shape[0], win), dtype=x.dtype)
    idx = np.arange(win)[None, :] + step * np.arange(n)[:, None]   # (n, win)
    return x[:, idx].transpose(1, 0, 2)
