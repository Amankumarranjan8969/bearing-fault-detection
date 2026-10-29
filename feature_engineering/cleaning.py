"""Feature cleaning. Correlated pairs are REPORTED, not deleted: physically meaningful correlation can help."""
import numpy as np
import pandas as pd


def clean_features(df, cols, max_bad_frac=0.05, min_rel_std=1e-9):
    """Returns (df_clean, kept_cols, report). Drops columns with many NaN/Inf, constant or near-zero-variance
    columns, then drops rows that still contain NaN/Inf."""
    report = {"dropped_nan_inf": [], "dropped_constant": []}
    d = df.copy()
    d[cols] = d[cols].replace([np.inf, -np.inf], np.nan)
    keep = []
    for c in cols:
        if d[c].isna().mean() > max_bad_frac:
            report["dropped_nan_inf"].append(c)
        elif d[c].nunique(dropna=True) <= 1 or d[c].std() <= min_rel_std * (abs(d[c].mean()) + 1e-12):
            report["dropped_constant"].append(c)
        else:
            keep.append(c)
    d = d.dropna(subset=keep).reset_index(drop=True)
    return d, keep, report


def correlated_pairs(df, cols, thr=0.98):
    c = df[cols].corr().abs()
    out = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if c.loc[a, b] >= thr:
                out.append((a, b, float(c.loc[a, b])))
    return out
