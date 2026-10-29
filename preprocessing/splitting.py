"""Group-level split: whole bearings go to train/val/test BEFORE windowing (prevents leakage)."""
import numpy as np


def group_split(meta, group_col="bearing_id", label_col="label", ratios=(0.70, 0.15, 0.15), seed=42):
    """meta: DataFrame (one row per recording). Returns Series of 'train'/'val'/'test'.
    Stratified at group level: each class keeps its own train/val/test bearings."""
    rng = np.random.default_rng(seed)
    out = {}
    for _, sub in meta.groupby(label_col):
        groups = sub[group_col].unique().tolist()
        rng.shuffle(groups)
        n = len(groups)
        n_val = max(1, round(ratios[1] * n)) if n >= 3 else 0
        n_test = max(1, round(ratios[2] * n)) if n >= 3 else 0
        # Ensure that if groups have fault-diameter prefixes (e.g. cwru_IR021_*), val+test don't consume 100% of a prefix
        if n_test + n_val >= 2 and n > n_test + n_val:
            prefix = lambda g: "_".join(str(g).split("_")[:-1])
            train_prefixes = {prefix(g) for g in groups[n_test + n_val:]}
            for idx in range(n_test, n_test + n_val):
                if prefix(groups[idx]) not in train_prefixes:
                    for swap_idx in range(n_test + n_val, n):
                        p_swap = prefix(groups[swap_idx])
                        if sum(1 for g in groups[n_test + n_val:] if prefix(g) == p_swap) > 1:
                            groups[idx], groups[swap_idx] = groups[swap_idx], groups[idx]
                            train_prefixes.add(prefix(groups[idx]))
                            break
        for i, g in enumerate(groups):
            out[g] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return meta[group_col].map(out)


def assert_no_leakage(df, group_col="bearing_id", split_col="split"):
    """Raises if any bearing/recording appears in more than one split."""
    bad = df.groupby(group_col)[split_col].nunique()
    bad = bad[bad > 1]
    if len(bad):
        raise AssertionError(f"Leakage: {group_col} in several splits: {bad.index.tolist()[:5]}")
