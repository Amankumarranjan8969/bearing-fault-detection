"""Three-stage selection: filter (MI + ANOVA) -> embedded (RF importance) -> wrapper (RFE). Fit on TRAIN only."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, f_classif, mutual_info_classif


def select_features(X, y, k=30, seed=42):
    cols = list(X.columns)
    if len(cols) <= k:
        return cols
    mi = mutual_info_classif(X.values, y, random_state=seed)
    f = np.nan_to_num(f_classif(X.values, y)[0])
    rank = pd.Series(mi, cols).rank(ascending=False) + pd.Series(f, cols).rank(ascending=False)
    stage1 = rank.sort_values().index[: min(3 * k, len(cols))].tolist()                      # filter
    rf = RandomForestClassifier(200, random_state=seed, n_jobs=-1).fit(X[stage1], y)
    imp = pd.Series(rf.feature_importances_, stage1).sort_values(ascending=False)
    stage2 = imp.index[: min(2 * k, len(stage1))].tolist()                                   # embedded
    rfe = RFE(RandomForestClassifier(100, random_state=seed, n_jobs=-1), n_features_to_select=k, step=0.2)
    rfe.fit(X[stage2], y)                                                                    # wrapper
    return [c for c, keep in zip(stage2, rfe.support_) if keep]
