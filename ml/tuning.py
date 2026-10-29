"""Group-aware hyper-parameter search: folds never split a bearing."""
import numpy as np
from sklearn.model_selection import GroupKFold, RandomizedSearchCV


def tune(estimator, param_dist, X, y, groups, n_iter=10, cv=3, seed=42):
    n_splits = max(2, min(cv, len(np.unique(groups))))
    folds = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
    rs = RandomizedSearchCV(estimator, param_dist, n_iter=n_iter, cv=folds, scoring="f1_macro",
                            random_state=seed, refit=True, error_score=0.0)
    rs.fit(X, y)
    return rs.best_estimator_, rs.best_params_, float(rs.best_score_)
