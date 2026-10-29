"""LightGBM if installed, otherwise scikit-learn's HistGradientBoosting (same idea, so the pipeline still runs)."""
PARAMS_LGBM = {"learning_rate": [0.03, 0.05, 0.1], "n_estimators": [200, 400], "num_leaves": [15, 31, 63],
               "max_depth": [-1, 8], "reg_lambda": [0.0, 1.0, 5.0]}
PARAMS_HGB = {"learning_rate": [0.03, 0.05, 0.1], "max_iter": [200, 400], "max_leaf_nodes": [15, 31, 63],
              "l2_regularization": [0.0, 1.0, 5.0]}

try:
    import lightgbm as lgb
    BACKEND, PARAMS = "lightgbm", PARAMS_LGBM
except ImportError:
    BACKEND, PARAMS = "sklearn-hgb", PARAMS_HGB


def build(seed=42):
    if BACKEND == "lightgbm":
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, class_weight="balanced",
                                  random_state=seed, verbose=-1)
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(class_weight="balanced", random_state=seed)
