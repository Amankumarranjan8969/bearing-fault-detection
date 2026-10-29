from sklearn.ensemble import RandomForestClassifier

PARAMS = {"n_estimators": [200, 400], "max_depth": [None, 10, 20], "min_samples_split": [2, 5],
          "min_samples_leaf": [1, 2], "max_features": ["sqrt", 0.3]}


def build(seed=42):
    return RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=seed)
