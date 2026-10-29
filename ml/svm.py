from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# The scaler lives INSIDE the pipeline, so it is fitted on training data only (no leakage).
PARAMS = {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.01, 0.001], "svc__kernel": ["rbf"]}


def build(seed=42):
    return Pipeline([("scaler", StandardScaler()),
                     ("svc", SVC(probability=True, class_weight="balanced", random_state=seed))])
