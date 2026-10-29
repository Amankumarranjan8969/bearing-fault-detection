"""Local explanations for the tabular models. SHAP TreeExplainer when installed and applicable,
otherwise an importance-weighted deviation proxy (clearly labelled in the `method` field)."""
import numpy as np


def _shap_values(model, x, class_idx):
    import shap
    sv = shap.TreeExplainer(model).shap_values(x)
    if isinstance(sv, list):                      # older shap: list per class
        return np.asarray(sv[class_idx])[0]
    sv = np.asarray(sv)
    return sv[0, :, class_idx] if sv.ndim == 3 else sv[0]


def explain_local(model, x_row, features, mean, std, class_idx=0, top=8, use_shap=False):
    """x_row: 1-D array in `features` order. mean/std: training statistics of those features."""
    x = np.asarray(x_row, float).reshape(1, -1)
    method, contrib = None, None
    if use_shap:
        try:
            contrib, method = _shap_values(model, x, class_idx), "shap"
        except Exception:
            contrib = None
    if contrib is None:
        z = (x[0] - np.asarray(mean)) / (np.asarray(std) + 1e-9)
        imp = getattr(model, "feature_importances_", None)
        imp = np.ones_like(z) / len(z) if imp is None else np.asarray(imp)
        contrib, method = imp * np.abs(z), "importance-weighted deviation (proxy, not SHAP)"
    order = np.argsort(-np.abs(contrib))[:top]
    return {"method": method, "features": [{"feature": features[i], "value": float(x[0, i]),
                                            "contribution": float(contrib[i])} for i in order]}


def global_importance(model, features, top=15):
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return []
    order = np.argsort(-imp)[:top]
    return [{"feature": features[i], "importance": float(imp[i])} for i in order]
