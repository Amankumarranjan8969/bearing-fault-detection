"""Trend, indicative time-to-threshold and alert logic. Time-to-threshold is NOT a validated RUL."""
import numpy as np

LEVELS = ["normal", "warning", "degradation", "critical"]


def fit_trend(values):
    """Least-squares slope (health points per step) over the given values."""
    v = np.asarray(values, float)
    if len(v) < 3:
        return None
    return float(np.polyfit(np.arange(len(v)), v, 1)[0])


def trend_label(slope, tol=0.2):
    if slope is None:
        return "insufficient data"
    return "decreasing" if slope < -tol else "increasing" if slope > tol else "stable"


def time_to_threshold(health, slope, threshold=20.0):
    """Indicative number of update steps until the health trend crosses `threshold`; None if not falling."""
    if slope is None or slope >= -1e-6 or health <= threshold:
        return None
    return float((health - threshold) / -slope)


def alert_level(health, p_fault, slope, temperature, supports, cfg_health):
    t = cfg_health["alert_thresholds"]
    lvl = 0 if health >= t["normal"] else 1 if health >= t["warning"] else 2 if health >= t["degradation"] else 3
    reasons = [f"health score {health:.0f}"]
    if p_fault >= cfg_health["fault_prob_warn"] and lvl < 1:
        lvl = 1
        reasons.append(f"fault probability {p_fault:.0%}")
    if slope is not None and slope <= -cfg_health["rapid_slope"]:
        lvl = min(lvl + 1, 3)
        reasons.append("rapid degradation")
    if temperature is not None and np.isfinite(temperature) and temperature >= cfg_health["temp_warn_c"]:
        lvl = max(lvl, 1)
        reasons.append(f"temperature {temperature:.0f} C")
    if supports is False and p_fault >= cfg_health["fault_prob_warn"]:
        reasons.append("physics check: no peak at the predicted fault frequency, verify manually")
    return {"level": LEVELS[lvl], "reasons": reasons}
