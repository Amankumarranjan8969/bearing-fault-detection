"""Per-bearing Digital Twin state: Kalman-filtered health + fault state + trend + alert."""
import time
from collections import deque
from health.degradation import fit_trend, trend_label, time_to_threshold, alert_level


class ScalarKalman:
    def __init__(self, x0=100.0, p0=25.0, q=1.0, r=25.0):
        self.x, self.p, self.q, self.r = x0, p0, q, r
        self.started = False

    def update(self, z):
        if not self.started:
            self.x, self.started = float(z), True
            return self.x
        p = self.p + self.q
        k = p / (p + self.r)
        self.x += k * (z - self.x)
        self.p = (1 - k) * p
        return self.x


class BearingTwin:
    def __init__(self, bearing_id, cfg):
        self.id, self.cfg = bearing_id, cfg
        self.kf = ScalarKalman()
        self.hist = deque(maxlen=500)
        self.state = None

    def update(self, probs, health_raw, sensors, supports):
        fault = max(probs, key=probs.get)
        conf = float(probs[fault])
        health = float(self.kf.update(health_raw))
        self.hist.append({"t": time.time(), "health": health, "health_raw": float(health_raw),
                          "fault": fault, "confidence": conf})
        slope = fit_trend([h["health"] for h in list(self.hist)[-10:]])
        ch = self.cfg["health"]
        alert = alert_level(health, 1.0 - probs.get("healthy", 0.0), slope, sensors.get("temperature"), supports, ch)
        self.state = {"bearing_id": self.id, "fault": fault, "confidence": conf, "health": health,
                      "alert": alert, "n_updates": len(self.hist),
                      "trend": {"slope_per_update": slope, "label": trend_label(slope),
                                "indicative_steps_to_threshold": time_to_threshold(health, slope, ch.get("end_threshold", 20.0)),
                                "note": "indicative only, not a validated RUL"},
                      "physics_supports_prediction": supports, "sensors": sensors}
        return self.state


class DigitalTwin:
    def __init__(self, cfg):
        self.cfg, self.twins = cfg, {}

    def update(self, bearing_id, probs, health_raw, sensors, supports):
        t = self.twins.setdefault(bearing_id, BearingTwin(bearing_id, self.cfg))
        return t.update(probs, health_raw, sensors, supports)

    def state(self, bearing_id):
        t = self.twins.get(bearing_id)
        return t.state if t else None

    def history(self, bearing_id, limit=100):
        t = self.twins.get(bearing_id)
        return list(t.hist)[-limit:] if t else []
