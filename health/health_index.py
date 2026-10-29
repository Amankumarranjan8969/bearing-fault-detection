"""Health index: Mahalanobis distance from a healthy baseline, mapped to 0-100 (100 = healthy)."""
import numpy as np


class HealthIndex:
    def __init__(self, features, shrinkage=1e-2):
        self.features, self.shrinkage = list(features), shrinkage

    def fit(self, healthy, failure=None):
        """healthy: baseline rows (n, f). failure: optional rows that should map to health ~0.
        Calibrate on run-to-failure data when you have it (NASA IMS)."""
        h = np.asarray(healthy, float)
        self.mu, self.sd = h.mean(0), h.std(0) + 1e-9
        z = (h - self.mu) / self.sd
        self.icov = np.linalg.inv(np.atleast_2d(np.cov(z.T)) + self.shrinkage * np.eye(z.shape[1]))
        self.d_ref = float(np.percentile(self.distance(h), 95))
        if failure is not None and len(failure):
            self.d_fail = max(float(np.median(self.distance(np.asarray(failure, float)))), 1.5 * self.d_ref)
        else:
            self.d_fail = 10 * self.d_ref
        return self

    def distance(self, X):
        z = (np.atleast_2d(np.asarray(X, float)) - self.mu) / self.sd
        return np.sqrt(np.einsum("ij,jk,ik->i", z, self.icov, z))

    def score(self, X):
        d = self.distance(X)
        return 100.0 * (1.0 - np.clip((d - self.d_ref) / (self.d_fail - self.d_ref), 0.0, 1.0))


def smooth(series, alpha=0.3):
    out, s = [], None
    for v in series:
        s = v if s is None else alpha * v + (1 - alpha) * s
        out.append(s)
    return np.asarray(out)
