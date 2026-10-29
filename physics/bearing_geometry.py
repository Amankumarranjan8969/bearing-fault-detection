import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BearingGeometry:
    n_balls: int
    ball_d_mm: float
    pitch_d_mm: float
    contact_deg: float = 0.0

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    @property
    def ratio(self):
        """(d / D) * cos(theta)"""
        return self.ball_d_mm / self.pitch_d_mm * math.cos(math.radians(self.contact_deg))
