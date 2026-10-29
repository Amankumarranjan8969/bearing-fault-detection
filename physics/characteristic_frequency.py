"""Bearing characteristic frequencies from shaft speed + geometry."""


def characteristic_frequencies(rpm, geom):
    fr = rpm / 60.0
    r = geom.ratio
    n = geom.n_balls
    d_over_D = geom.ball_d_mm / geom.pitch_d_mm
    return {
        "FR": fr,                                        # shaft rotation
        "FTF": 0.5 * fr * (1 - r),                       # cage
        "BPFO": 0.5 * n * fr * (1 - r),                  # outer race
        "BPFI": 0.5 * n * fr * (1 + r),                  # inner race
        "BSF": 0.5 * fr / d_over_D * (1 - r ** 2),       # rolling element (ball spin)
    }


FAULT_TO_FREQ = {"outer_race": "BPFO", "inner_race": "BPFI", "ball": "BSF"}
PHYS_KEYS = ["BPFO", "BPFI", "BSF", "FTF"]                # order of physics targets/head
