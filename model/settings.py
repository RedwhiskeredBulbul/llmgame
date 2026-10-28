"""Parameters used by the active decision model and controller."""

import os

SIM_DT: float = float(os.getenv("SIM_DT", 0.0792))
DELTA_MAX = 0.6
FRONT_OVERHANG = 1.1
REAR_OVERHANG = 1.0
FRONT_AXLE = 1.15
REAR_AXLE = 1.85
WHEEL_BASE = FRONT_AXLE + REAR_AXLE
VEHICLE_LENGTH = FRONT_OVERHANG + WHEEL_BASE + REAR_OVERHANG
VEHICLE_WIDTH = 2.105
