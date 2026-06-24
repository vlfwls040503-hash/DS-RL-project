# -*- coding: utf-8 -*-
"""Shared config for the behavioral-cloning pipeline."""
import os, re

RAW_DIR = r"<DATA_DIR set your own>"
PROJ = r"D:\driving_bc"
CACHE = os.path.join(PROJ, "cache")
ART = os.path.join(PROJ, "artifacts")
REP = os.path.join(PROJ, "reports")
for d in (CACHE, ART, REP):
    os.makedirs(d, exist_ok=True)

RUN_PAT = re.compile(r"남산터널_피실험자(\d+)_S([1-4])\.csv$")

# --- Action columns (human control input) ---
ACTIONS = ["steering", "throttle", "brake"]

# --- State feature columns taken directly from the CSV ---
RAW_STATE = [
    "speedInKmPerHour",
    "localAccelInMetresPerSecond2 X",   # longitudinal accel
    "localAccelInMetresPerSecond2 Y",   # lateral accel
    "bodyRotSpeedInRadsPerSecond Yaw",  # yaw rate
    "laneCurvature",
    "offsetFromLaneCenter",
    "offsetFromRoadCenter",
    "standardDeviationFromLaneCenter",
    "distanceToLeftBorder",
    "distanceToRightBorder",
    "carriagewayWidth",
    "laneWidth",
    "roadLongitudinalSlope",
    "roadLateralSlope",
    "leftLaneOverLap",
    "rightLaneOverLap",
]
# Front-vehicle columns get special cleaning -> produce engineered features below
FRONT_RAW = ["distanceToFrontVehicle", "TTCToFrontVehicle"]

# Engineered feature names appended after RAW_STATE (order matters!)
ENG_FEATS = ["frontDist_capped", "has_front", "invTTC", "TTC_capped"]
# Scenario one-hot appended last
SCEN_FEATS = ["scen_S1", "scen_S2", "scen_S3", "scen_S4"]

FEATURES = RAW_STATE + ENG_FEATS + SCEN_FEATS

# Caps for front-vehicle engineering
DIST_CAP = 250.0     # m  (no front vehicle -> this)
TTC_CAP = 100.0      # s  (INF / non-approaching -> this)

# Subject-level split (fixed). 29 subjects total.
TEST_SUBJECTS = [5, 12, 19, 26]
VAL_SUBJECTS = [3, 16, 23, 29]
# everything else -> train
