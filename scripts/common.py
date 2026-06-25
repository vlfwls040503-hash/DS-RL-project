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


# =========================================================================
# Generative model (CVAE) — distance-reindexed trajectory generation
# =========================================================================
import numpy as np
import pandas as pd

# behavior channels (generation target) — functions of distance along road
GEN_BEHAVIOR = ["offsetFromLaneCenter", "speed_mps"]   # speed_mps derived from speedInKmPerHour
# geometry condition candidates (transferable across experiments only)
GEN_GEO_CANDIDATES = ["laneCurvature", "roadLongitudinalSlope", "roadLateralSlope",
                      "laneWidth", "carriagewayWidth"]
GEN_DD = 2.0          # distance grid spacing (m)
GEN_W = 256           # window length in grid points (~512 m at 2 m)
GEN_TEST_FRAC = 1.0 / 7
GEN_VAL_FRAC = 1.0 / 7
DEFAULT_DT = 0.0229   # median time step fallback (s)


def read_csv_fallback(path, usecols=None):
    """pandas.read_csv with encoding fallback (no errors= arg)."""
    for enc in ("utf-8-sig", "cp949", "utf-8", "latin-1"):
        try:
            return pd.read_csv(path, usecols=usecols, encoding=enc, low_memory=False)
        except (UnicodeDecodeError, ValueError):
            continue
    return pd.read_csv(path, usecols=usecols, encoding="latin-1", low_memory=False)


def reindex_run(df, geo_cols, dd=GEN_DD):
    """Resample one uv run onto a uniform distance grid.
    Returns (beh[M,2], geo[M,G]) float32, or None if too short. geo_cols missing -> 0 column."""
    if "speedInKmPerHour" not in df.columns or "offsetFromLaneCenter" not in df.columns:
        return None
    spd = (pd.to_numeric(df.get("speedInKmPerHour"), errors="coerce") / 3.6).to_numpy(dtype="float64")
    off = pd.to_numeric(df.get("offsetFromLaneCenter"), errors="coerce").to_numpy(dtype="float64")
    geo = {c: pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float64")
           for c in geo_cols if c in df.columns}

    # cumulative distance d
    d = None
    if "distanceAlongRoad" in df.columns:
        da = pd.to_numeric(df["distanceAlongRoad"], errors="coerce").to_numpy(dtype="float64")
        if np.isfinite(da).all() and (np.diff(da) > -1e-6).all() and (da[-1] - da[0]) > dd * 4:
            d = da
    if d is None:  # fallback: integrate speed (works regardless of schema)
        if "time" in df.columns:
            t = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype="float64")
            dts = np.diff(t); dts = dts[(dts > 0) & (dts < 1)]
            dtm = float(np.median(dts)) if len(dts) else DEFAULT_DT
        else:
            dtm = DEFAULT_DT
        sp = np.where(np.isfinite(spd), spd, 0.0)
        d = np.cumsum(sp * dtm)

    base = np.isfinite(d) & np.isfinite(spd) & np.isfinite(off)
    for c in geo:
        base &= np.isfinite(geo[c])
    d, spd, off = d[base], spd[base], off[base]
    geo = {c: v[base] for c, v in geo.items()}
    if len(d) < 5:
        return None
    order = np.argsort(d)
    d, spd, off = d[order], spd[order], off[order]
    geo = {c: v[order] for c, v in geo.items()}
    keep = np.concatenate([[True], np.diff(d) > 1e-6])   # strictly increasing for interp
    d, spd, off = d[keep], spd[keep], off[keep]
    geo = {c: v[keep] for c, v in geo.items()}
    if len(d) < 5 or (d[-1] - d[0]) < dd * 4:
        return None

    grid = np.arange(d[0], d[-1], dd)
    beh = np.column_stack([np.interp(grid, d, off), np.interp(grid, d, spd)])
    G = np.column_stack([np.interp(grid, d, geo[c]) if c in geo else np.zeros(len(grid))
                         for c in geo_cols])
    return beh.astype("float32"), G.astype("float32")


def build_gen_dataset(run_iter, geo_cols, dd=GEN_DD, W=GEN_W, stride=None):
    """run_iter yields (df, subject:int, run:int, cond:int). Returns window arrays."""
    if stride is None:
        stride = W // 2
    Xg, Yb, wsub, wrun, wcond = [], [], [], [], []
    for df, subj, run, cond in run_iter:
        uv = df[df["type"] == "uv"] if "type" in df.columns else df
        r = reindex_run(uv, geo_cols, dd)
        if r is None:
            continue
        beh, G = r
        for s in range(0, len(beh) - W + 1, stride):
            Yb.append(beh[s:s + W]); Xg.append(G[s:s + W])
            wsub.append(subj); wrun.append(run); wcond.append(cond)
    if not Xg:
        raise RuntimeError("no windows built (runs too short for window W?)")
    return (np.stack(Xg).astype("float32"), np.stack(Yb).astype("float32"),
            np.array(wsub, "int64"), np.array(wrun, "int64"), np.array(wcond, "int8"))


def gen_split(win_subject, seed=0, test_frac=GEN_TEST_FRAC, val_frac=GEN_VAL_FRAC):
    """Subject-level split. Returns (train_mask, val_mask, test_mask)."""
    u = np.array(sorted(set(win_subject.tolist())))
    rng = np.random.RandomState(seed); rng.shuffle(u)
    n = len(u); nt = max(1, int(round(n * test_frac))); nv = max(1, int(round(n * val_frac)))
    test_s, val_s = set(u[:nt].tolist()), set(u[nt:nt + nv].tolist())
    te = np.isin(win_subject, list(test_s)); va = np.isin(win_subject, list(val_s))
    return ~(te | va), va, te


# ---- 1D distribution distances (numpy; avoids scipy dependency) ----
def wasserstein1d(a, b):
    a = np.sort(np.asarray(a, dtype="float64")); b = np.sort(np.asarray(b, dtype="float64"))
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    allv = np.sort(np.concatenate([a, b]))
    deltas = np.diff(allv)
    ca = np.searchsorted(a, allv[:-1], side="right") / len(a)
    cb = np.searchsorted(b, allv[:-1], side="right") / len(b)
    return float(np.sum(np.abs(ca - cb) * deltas))


def ks_stat(a, b):
    a = np.sort(np.asarray(a, dtype="float64")); b = np.sort(np.asarray(b, dtype="float64"))
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    allv = np.sort(np.concatenate([a, b]))
    ca = np.searchsorted(a, allv, side="right") / len(a)
    cb = np.searchsorted(b, allv, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


# ---- smoke synthetic data (schema-faithful) for self-test without real data ----
def make_smoke_runs(n_subj=10, n_runs=2, seed=0, npts=2400):
    """Yield (df, subject, run, cond) mimicking the CSV schema, with subject-specific
    SDLP and condition-specific speed so split/z-space/condition code is exercised."""
    rng = np.random.RandomState(seed)
    rid = 0
    for s in range(n_subj):
        sub_sdlp = 0.08 + 0.30 * rng.rand()        # subject-specific weave amplitude
        sub_bias = 0.15 * rng.randn()              # subject lane-position bias
        for c in range(n_runs):                    # run index acts as condition
            t = np.arange(npts) * DEFAULT_DT
            base_speed = 22.0 + 4.0 * c + 1.5 * rng.randn()   # condition affects speed
            speed = base_speed + 1.0 * np.sin(np.linspace(0, 8, npts)) + 0.3 * rng.randn(npts)
            speed = np.clip(speed, 1.0, None)
            dist = np.cumsum(speed * DEFAULT_DT)
            curv = 0.0020 * np.sin(np.linspace(0, 6, npts)) + 0.0004 * rng.randn(npts)
            offset = (sub_bias + sub_sdlp * np.sin(np.linspace(0, 22, npts) + rng.rand() * 6)
                      + 80.0 * curv + 0.02 * rng.randn(npts))
            df = pd.DataFrame({
                "time": t, "type": "uv",
                "speedInKmPerHour": speed * 3.6,
                "offsetFromLaneCenter": offset,
                "distanceAlongRoad": dist,
                "laneCurvature": curv,
                "roadLongitudinalSlope": 0.01 * np.sin(np.linspace(0, 3, npts)),
                "roadLateralSlope": 0.001 * rng.randn(npts),
                "laneWidth": 3.0 + 0.05 * rng.randn(npts),
                "carriagewayWidth": 6.5 + 0.1 * rng.randn(npts),
            })
            yield df, s, rid, c
            rid += 1


def build_smoke_dataset(out_path, geo_cols=None, dd=GEN_DD, W=GEN_W, stride=None, seed=0, n_subj=10):
    geo_cols = list(geo_cols) if geo_cols else list(GEN_GEO_CANDIDATES)
    Xg, Yb, ws, wr, wc = build_gen_dataset(
        make_smoke_runs(n_subj=n_subj, seed=seed), geo_cols, dd=dd, W=W, stride=stride)
    np.savez_compressed(out_path, X_geo=Xg, Y_beh=Yb, win_subject=ws, win_run=wr, win_cond=wc,
                        feat_geo=np.array(geo_cols), dd=dd)
    return out_path
