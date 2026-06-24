# -*- coding: utf-8 -*-
"""
build_2024.py  --  ingest the 2024 long-distance underground dataset (PoC base).
32 subjects x {지상 ground, 지하 under}. NOTE schema asymmetry:
  지상=654-col export, 지하=179-col export. SDLP & front-vehicle exist ONLY in 지하.
Strategy (conservative): use the COMMON columns present in BOTH schemas as features
(auto-intersected). SDLP is recomputed from offsetFromLaneCenter at eval time, not used
as an input. Scenario one-hot -> 지상/지하 condition one-hot.

Output: cache/dataset_2024.npz  (X, Y, run_id, subject, cond[0=지상,1=지하], feat_names)
"""
import os, re, glob, json
import numpy as np
import pandas as pd
from common import CACHE, ACTIONS

D = r"<NAS_PATH set your own>"

# desired state features (front-veh & SDLP excluded: not in 지상 schema)
DESIRED_STATE = [
    "speedInKmPerHour",
    "localAccelInMetresPerSecond2 X", "localAccelInMetresPerSecond2 Y",
    "bodyRotSpeedInRadsPerSecond Yaw",
    "laneCurvature",
    "offsetFromLaneCenter", "offsetFromRoadCenter",
    "distanceToLeftBorder", "distanceToRightBorder",
    "carriagewayWidth", "laneWidth",
    "roadLongitudinalSlope", "roadLateralSlope",
    "leftLaneOverLap", "rightLaneOverLap",
]
COND_FEATS = ["cond_ground", "cond_under"]
FNPAT = re.compile(r"피실험자(\d+)_(지상|지하)주행")


def read_header(path):
    for enc in ("utf-8-sig", "cp949", "utf-8"):
        try:
            return list(pd.read_csv(path, nrows=0, encoding=enc).columns), enc
        except (UnicodeDecodeError, ValueError):
            continue
    return list(pd.read_csv(path, nrows=0, encoding="latin-1").columns), "latin-1"


def read_csv(path, usecols, enc):
    return pd.read_csv(path, usecols=lambda c: c in usecols, encoding=enc, low_memory=False)


def main():
    files = sorted(glob.glob(os.path.join(D, "피실험자*_*주행.csv")))
    print(f"files: {len(files)}")

    # determine common state columns across BOTH schemas (use one 지상 + one 지하)
    sang = next(f for f in files if "지상" in os.path.basename(f))
    ha = next(f for f in files if "지하" in os.path.basename(f))
    hs, _ = read_header(sang); hh, _ = read_header(ha)
    common = [c for c in DESIRED_STATE if c in hs and c in hh]
    dropped = [c for c in DESIRED_STATE if c not in common]
    print("common state features:", len(common))
    if dropped:
        print("  dropped (not in both schemas):", dropped)
    FEATURES = common + COND_FEATS
    need = set(["time", "scenarioTime", "type"] + ACTIONS + common)

    Xs, Ys, runs, subs, conds, idx = [], [], [], [], [], []
    rid = 0
    for i, f in enumerate(files):
        m = FNPAT.search(os.path.basename(f))
        if not m:
            print("  skip(name):", os.path.basename(f)); continue
        subj = int(m.group(1)); cond = 0 if m.group(2) == "지상" else 1
        _, enc = read_header(f)
        try:
            df = read_csv(f, need, enc)
        except Exception as e:
            print("  READ FAIL", os.path.basename(f), e); continue
        if "type" in df.columns:
            df = df[df["type"] == "uv"]
        if len(df) == 0:
            print("  no uv:", os.path.basename(f)); continue
        Y = np.column_stack([pd.to_numeric(df[a], errors="coerce").astype("float32") for a in ACTIONS])
        raw = np.column_stack([pd.to_numeric(df[c], errors="coerce").astype("float64") for c in common])
        oh = np.zeros((len(df), 2)); oh[:, cond] = 1.0
        X = np.concatenate([raw, oh], axis=1).astype("float32")
        good = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
        X, Y = X[good], Y[good]
        n = len(X)
        if n == 0:
            continue
        Xs.append(X); Ys.append(Y)
        runs.append(np.full(n, rid, "int32")); subs.append(np.full(n, subj, "int16"))
        conds.append(np.full(n, cond, "int8"))
        idx.append(dict(run_id=rid, subject=subj, cond=m.group(2), n=int(n), file=os.path.basename(f)))
        rid += 1
        if (i + 1) % 8 == 0 or i < 2:
            print(f"  [{i+1}/{len(files)}] s{subj:02d} {m.group(2)} uv={n} (cum {sum(len(x) for x in Xs):,})")
        del df

    X = np.concatenate(Xs); Y = np.concatenate(Ys)
    run_id = np.concatenate(runs); subject = np.concatenate(subs); cond = np.concatenate(conds)
    print(f"\nFINAL X{X.shape} Y{Y.shape} runs={rid} subjects={len(set(subject.tolist()))}")
    np.savez_compressed(os.path.join(CACHE, "dataset_2024.npz"),
                        X=X, Y=Y, run_id=run_id, subject=subject, cond=cond,
                        feat_names=np.array(FEATURES), act_names=np.array(ACTIONS))
    json.dump(idx, open(os.path.join(CACHE, "run_index_2024.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    for j, a in enumerate(ACTIONS):
        c = Y[:, j]; print(f"  {a:9s} min={c.min():.4f} max={c.max():.4f} mean={c.mean():.4f} std={c.std():.4f}")
    sp = X[:, 0]
    print(f"  speed mean 지상={sp[cond==0].mean():.1f} 지하={sp[cond==1].mean():.1f}")
    # offset variability (proxy for SDLP) ground vs under
    oi = FEATURES.index("offsetFromLaneCenter")
    print(f"  offset std 지상={X[cond==0,oi].std():.4f} 지하={X[cond==1,oi].std():.4f}")
    print("saved -> cache/dataset_2024.npz  feat_names:", FEATURES)


if __name__ == "__main__":
    main()
