# -*- coding: utf-8 -*-
"""
02_build_dataset.py
Extract uv (user-vehicle) rows from every main-run CSV, select/clean features
and actions, and save a single compact cache (float32) preserving per-run
temporal order for sequence models.

Output: cache/dataset.npz
  X        (N, F) float32   raw (unscaled) features
  Y        (N, 3) float32   actions [steering, throttle, brake]
  run_id   (N,)   int32     contiguous id per file (for sequence windowing)
  subject  (N,)   int16
  scenario (N,)   int8      1..4
  feat_names, act_names     saved as arrays of str
"""
import os, glob, json
import numpy as np
import pandas as pd
from common import (RAW_DIR, CACHE, RUN_PAT, ACTIONS, RAW_STATE, FRONT_RAW,
                    FEATURES, DIST_CAP, TTC_CAP)

USECOLS = ["time", "type"] + ACTIONS + RAW_STATE + FRONT_RAW


def clean_front(df):
    """Return engineered front-vehicle features as a DataFrame aligned to df."""
    dist = pd.to_numeric(df["distanceToFrontVehicle"], errors="coerce").astype("float64")
    ttc = pd.to_numeric(df["TTCToFrontVehicle"], errors="coerce").astype("float64")

    has_front = np.isfinite(dist.to_numpy()) & (dist.to_numpy() > 0)
    dist_capped = np.where(has_front, np.clip(dist.to_numpy(), 0, DIST_CAP), DIST_CAP)

    ttc_arr = ttc.to_numpy()
    # valid approaching TTC: finite & positive
    valid_ttc = np.isfinite(ttc_arr) & (ttc_arr > 0)
    ttc_capped = np.where(valid_ttc, np.clip(ttc_arr, 0, TTC_CAP), TTC_CAP)
    inv_ttc = np.where(valid_ttc, 1.0 / np.clip(ttc_arr, 0.1, None), 0.0)  # 0 when safe/none

    return pd.DataFrame({
        "frontDist_capped": dist_capped,
        "has_front": has_front.astype("float64"),
        "invTTC": inv_ttc,
        "TTC_capped": ttc_capped,
    }, index=df.index)


def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    runs = []
    for f in files:
        m = RUN_PAT.search(os.path.basename(f))
        if m:
            runs.append((f, int(m.group(1)), int(m.group(2))))
    print(f"main-run files: {len(runs)}")

    X_parts, Y_parts, run_parts, subj_parts, scen_parts = [], [], [], [], []
    run_index = []  # (run_id, file, subject, scenario, n)
    rid = 0
    for i, (f, subj, scen) in enumerate(runs):
        df = pd.read_csv(f, usecols=lambda c: c in USECOLS, low_memory=False)
        uv = df[df["type"] == "uv"].copy()
        if len(uv) == 0:
            continue
        # actions
        Y = np.column_stack([pd.to_numeric(uv[a], errors="coerce").astype("float32") for a in ACTIONS])
        # raw state
        raw = {c: pd.to_numeric(uv[c], errors="coerce").astype("float64") for c in RAW_STATE}
        rawdf = pd.DataFrame(raw, index=uv.index)
        # engineered front feats
        eng = clean_front(uv)
        # scenario one-hot
        oh = np.zeros((len(uv), 4), dtype="float64")
        oh[:, scen - 1] = 1.0
        ohdf = pd.DataFrame(oh, columns=["scen_S1", "scen_S2", "scen_S3", "scen_S4"], index=uv.index)

        feat = pd.concat([rawdf, eng, ohdf], axis=1)[FEATURES]
        Xv = feat.to_numpy(dtype="float32")

        # drop rows with any remaining NaN/Inf in X or Y
        good = np.isfinite(Xv).all(axis=1) & np.isfinite(Y).all(axis=1)
        Xv, Yv = Xv[good], Y[good]
        n = len(Xv)
        if n == 0:
            continue
        X_parts.append(Xv); Y_parts.append(Yv)
        run_parts.append(np.full(n, rid, dtype="int32"))
        subj_parts.append(np.full(n, subj, dtype="int16"))
        scen_parts.append(np.full(n, scen, dtype="int8"))
        run_index.append(dict(run_id=rid, file=os.path.basename(f), subject=subj, scenario=scen, n=int(n)))
        rid += 1
        if (i + 1) % 20 == 0:
            print(f"  processed {i+1}/{len(runs)}  (rows so far: {sum(len(p) for p in X_parts):,})")
        del df, uv

    X = np.concatenate(X_parts); Y = np.concatenate(Y_parts)
    run_id = np.concatenate(run_parts); subject = np.concatenate(subj_parts)
    scenario = np.concatenate(scen_parts)
    print(f"FINAL: X{X.shape}  Y{Y.shape}  runs={rid}")

    out = os.path.join(CACHE, "dataset.npz")
    np.savez_compressed(out, X=X, Y=Y, run_id=run_id, subject=subject, scenario=scenario,
                        feat_names=np.array(FEATURES), act_names=np.array(ACTIONS))
    with open(os.path.join(CACHE, "run_index.json"), "w", encoding="utf-8") as fh:
        json.dump(run_index, fh, ensure_ascii=False, indent=2)
    print("saved ->", out)
    # quick action stats
    for j, a in enumerate(ACTIONS):
        col = Y[:, j]
        print(f"  {a:9s} min={col.min():.4f} max={col.max():.4f} mean={col.mean():.4f} std={col.std():.4f}")


if __name__ == "__main__":
    main()
