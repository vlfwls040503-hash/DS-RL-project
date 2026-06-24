# -*- coding: utf-8 -*-
"""
build_merge.py  --  ingest merge/diverge datasets (신월여의 본실험/추가 + 복층 지하도로DS).
Full-input model baseline: common core state features + experiment/지하 condition.
Subject = parsed person name (+ experiment prefix). Saves cache/dataset_merge.npz.

Usage:
  python build_merge.py --limit 3     # smoke test: 3 files per folder
  python build_merge.py               # full build
"""
import os, re, glob, json, argparse
import numpy as np
import pandas as pd
from common import CACHE, ACTIONS

FOLDERS = [
    ("shinwol", r"<NAS_PATH set your own>"),
    ("shinwol", r"<NAS_PATH set your own>"),
    ("bokcheung", r"<NAS_PATH set your own>"),
]
EXPS = ["shinwol", "bokcheung"]
DESIRED_STATE = [
    "speedInKmPerHour", "localAccelInMetresPerSecond2 X", "localAccelInMetresPerSecond2 Y",
    "bodyRotSpeedInRadsPerSecond Yaw", "laneCurvature", "offsetFromLaneCenter",
    "offsetFromRoadCenter", "distanceToLeftBorder", "distanceToRightBorder",
    "carriagewayWidth", "laneWidth", "roadLongitudinalSlope", "roadLateralSlope",
    "leftLaneOverLap", "rightLaneOverLap",
]


def parse_name(fn):
    base = fn.replace(".User_Master.csv", "").replace(".csv", "")
    is_jiha = 1 if "지하" in base else 0
    m = re.search(r"_([가-힣]{2,4})-", base)
    person = m.group(1) if m else None
    m2 = re.match(r"Log_\d+_([^_]+)", base)
    road = m2.group(1).strip() if m2 else "?"
    return person, is_jiha, road


def read_header(p):
    for enc in ("utf-8-sig", "cp949", "utf-8"):
        try:
            return list(pd.read_csv(p, nrows=0, encoding=enc).columns), enc
        except (UnicodeDecodeError, ValueError):
            continue
    return list(pd.read_csv(p, nrows=0, encoding="latin-1").columns), "latin-1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)  # 0 = all
    args = ap.parse_args()

    # common feature columns across all sampled headers
    sample_files = []
    for _, folder in FOLDERS:
        fs = glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
        if fs:
            sample_files.append(fs[0])
    common = set(DESIRED_STATE)
    for sf in sample_files:
        h, _ = read_header(sf)
        common &= set(h)
    common = [c for c in DESIRED_STATE if c in common]
    dropped = [c for c in DESIRED_STATE if c not in common]
    print("common state feats:", len(common), " dropped:", dropped)
    COND = [f"exp_{e}" for e in EXPS] + ["is_jiha"]
    FEATURES = common + COND
    need = set(["type"] + ACTIONS + common)

    Xs, Ys, runs, subs, jih, exps_arr, idx = [], [], [], [], [], [], []
    rid = 0
    subj_map = {}
    skipped = 0
    required = set(ACTIONS) | set(common)
    for exp, folder in FOLDERS:
        files = sorted(glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True))
        if args.limit:
            files = files[:args.limit]
        for f in files:
            fn = os.path.basename(f)
            person, is_jiha, road = parse_name(fn)
            sid = f"{exp}_{person}" if person else f"{exp}_file{rid}"
            try:
                hdr, enc = read_header(f)
            except Exception as e:
                skipped += 1; continue
            if not required <= set(hdr):     # not a standard telemetry log -> skip
                skipped += 1
                continue
            if sid not in subj_map:
                subj_map[sid] = len(subj_map)
            sidx = subj_map[sid]
            try:
                df = pd.read_csv(f, usecols=lambda c: c in need, encoding=enc, low_memory=False)
            except Exception as e:
                print("  FAIL", fn[:40], e); skipped += 1; continue
            if "type" in df.columns:
                df = df[df["type"] == "uv"]
            if len(df) == 0:
                continue
            Y = np.column_stack([pd.to_numeric(df[a], errors="coerce").astype("float32") for a in ACTIONS])
            raw = np.column_stack([pd.to_numeric(df[c], errors="coerce").astype("float64") for c in common])
            cond = np.zeros((len(df), len(COND)))
            cond[:, EXPS.index(exp)] = 1.0
            cond[:, -1] = is_jiha
            X = np.concatenate([raw, cond], axis=1).astype("float32")
            good = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
            X, Y = X[good], Y[good]
            if len(X) == 0:
                continue
            Xs.append(X); Ys.append(Y)
            runs.append(np.full(len(X), rid, "int32")); subs.append(np.full(len(X), sidx, "int16"))
            jih.append(np.full(len(X), is_jiha, "int8")); exps_arr.append(np.full(len(X), EXPS.index(exp), "int8"))
            idx.append(dict(run_id=rid, subject=sid, exp=exp, is_jiha=is_jiha, road=road, n=int(len(X)), file=fn))
            rid += 1
            if rid % 40 == 0:
                print(f"  {rid} runs, rows={sum(len(x) for x in Xs):,}")
            del df

    X = np.concatenate(Xs); Y = np.concatenate(Ys)
    run_id = np.concatenate(runs); subject = np.concatenate(subs)
    is_jiha_a = np.concatenate(jih); exp_a = np.concatenate(exps_arr)
    print(f"\nFINAL X{X.shape} Y{Y.shape} runs={rid} subjects={len(subj_map)} skipped_files={skipped}")
    np.savez_compressed(os.path.join(CACHE, "dataset_merge.npz"),
                        X=X, Y=Y, run_id=run_id, subject=subject, is_jiha=is_jiha_a, exp=exp_a,
                        feat_names=np.array(FEATURES), act_names=np.array(ACTIONS))
    json.dump(dict(subjects=list(subj_map.keys()), runs=idx, features=FEATURES),
              open(os.path.join(CACHE, "run_index_merge.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    for j, a in enumerate(ACTIONS):
        c = Y[:, j]; print(f"  {a:9s} min={c.min():.4f} max={c.max():.4f} mean={c.mean():.4f} std={c.std():.4f}")
    print(f"  subjects={len(subj_map)}  지하행 {int(is_jiha_a.sum()):,}/{len(is_jiha_a):,}")
    print("saved -> cache/dataset_merge.npz  feats:", FEATURES)


if __name__ == "__main__":
    main()
