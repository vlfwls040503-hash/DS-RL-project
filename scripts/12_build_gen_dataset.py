# -*- coding: utf-8 -*-
"""
12_build_gen_dataset.py  --  distance-reindexed trajectory windows for the CVAE.

  python 12_build_gen_dataset.py --smoke                 # synthetic self-test
  python 12_build_gen_dataset.py --exp 2024              # real (needs data paths)
  python 12_build_gen_dataset.py --exp namsan --feat_set laneCurvature,laneWidth

Output: cache/dataset_gen_{exp}.npz
  X_geo (N,W,G) geometry, Y_beh (N,W,2) [offset, speed_mps],
  win_subject, win_run, win_cond, feat_geo
"""
import os, re, glob, argparse
import numpy as np
from common import (CACHE, RAW_DIR, RUN_PAT, GEN_GEO_CANDIDATES, GEN_DD, GEN_W,
                    read_csv_fallback, build_gen_dataset, build_smoke_dataset)

NEED_BASE = ["type", "time", "speedInKmPerHour", "offsetFromLaneCenter", "distanceAlongRoad"]


def _usecols(need):
    return lambda c: c in need


def runs_namsan(geo_cols):
    need = set(NEED_BASE) | set(geo_cols)
    files = sorted(glob.glob(os.path.join(RAW_DIR, "남산터널_피실험자*_S*.csv")))
    for rid, f in enumerate(files):
        m = RUN_PAT.search(os.path.basename(f))
        if not m:
            continue
        df = read_csv_fallback(f, usecols=_usecols(need))
        yield df, int(m.group(1)), rid, int(m.group(2)) - 1


def runs_2024(geo_cols):
    from build_2024 import D as DIR_2024
    need = set(NEED_BASE) | set(geo_cols)
    pat = re.compile(r"피실험자(\d+)_(지상|지하)주행")
    files = sorted(glob.glob(os.path.join(DIR_2024, "피실험자*_*주행.csv")))
    for rid, f in enumerate(files):
        m = pat.search(os.path.basename(f))
        if not m:
            continue
        df = read_csv_fallback(f, usecols=_usecols(need))
        yield df, int(m.group(1)), rid, 0 if m.group(2) == "지상" else 1


def runs_merge(geo_cols):
    from build_merge import FOLDERS, parse_name
    need = set(NEED_BASE) | set(geo_cols)
    subj_map = {}
    rid = 0
    for exp, folder in FOLDERS:
        for f in sorted(glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)):
            person, is_jiha, road = parse_name(os.path.basename(f))
            sid = f"{exp}_{person}" if person else f"{exp}_f{rid}"
            subj_map.setdefault(sid, len(subj_map))
            try:
                df = read_csv_fallback(f, usecols=_usecols(need))
            except Exception:
                continue
            if "steering" not in df.columns and "offsetFromLaneCenter" not in df.columns:
                continue
            yield df, subj_map[sid], rid, int(is_jiha)
            rid += 1


RESOLVERS = {"namsan": runs_namsan, "2024": runs_2024, "merge": runs_merge}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["namsan", "2024", "merge", "smoke"], default="smoke")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--feat_set", default="", help="comma-separated geo features (cross-exp intersection)")
    ap.add_argument("--use_condition", action="store_true")
    ap.add_argument("--dd", type=float, default=GEN_DD)
    ap.add_argument("--window", type=int, default=GEN_W)
    ap.add_argument("--stride", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    exp = "smoke" if args.smoke else args.exp
    stride = args.stride if args.stride > 0 else args.window // 2
    geo_cols = [c.strip() for c in args.feat_set.split(",") if c.strip()] or list(GEN_GEO_CANDIDATES)
    out = args.out or os.path.join(CACHE, f"dataset_gen_{exp}.npz")

    if exp == "smoke":
        build_smoke_dataset(out, geo_cols=geo_cols, dd=args.dd, W=args.window, stride=stride)
        d = np.load(out, allow_pickle=True)
        print(f"[smoke] X_geo{d['X_geo'].shape} Y_beh{d['Y_beh'].shape} "
              f"subjects={len(set(d['win_subject'].tolist()))} feat_geo={list(d['feat_geo'])}")
        print("saved ->", out)
        return

    print(f"building exp={exp} geo={geo_cols} dd={args.dd} W={args.window} stride={stride}")
    Xg, Yb, ws, wr, wc = build_gen_dataset(RESOLVERS[exp](geo_cols), geo_cols,
                                           dd=args.dd, W=args.window, stride=stride)
    feat_geo = list(geo_cols)
    if args.use_condition:
        ncond = int(wc.max()) + 1
        oh = np.eye(ncond, dtype="float32")[wc]                    # (N, ncond)
        oh = np.repeat(oh[:, None, :], Xg.shape[1], axis=1)        # (N, W, ncond)
        Xg = np.concatenate([Xg, oh], axis=2)
        feat_geo += [f"cond_{i}" for i in range(ncond)]
        print(f"  + condition one-hot ({ncond}) -> geo dim {Xg.shape[2]}")

    np.savez_compressed(out, X_geo=Xg, Y_beh=Yb, win_subject=ws, win_run=wr, win_cond=wc,
                        feat_geo=np.array(feat_geo), dd=args.dd)
    print(f"X_geo{Xg.shape} Y_beh{Yb.shape} windows={len(Xg)} "
          f"subjects={len(set(ws.tolist()))} runs={len(set(wr.tolist()))}")
    print("saved ->", out)


if __name__ == "__main__":
    main()
