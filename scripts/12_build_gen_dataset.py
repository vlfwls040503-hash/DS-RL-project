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


def runs_wangsuk(geo_cols):
    """왕숙 지하도로 (솔로 주행, S1~S12 조건, 다속도역 34~95km/h).
    파일명 변형 다수(No./NO./NO3.13/NO7./S04re) → 'Road_' 뒤 마지막 숫자군=피험자,
    '_S' 뒤 숫자=조건. '본 실험데이터' 폴더만 사용(분석데이터 폴더는 중복 사본)."""
    DIR_W = r"<NAS_PATH set your own>"  # noqa: E501 — NAS 경로는 퍼블리셔 정규식이 한 줄 단위로 치환하므로 절대 줄바꿈 금지
    need = set(NEED_BASE) | set(geo_cols)
    files = sorted(glob.glob(os.path.join(DIR_W, "**", "*.csv"), recursive=True))
    files = [f for f in files if "분석데이터" not in f]
    pat_s = re.compile(r"_S0*(\d+)(re)?_", re.IGNORECASE)
    rid = 0
    for f in files:
        name = os.path.basename(f)
        ms = pat_s.search(name)
        head = name.split("_S")[0]
        digits = re.findall(r"(\d+)", head.split("Road_")[-1]) if "Road_" in head else []
        if not ms or not digits:
            continue                     # 태그 없는 연습주행 제외
        subj, cond = int(digits[-1]), int(ms.group(1)) - 1
        try:
            df = read_csv_fallback(f, usecols=_usecols(need))
        except Exception:
            continue
        yield df, subj, rid, cond
        rid += 1


def runs_icing(geo_cols):
    """결빙 2022 (감응형결빙주의표지판, 47~49km/h — 남산급 저속 질감 원산지).
    파일명에 실명 포함 → 이름은 id 매핑만, 어떤 산출물에도 미기록."""
    DIR_I = r"<NAS_PATH set your own>"  # noqa: E501 — NAS 경로 줄바꿈 금지
    need = set(NEED_BASE) | set(geo_cols)
    name_map = {}
    rid = 0
    for sc in sorted(os.listdir(DIR_I)):
        sp = os.path.join(DIR_I, sc)
        if not os.path.isdir(sp):
            continue
        cond = rid_c = int(re.sub(r"\D", "", sc) or 0) - 1
        for f in sorted(glob.glob(os.path.join(sp, "*.csv"))):
            m = re.search(r"_\d+_([가-힣]{2,4})_", os.path.basename(f))
            key = m.group(1) if m else os.path.basename(f)[:20]
            name_map.setdefault(key, len(name_map))
            try:
                df = read_csv_fallback(f, usecols=_usecols(need))
            except Exception:
                continue
            if "distanceAlongRoad" not in df.columns:
                continue
            if "time" not in df.columns:                 # distance-based log 호환
                df["time"] = np.arange(len(df), dtype=float)
            yield df, name_map[key], rid, max(cond, 0)
            rid += 1


def runs_underpass21(geo_cols):
    """지하유출입 2차년도(2021) 시뮬레이션 수행 로그 (저속 지하도로 — 잔차 원산지 2호)."""
    DIR_U = r"<NAS_PATH set your own>"  # noqa: E501 — NAS 경로 줄바꿈 금지
    need = set(NEED_BASE) | set(geo_cols)
    rid = 0
    for sc in sorted(os.listdir(DIR_U)):
        sp = os.path.join(DIR_U, sc)
        if not os.path.isdir(sp):
            continue
        cond = max(int(re.sub(r"\D", "", sc) or 1) - 1, 0)
        for f in sorted(glob.glob(os.path.join(sp, "*.csv"))):
            m = re.search(r"Road_\s*(\d+)", os.path.basename(f))
            subj = int(m.group(1)) if m else rid
            try:
                df = read_csv_fallback(f, usecols=_usecols(need))
            except Exception:
                continue
            if "distanceAlongRoad" not in df.columns:
                continue
            if "time" not in df.columns:
                df["time"] = np.arange(len(df), dtype=float)
            yield df, subj, rid, cond
            rid += 1


RESOLVERS = {"namsan": runs_namsan, "2024": runs_2024, "merge": runs_merge,
             "wangsuk": runs_wangsuk, "icing": runs_icing,
             "underpass21": runs_underpass21}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["namsan", "2024", "merge", "wangsuk", "icing", "underpass21", "smoke"], default="smoke")
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
