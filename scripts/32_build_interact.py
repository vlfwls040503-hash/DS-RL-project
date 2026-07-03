# -*- coding: utf-8 -*-
"""
32_build_interact.py  --  C1: 신월여의 합류부에서 선행차(fv) 재생 트랙 추출.

각 주행에서 uv(자차)로 도로 프로파일(기존 reindex_run)과, fv(선행차)의 시간-위치-속도
트랙을 추출. 좌표는 uv 시작점 기준으로 재영점(자차 s=0과 정렬). env에서 시간보간으로
차간거리/상대속도를 재생한다.

Output: cache/env_roads_interact.npz
  (기존 도로 채널 + leader_t/leader_s/leader_v ragged 배열)
"""
import os, glob, re
import numpy as np
import pandas as pd

from common import CACHE, GEN_DD, GEN_GEO_CANDIDATES, reindex_run, read_csv_fallback
from build_merge import FOLDERS, parse_name

MIN_GRID_PTS = 200
NEED = ["type", "time", "speedInKmPerHour", "offsetFromLaneCenter", "distanceAlongRoad"] \
       + list(GEN_GEO_CANDIDATES)
LEAD_DT = 0.5          # 선행차 트랙 시간 그리드 (s)


def main():
    geo_cols = list(GEN_GEO_CANDIDATES)
    roads, subj_map, rid, skipped = [], {}, 0, 0
    shinwol = [f for exp, f in FOLDERS if exp == "shinwol"]
    files = []
    for folder in shinwol:
        files += sorted(glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True))
    print(f"shinwol files: {len(files)}", flush=True)

    for f in files:
        person, is_jiha, _ = parse_name(os.path.basename(f))
        sid = person or f"f{rid}"
        subj_map.setdefault(sid, len(subj_map))
        try:
            df = read_csv_fallback(f, usecols=lambda c: c in set(NEED))
        except Exception:
            skipped += 1; continue
        if "type" not in df.columns:
            skipped += 1; continue
        uv = df[df["type"] == "uv"]
        fv = df[df["type"] == "fv"]
        if len(uv) < 500 or len(fv) < 500:
            skipped += 1; continue
        r = reindex_run(uv, geo_cols, GEN_DD)
        if r is None:
            skipped += 1; continue
        beh, G = r
        if len(beh) < MIN_GRID_PTS:
            skipped += 1; continue

        # 선행차 트랙: uv 시작(시간·위치) 기준 재영점 → 균일 0.5s 그리드
        t0 = float(uv["time"].iloc[0])
        s0 = float(pd.to_numeric(uv["distanceAlongRoad"], errors="coerce").iloc[0])
        ft = pd.to_numeric(fv["time"], errors="coerce").to_numpy(float) - t0
        fs = pd.to_numeric(fv["distanceAlongRoad"], errors="coerce").to_numpy(float) - s0
        fvv = pd.to_numeric(fv["speedInKmPerHour"], errors="coerce").to_numpy(float) / 3.6
        ok = np.isfinite(ft) & np.isfinite(fs) & np.isfinite(fvv) & (ft >= 0)
        ft, fs, fvv = ft[ok], fs[ok], fvv[ok]
        if len(ft) < 100:
            skipped += 1; continue
        tg = np.arange(0.0, ft[-1], LEAD_DT)
        roads.append(dict(curv=G[:, 0], slope=G[:, 1], lane_w=G[:, 3], cw=G[:, 4],
                          e_ref=beh[:, 0], v_ref=beh[:, 1],
                          subject=subj_map[sid], cond=int(is_jiha),
                          leader_t=tg.astype("float32"),
                          leader_s=np.interp(tg, ft, fs).astype("float32"),
                          leader_v=np.interp(tg, ft, fvv).astype("float32")))
        rid += 1
        if len(roads) % 20 == 0:
            print(f"  {len(roads)} roads...", flush=True)

    # ragged 저장 (도로채널 + 리더채널 별도 ptr)
    CH = ["curv", "slope", "lane_w", "cw", "e_ref", "v_ref"]
    LCH = ["leader_t", "leader_s", "leader_v"]
    ptr = np.cumsum([0] + [len(r["curv"]) for r in roads]).astype("int64")
    lptr = np.cumsum([0] + [len(r["leader_t"]) for r in roads]).astype("int64")
    data = {c: np.concatenate([np.asarray(r[c], np.float32) for r in roads]) for c in CH}
    ldata = {c: np.concatenate([np.asarray(r[c], np.float32) for r in roads]) for c in LCH}
    out = os.path.join(CACHE, "env_roads_interact.npz")
    np.savez_compressed(out, ptr=ptr, lptr=lptr, dd=GEN_DD,
                        subject=np.array([r["subject"] for r in roads], "int64"),
                        cond=np.array([r["cond"] for r in roads], "int8"),
                        **data, **ldata)
    # 요약 통계 (차간거리 분포 sanity)
    gaps = []
    for r in roads[:50]:
        n = len(r["v_ref"])
        tt = np.cumsum(np.full(n, GEN_DD) / np.maximum(r["v_ref"], 1.0))
        sl = np.interp(tt, r["leader_t"], r["leader_s"])
        gaps.append(np.median(sl - np.arange(n) * GEN_DD))
    print(f"roads={len(roads)} (skipped {skipped}) subjects={len(subj_map)} "
          f"| 표본 중앙 차간거리 median {np.median(gaps):.0f} m", flush=True)
    print("saved ->", out, flush=True)


def load_interact(path=None):
    d = np.load(path or os.path.join(CACHE, "env_roads_interact.npz"))
    ptr, lptr = d["ptr"], d["lptr"]
    roads = []
    for k in range(len(ptr) - 1):
        a, b = int(ptr[k]), int(ptr[k + 1])
        la, lb = int(lptr[k]), int(lptr[k + 1])
        r = {c: d[c][a:b] for c in ["curv", "slope", "lane_w", "cw", "e_ref", "v_ref"]}
        for c in ["leader_t", "leader_s", "leader_v"]:
            r[c] = d[c][la:lb]
        r["subject"] = int(d["subject"][k]); r["cond"] = int(d["cond"][k])
        roads.append(r)
    return roads, float(d["dd"])


if __name__ == "__main__":
    main()
