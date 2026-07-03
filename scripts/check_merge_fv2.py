# -*- coding: utf-8 -*-
"""C 최종 진단: 시점별 uv 전방 최근접 차량은 어떤 type인가 (읽기전용)."""
import os, glob
import numpy as np
import pandas as pd
from build_merge import FOLDERS, read_header

files = []
for exp, folder in FOLDERS:
    if exp == "shinwol":
        files += glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)

for f in files[:3]:
    cols, enc = read_header(f)
    use = [c for c in ["type", "time", "position X", "position Y", "speedInKmPerHour"] if c in cols]
    df = pd.read_csv(f, usecols=use, encoding=enc, low_memory=False)
    for c in ["time", "position X", "position Y", "speedInKmPerHour"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    uv = df[df["type"] == "uv"].dropna().sort_values("time")
    # uv 진행방향 (위치 미분)
    uv = uv.assign(hx=np.gradient(uv["position X"]), hy=np.gradient(uv["position Y"]))
    hn = np.hypot(uv["hx"], uv["hy"]); uv = uv.assign(hx=uv["hx"]/hn.clip(1e-6), hy=uv["hy"]/hn.clip(1e-6))
    others = df[df["type"] != "uv"].dropna()
    m = others.merge(uv[["time", "position X", "position Y", "hx", "hy"]], on="time",
                     suffixes=("", "_u"))
    gap = ((m["position X"] - m["position X_u"]) * m["hx"]
           + (m["position Y"] - m["position Y_u"]) * m["hy"])
    lat = np.hypot(m["position X"] - m["position X_u"], m["position Y"] - m["position Y_u"])
    m = m.assign(gap=gap, lat=np.sqrt(np.maximum(lat**2 - gap**2, 0)))
    # 전방 200m 이내 + 횡방향 6m 이내(같은 주행로) 차량만
    ahead = m[(m["gap"] > 0) & (m["gap"] < 200) & (m["lat"] < 6)]
    stat = ahead.groupby("type")["gap"].agg(["count", "median"])
    tot_t = uv["time"].nunique()
    print(os.path.basename(f)[:40])
    print("  type별 전방(<200m,횡<6m) 등장:", {t: (int(r["count"]), round(float(r["median"]), 1))
                                              for t, r in stat.iterrows()},
          f"| uv 시점수 {tot_t}")
    # fv의 gap 분포(전체)
    fvg = m[m["type"] == "fv"]["gap"]
    if len(fvg):
        print(f"  fv gap 분포: p10 {fvg.quantile(.1):.0f} / p50 {fvg.median():.0f} / p90 {fvg.quantile(.9):.0f} m")
