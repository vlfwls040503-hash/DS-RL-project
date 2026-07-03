# -*- coding: utf-8 -*-
"""C1 정찰: merge 데이터에서 선행차(fv) 궤적 추출 가능성 검증 (읽기전용)."""
import os, glob
import numpy as np
import pandas as pd
from build_merge import FOLDERS, read_header

f = None
for _, folder in FOLDERS:
    fs = glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
    if fs:
        f = fs[0]; break
print("sample:", os.path.basename(f))
cols, enc = read_header(f)
cand = ["type", "time", "scenarioTime", "distanceAlongRoad", "position X", "position Y",
        "speedInKmPerHour", "distanceToFrontVehicle", "TTCToFrontVehicle", "name"]
have = [c for c in cand if c in cols]
print("cols present:", have)
df = pd.read_csv(f, usecols=[c for c in have], encoding=enc, low_memory=False)
print("type counts:", df["type"].value_counts().to_dict())
tcol = "scenarioTime" if "scenarioTime" in df.columns else "time"
g = df.groupby(tcol)["type"].apply(lambda s: tuple(sorted(s)))
from collections import Counter
print("시점당 차량조합 상위:", Counter(g).most_common(4))
# uv 행에 distanceToFrontVehicle 있으면 그걸로 충분한지
uv = df[df["type"] == "uv"]
if "distanceToFrontVehicle" in uv.columns:
    d = pd.to_numeric(uv["distanceToFrontVehicle"], errors="coerce")
    print(f"uv.distanceToFrontVehicle: 유효 {d.notna().mean():.0%}, "
          f"중앙값 {d.median():.1f}, p10 {d.quantile(.1):.1f}, 999류 비율 {(d>500).mean():.0%}")
# fv 행의 위치/속도로 궤적 재구성 가능?
fv = df[df["type"] == "fv"]
if len(fv):
    print(f"fv rows: {len(fv)}, distanceAlongRoad 유효 "
          f"{pd.to_numeric(fv.get('distanceAlongRoad'), errors='coerce').notna().mean():.0%}, "
          f"speed 유효 {pd.to_numeric(fv.get('speedInKmPerHour'), errors='coerce').notna().mean():.0%}")
