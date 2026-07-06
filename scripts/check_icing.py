# -*- coding: utf-8 -*-
"""결빙 2022 스키마·속도역·파일명 정찰 (읽기전용)."""
import os, glob
import pandas as pd
import numpy as np

BASE = r"<NAS_PATH set your own>"
L = []
for sc in sorted(os.listdir(BASE)):
    sp = os.path.join(BASE, sc)
    if not os.path.isdir(sp):
        continue
    fs = sorted(glob.glob(os.path.join(sp, "*.csv")))
    L.append(f"{sc}: {len(fs)} files | 예: {os.path.basename(fs[0]) if fs else '-'}")
    if fs:
        for enc in ("utf-8-sig", "cp949", "utf-8"):
            try:
                cols = list(pd.read_csv(fs[0], nrows=0, encoding=enc).columns); break
            except Exception:
                continue
        need = ["distanceAlongRoad", "offsetFromLaneCenter", "speedInKmPerHour",
                "laneCurvature", "roadLongitudinalSlope", "laneWidth",
                "carriagewayWidth", "time", "type"]
        L.append("  cols: " + " ".join(("O" if c in cols else "X") + c for c in need))
        df = pd.read_csv(fs[0], usecols=[c for c in ["speedInKmPerHour", "type"] if c in cols],
                         encoding=enc)
        v = df[df["type"] == "uv"]["speedInKmPerHour"] if "type" in df.columns \
            else df["speedInKmPerHour"]
        L.append(f"  v: mean {v.mean():.0f} p5 {v.quantile(.05):.0f} "
                 f"p95 {v.quantile(.95):.0f} km/h")

open(r"D:\driving_bc\reports\icing_scan.txt", "w", encoding="utf-8").write("\n".join(L))
print("written icing_scan.txt", len(L))
