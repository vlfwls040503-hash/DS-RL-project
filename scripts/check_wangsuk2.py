# -*- coding: utf-8 -*-
import os, glob, re
import pandas as pd

D = r"<NAS_PATH set your own>"

files = glob.glob(os.path.join(D, "**", "*.csv"), recursive=True)
print("files:", len(files))
cols = list(pd.read_csv(files[0], nrows=0, encoding="utf-8-sig").columns)
need = ["laneCurvature", "roadLongitudinalSlope", "roadLateralSlope", "laneWidth",
        "carriagewayWidth", "distanceAlongRoad", "time", "scenarioTime"]
for c in need:
    print("O" if c in cols else "X", c)

names = [os.path.basename(x) for x in files]
pat = re.compile(r"No\.(\d+)_S(\d)")
ok = [m for m in (pat.search(n) for n in names) if m]
subs = sorted({int(m.group(1)) for m in ok})
conds = sorted({int(m.group(2)) for m in ok})
print("tagged:", len(ok), "/", len(names), "| subjects:", len(subs), subs[:10],
      "| conds:", conds)

# 폴더 구조 힌트 (피험자별 하위폴더?)
dirs = sorted({os.path.relpath(os.path.dirname(f), D) for f in files})
print("subdirs:", len(dirs), dirs[:6])

# 속도 분포 훑기 (파일 3개)
import numpy as np
for f in files[:3]:
    df = pd.read_csv(f, usecols=["speedInKmPerHour", "type"], encoding="utf-8-sig")
    v = df[df["type"] == "uv"]["speedInKmPerHour"] if "type" in df.columns else df["speedInKmPerHour"]
    print(os.path.basename(f)[:60], f"v: mean {v.mean():.0f} p5 {v.quantile(.05):.0f} p95 {v.quantile(.95):.0f} km/h")
