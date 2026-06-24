# -*- coding: utf-8 -*-
import os, glob
import pandas as pd
import numpy as np

D = r"<NAS_PATH set your own>"

CORE = ["speedInKmPerHour","steering","throttle","brake","offsetFromLaneCenter","laneCurvature",
        "distanceToLeftBorder","distanceToRightBorder","standardDeviationFromLaneCenter",
        "distanceToFrontVehicle","TTCToFrontVehicle","position X","position Y","type"]

if not os.path.isdir(D):
    print("PATH NOT FOUND:", D); raise SystemExit

files = glob.glob(os.path.join(D, "**", "*.csv"), recursive=True)
print(f"csv files: {len(files)}")
for f in files[:8]:
    print("   ", os.path.basename(f))

if files:
    f0 = files[0]
    for enc in ("utf-8-sig","cp949","utf-8","latin-1"):
        try:
            cols = list(pd.read_csv(f0, nrows=0, encoding=enc).columns); used=enc; break
        except Exception: continue
    print(f"\nsample: {os.path.basename(f0)}  (enc={used}, cols={len(cols)})")
    print("핵심 컬럼 존재여부:")
    for c in CORE:
        print(f"   {'O' if c in cols else 'X'}  {c}")
    # type values + multi-vehicle check
    df = pd.read_csv(f0, nrows=30000, encoding=used, low_memory=False)
    if "type" in df.columns:
        print("\ntype 분포(상위3만행):", df["type"].value_counts().to_dict())
    # vehicles per timestamp (using scenarioTime if present)
    tcol = "scenarioTime" if "scenarioTime" in df.columns else ("time" if "time" in df.columns else None)
    if tcol and "type" in df.columns:
        g = df.groupby(tcol)["type"].count()
        print(f"시점당 차량행 수: 평균 {g.mean():.1f}, 최대 {g.max()}  → 주변차량 계산 {'가능' if g.max()>1 else '불가(혼자주행)'}")
    # subjects/condition hint from filenames
    import re
    names = [os.path.basename(x) for x in files]
    jiha = sum(1 for n in names if "지하" in n); jisang = sum(1 for n in names if "지상" in n or "(상)" in n)
    print(f"\n파일명 태그: 지하={jiha} 지상={jisang} 기타={len(names)-jiha-jisang}")
