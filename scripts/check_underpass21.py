# -*- coding: utf-8 -*-
"""지하유출입 R&D 2차년도(2021) 주행데이터 정찰 (읽기전용)."""
import os, glob
import pandas as pd

RND = (r"<NAS_PATH set your own>"
       r"\국가연구개발사업(R&D), 다년도 연구용역"
       r"\(2020)운전자 형태를 고려한 지하교통 유출입부 입체 연결부 인프라 설계 가이드라인 개발"
       r"\2차년도(2021)\4. 과업수행단계")
L = [f"ROOT exists: {os.path.isdir(RND)}"]
if os.path.isdir(RND):
    for a in sorted(os.listdir(RND)):
        L.append("  " + a)
        ap = os.path.join(RND, a)
        if os.path.isdir(ap):
            for b in sorted(os.listdir(ap))[:8]:
                L.append("      " + b)
    csvs = glob.glob(os.path.join(RND, "**", "*.csv"), recursive=True)
    L.append(f"CSV total: {len(csvs)}")
    for f in csvs[:5]:
        L.append("  ex: " + os.path.relpath(f, RND))
    if csvs:
        for enc in ("utf-8-sig", "cp949", "utf-8"):
            try:
                cols = list(pd.read_csv(csvs[0], nrows=0, encoding=enc).columns)
                break
            except Exception:
                continue
        need = ["distanceAlongRoad", "offsetFromLaneCenter", "speedInKmPerHour",
                "laneCurvature", "laneWidth", "carriagewayWidth",
                "roadLongitudinalSlope", "time", "type"]
        L.append("cols: " + " ".join(("O" if c in cols else "X") + c for c in need))

open(r"D:\driving_bc\reports\underpass21_scan.txt", "w", encoding="utf-8").write("\n".join(L))
print("written", len(L))
