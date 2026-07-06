# -*- coding: utf-8 -*-
"""지하유출입부 2021 / 결빙 2022 데이터 정찰 (읽기전용): 경로·스키마·속도역."""
import os, glob
import pandas as pd

ROOT = r"<NAS_PATH set your own>"
KEYS = ["결빙", "유출입", "지하"]
L = []
for year in sorted(os.listdir(ROOT)):
    yp = os.path.join(ROOT, year)
    if not os.path.isdir(yp) or not any(y in year for y in ["2020", "2021", "2022", "2023"]):
        continue
    for s in sorted(os.listdir(yp)):
        if any(k in s for k in KEYS):
            L.append(f"{year} | {s}")

out = r"D:\driving_bc\reports\nas_lowspeed_scan.txt"
open(out, "w", encoding="utf-8").write("\n".join(L))
print("written", out, len(L), "lines")
