# -*- coding: utf-8 -*-
import os, pandas as pd
D = r"<NAS_PATH set your own>"
sang = os.path.join(D, "피실험자1_지상주행.csv")
ha = os.path.join(D, "피실험자1_지하주행.csv")
cs = list(pd.read_csv(sang, nrows=0, encoding="utf-8-sig").columns)
ch = list(pd.read_csv(ha, nrows=0, encoding="utf-8-sig").columns)
print("지상 cols:", len(cs), " 지하 cols:", len(ch))
print("\n지하에 있고 지상에 없는 핵심:", [c for c in ch if c not in cs])
print("\n지상에서 관심 키워드 매칭:")
for kw in ["Lane", "lane", "Deviation", "standard", "Standard", "Front", "front",
           "TTC", "Border", "Offset", "offset", "Curv", "deviation"]:
    hits = [c for c in cs if kw in c]
    if hits:
        print(f"  [{kw}] -> {hits}")
print("\n지상 컬럼 전체:")
for i, c in enumerate(cs):
    print(f"{i:3d} {c}")
