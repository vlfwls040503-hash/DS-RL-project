# -*- coding: utf-8 -*-
"""홍대/조명 관련 프로젝트 폴더를 NAS에서 얕게 탐색 (읽기전용)."""
import os

BASE = r"<NAS_PATH set your own>"
ROOT = os.path.join(BASE, "연구실 업무과제 모음")
KEYS = ["홍대", "조명", "지하차도"]

L = []
TP = os.path.join(BASE, "연구실 자료", "실험 데이터+영상")
L.append("== 실험 데이터+영상 ==")
for s in sorted(os.listdir(TP)):
    mark = " <<<" if any(k in s for k in KEYS) else ""
    L.append("  " + s + mark)
    sp = os.path.join(TP, s)
    if os.path.isdir(sp):
        try:
            for x in sorted(os.listdir(sp)):
                m2 = " <<<" if any(k in x for k in KEYS) else ""
                L.append("      " + x + m2)
        except Exception:
            pass

out = r"D:\driving_bc\reports\nas_hongdae_scan3.txt"
open(out, "w", encoding="utf-8").write("\n".join(L))
print("written", out, len(L), "lines")
