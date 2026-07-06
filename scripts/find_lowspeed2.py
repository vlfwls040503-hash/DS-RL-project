# -*- coding: utf-8 -*-
"""R&D 브랜치(유출입 과제) 연차 구조 + 결빙 심층 탐색 (읽기전용)."""
import os

ROOT = r"<NAS_PATH set your own>"
RND = os.path.join(ROOT, "국가연구개발사업(R&D), 다년도 연구용역",
                   "(2020)운전자 형태를 고려한 지하교통 유출입부 입체 연결부 인프라 설계 가이드라인 개발")
L = []
if os.path.isdir(RND):
    L.append("== 유출입 R&D 연차 ==")
    for y in sorted(os.listdir(RND)):
        L.append("  " + y)
        yp = os.path.join(RND, y)
        if os.path.isdir(yp):
            for s in sorted(os.listdir(yp))[:10]:
                L.append("      " + s)

# 결빙: 업무과제 전체 depth2에서 결빙/빙판/아이스 탐색
L.append("== 결빙 탐색 (depth2) ==")
for year in sorted(os.listdir(ROOT)):
    yp = os.path.join(ROOT, year)
    if not os.path.isdir(yp):
        continue
    try:
        for s in os.listdir(yp):
            sp = os.path.join(yp, s)
            if any(k in s for k in ["결빙", "빙판", "아이스"]):
                L.append(f"  {year} | {s}")
            elif os.path.isdir(sp):
                try:
                    for x in os.listdir(sp):
                        if any(k in x for k in ["결빙", "빙판", "아이스"]):
                            L.append(f"  {year} | {s} \\ {x}")
                except Exception:
                    pass
    except Exception:
        pass

out = r"D:\driving_bc\reports\nas_lowspeed_scan2.txt"
open(out, "w", encoding="utf-8").write("\n".join(L))
print("written", out, len(L))
