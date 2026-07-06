# -*- coding: utf-8 -*-
"""
30_build_multi.py  --  B2용 합본 도로 캐시 (2024 고속 + 왕숙 다속도).

- 왕숙은 max|curv|<=0.02만 (헤어핀 램프 제외 — 조향권한 0.025로 커버 가능한 범위)
- 왕숙 피험자 번호 +100 오프셋 (2024와 분할 충돌 방지)
Output: cache/env_roads_multi.npz
"""
import os
import numpy as np
from common import CACHE
from driving_env import load_roads, save_roads

import sys
K_MAX = float(sys.argv[1]) if len(sys.argv) > 1 else 0.005   # 여유율 원칙: 권한/2.4
OUT_TAG = sys.argv[2] if len(sys.argv) > 2 else ""

r24, _, dd = load_roads(os.path.join(CACHE, "env_roads_2024.npz"))
rw, _, ddw = load_roads(os.path.join(CACHE, "env_roads_wangsuk.npz"))
assert dd == ddw
kept = []
for r in rw:
    if float(np.abs(r["curv"]).max()) <= K_MAX:
        r2 = dict(r)
        r2["subject"] = int(r["subject"]) + 100
        kept.append(r2)
print(f"2024 {len(r24)} + wangsuk {len(kept)}/{len(rw)} (max|curv|<={K_MAX})")
roads = r24 + kept
save_roads(os.path.join(CACHE, f"env_roads_multi{OUT_TAG}.npz"), roads, dd=dd)
v = np.array([np.mean(r["v_ref"]) for r in roads]) * 3.6
print(f"multi: {len(roads)} roads, subjects {len(set(r['subject'] for r in roads))}, "
      f"v p5-p95 {np.percentile(v,5):.0f}-{np.percentile(v,95):.0f} km/h")
print("saved -> env_roads_multi.npz")
