# -*- coding: utf-8 -*-
"""
48_origin_chunks.py  --  원산지 청크: 결빙2022(47~49km/h) 잔차 라이브러리 → 남산 시험.

주입 질감의 원재료를 2024 고속 청크(100km/h)에서 남산과 동급 속도역의 실제 사람
잔차로 교체. 승자 구성(46d 응답 + 사전강조) 위에서 도메인 보정 후 공정시험.

  python 48_origin_chunks.py
"""
import os, glob, json, importlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import REP, CACHE, gen_split, read_csv_fallback
from driving_env import DrivingEnv, load_roads, rollout, trim_roads, OBS_DIM

p20 = importlib.import_module("20_profile_eval")
p21 = importlib.import_module("21_validation")
p34 = importlib.import_module("34_virtual_cohort")
p39 = importlib.import_module("39_discriminator_audit")
p41 = importlib.import_module("41_midband")
p42 = importlib.import_module("42_preemphasis")
p43 = importlib.import_module("43_gbm_anatomy")
p44 = importlib.import_module("44_bc_native")

np.random.seed(0); torch.manual_seed(0)
GAIN = 0.012
ICE = r"<NAS_PATH set your own>"
LIB_NPZ = os.path.join(CACHE, "icing_chunks.npz")


def build_lib():
    if os.path.exists(LIB_NPZ):
        d = np.load(LIB_NPZ)
        return [d[k] for k in d.files]
    lib = []
    files = sorted(glob.glob(os.path.join(ICE, "**", "*.csv"), recursive=True))
    for f in files:
        try:
            df = read_csv_fallback(f, usecols=lambda c: c in
                                   {"type", "distanceAlongRoad", "offsetFromLaneCenter"})
        except Exception:
            continue
        if "distanceAlongRoad" not in df.columns or "offsetFromLaneCenter" not in df.columns:
            continue
        uv = df[df["type"] == "uv"] if "type" in df.columns else df
        s = pd.to_numeric(uv["distanceAlongRoad"], errors="coerce").to_numpy(float)
        e = pd.to_numeric(uv["offsetFromLaneCenter"], errors="coerce").to_numpy(float)
        ok = np.isfinite(s) & np.isfinite(e)
        s, e = s[ok], e[ok]
        if len(s) < 300:
            continue
        o = np.argsort(s); s, e = s[o], e[o]
        g = np.arange(s[0], s[-1], 10.0)
        eg = np.interp(g, s, e)
        # 2024 청크 프로토콜과 동일: 평균만 제거(장파 배회 보존) — 300m 디트렌드는
        # std_e 분포를 깨뜨림(1차 시도 실패 원인)
        res = eg - eg.mean()
        if len(res) >= 200 and res.std() > 1e-3:
            lib.append((res / res.std()).astype(np.float64))
    np.savez_compressed(LIB_NPZ, *lib)
    return lib


def main():
    lib = build_lib()
    print(f"icing chunk lib: {len(lib)}개 (중앙길이 "
          f"{np.median([len(c) for c in lib])*10/1000:.1f}km)", flush=True)

    net = nn.Sequential(nn.Linear(OBS_DIM, 64), nn.ReLU(),
                        nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2), nn.Tanh())
    from common import ART
    net.load_state_dict(torch.load(os.path.join(ART, "bc_dagger46_inj.pt")))
    net.eval()
    bc = p44.BCAdapter(net)
    ch = p34.load_champion()          # fr/A 스펙트럼만 (lib은 교체)
    cal = json.load(open(os.path.join(REP, "v3_library_2024_multi.json"), encoding="utf-8"))
    sigma = float(cal["sigma"])

    roadsN, _, ddN = load_roads(os.path.join(CACHE, "env_roads_namsan.npz"))
    roadsN = trim_roads(roadsN)
    sN = np.array([r["subject"] for r in roadsN], "int64")
    _, vaN, _ = gen_split(sN, seed=0)
    valN = [r for r, m in zip(roadsN, vaN) if m]
    env_v = DrivingEnv([dict(r, e_ref=np.zeros_like(r["v_ref"])) for r in valN],
                       dd=ddN, record=True, steer_gain=GAIN)
    hv = [p20.human_signals(r, ddN) for r in valN]
    tgt_de, _, tgt_sd = p41.stats(hv)

    best = None
    for beta in [0.5, 1.0, 2.0]:
        sds, des, offs = [], [], 0
        for k in range(len(valN)):
            pol = p42.PreemphPolicy(bc, ch["fr"], ch["A"], sigma, lib=lib,
                                    beta=beta, seed=61 + k)
            pol.reset()
            traj, off = rollout(env_v, pol, k)
            offs += int(off)
            if len(traj) > 60:
                s = p39.symmetric_signals(traj, valN[k], ddN, GAIN)
                sds.append(float(np.std(s["e"])))
                des.append(float(np.mean(np.abs(np.diff(s["e"])))))
        gap = abs(np.mean(des) - tgt_de) / tgt_de + 0.5 * abs(np.mean(sds) - tgt_sd) / tgt_sd \
            + 100.0 * (offs / len(valN) > 0.15)
        print(f"  beta={beta}: |de|={np.mean(des):.4f}({tgt_de:.4f}) "
              f"SDLP={np.mean(sds):.3f}({tgt_sd:.3f}) off={offs} gap={gap:.3f}", flush=True)
        if best is None or gap < best[1]:
            best = (beta, gap, float(np.mean(sds)))
    b_b, _, sd_now = best
    sigma2 = float(np.clip(sigma * (tgt_sd / max(sd_now, 1e-6)), 0.02, 1.2))
    print(f"선택 beta={b_b} σ->{sigma2:.3f}", flush=True)

    roads = roadsN
    env = DrivingEnv([dict(r, e_ref=np.zeros_like(r["v_ref"])) for r in roads],
                     dd=ddN, record=True, steer_gain=GAIN)
    T, offc = [], 0
    for k in range(len(roads)):
        for j in range(2):
            pol = p42.PreemphPolicy(bc, ch["fr"], ch["A"], sigma2, lib=lib,
                                    beta=b_b, seed=7000 + k * 20 + j)
            pol.reset()
            traj, off = rollout(env, pol, k)
            offc += int(off)
            if len(traj) > 60:
                T.append((traj, roads[k]))
    S = [p21.seg_features(p39.symmetric_signals(t, r, ddN, GAIN)) for t, r in T]
    H = [p21.seg_features(p20.human_signals(r, ddN)) for r in roads]
    rngb = np.random.RandomState(1)
    nu = min(len(H), len(S))
    Hb = [H[i] for i in rngb.choice(len(H), nu, replace=False)]
    Sb = [S[i] for i in rngb.choice(len(S), nu, replace=False)]
    auc, imp = p43.fair_exam(Hb, Sb, ret_imp=True)
    top = np.argsort(-imp)[:3]
    print(f"[namsan] 원산지청크 GBM AUC={auc:.3f} off={offc/max(len(roads)*2,1):.2f} | 상위: "
          + ", ".join(f"{p43.FEATS[i]}={imp[i]:.3f}" for i in top), flush=True)
    json.dump(dict(auc=float(auc), beta=b_b, sigma=sigma2, n_chunks=len(lib)),
              open(os.path.join(REP, "origin_chunks.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved origin_chunks.json", flush=True)


if __name__ == "__main__":
    main()
