# -*- coding: utf-8 -*-
"""
44_bc_native.py  --  정책 수술 3호: BC-네이티브 조향 (사람 유래 폐루프 응답).

3연속 수술 무효의 결론: 고정 RL 정책+목표 주입으로는 루프 전달함수(교정 동역학
서명)를 못 바꾼다. BC는 사람의 obs→조향 매핑을 직접 회귀 — 폐루프 응답 자체가
사람 유래. 드리프트(역사적 약점)는 청크 목표 주입이 보정하는지 게이트로 확인.

  python 44_bc_native.py
"""
import os, json, importlib
import numpy as np
import torch
import torch.nn as nn

from common import REP, CACHE, gen_split
from driving_env import (DrivingEnv, load_roads, rollout, trim_roads,
                         make_expert_dataset, OBS_DIM)

p20 = importlib.import_module("20_profile_eval")
p21 = importlib.import_module("21_validation")
p22 = importlib.import_module("22_v3_spectral")
p34 = importlib.import_module("34_virtual_cohort")
p39 = importlib.import_module("39_discriminator_audit")
p43 = importlib.import_module("43_gbm_anatomy")

np.random.seed(0); torch.manual_seed(0)
GAIN = 0.012


class BCAdapter:
    def __init__(self, net):
        self.net = net

    def predict(self, o, deterministic=True):
        with torch.no_grad():
            a = self.net(torch.from_numpy(np.asarray(o, "float32"))).numpy()
        return a, None


def main():
    roads8, _, dd8 = load_roads(os.path.join(CACHE, "env_roads_multi8.npz"))
    roads8 = trim_roads(roads8)
    subj = np.array([r["subject"] for r in roads8], "int64")
    tr, va, te = gen_split(subj, seed=0)
    train_roads = [r for r, m in zip(roads8, tr) if m]
    X, Y = make_expert_dataset(train_roads, dd8, gain=GAIN)
    print(f"expert pairs: {len(X):,}", flush=True)
    net = nn.Sequential(nn.Linear(OBS_DIM, 64), nn.ReLU(),
                        nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2), nn.Tanh())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.SmoothL1Loss()
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    for ep in range(30):
        perm = torch.randperm(len(Xt))
        for s in range(0, len(Xt), 1024):
            b = perm[s:s + 1024]
            opt.zero_grad()
            loss = lossf(net(Xt[b]), Yt[b]); loss.backward(); opt.step()
    net.eval()
    bc = BCAdapter(net)
    torch.save(net.state_dict(), os.path.join(REP, "..", "artifacts", "bc_native44.pt"))

    ch = p34.load_champion()               # 청크·스펙트럼 자산만 재사용
    cal = json.load(open(os.path.join(REP, "v3_library_2024_multi.json"), encoding="utf-8"))
    sigma = float(cal["sigma"])

    # ---- 게이트: 남산 val 안정성 (BC 드리프트) ----
    roadsN, _, ddN = load_roads(os.path.join(CACHE, "env_roads_namsan.npz"))
    roadsN = trim_roads(roadsN)
    sN = np.array([r["subject"] for r in roadsN], "int64")
    trN, vaN, teN = gen_split(sN, seed=0)
    valN = [r for r, m in zip(roadsN, vaN) if m]
    env_v = DrivingEnv([dict(r, e_ref=np.zeros_like(r["v_ref"])) for r in valN],
                       dd=ddN, record=True, steer_gain=GAIN)
    offs, sds = 0, []
    for k in range(len(valN)):
        pol = p22.SpectralPolicy(bc, ch["fr"], ch["A"], sigma, lib=ch["lib"], seed=k)
        pol.reset()
        traj, off = rollout(env_v, pol, k)
        offs += int(off)
        if len(traj) > 60:
            sds.append(float(np.std(p20.rl_signals(traj, gain=GAIN)["e"])))
    off_rate = offs / len(valN)
    print(f"[게이트] 남산 val: 이탈 {off_rate:.2f} SDLP {np.mean(sds):.3f}", flush=True)
    if off_rate > 0.3:
        print("GATE FAIL - BC 드리프트 미보정, 기록 후 종료", flush=True)
        json.dump(dict(gate="FAIL", off=off_rate),
                  open(os.path.join(REP, "bc_native.json"), "w", encoding="utf-8"))
        return

    # σ 1노브 보정
    hv = [p20.human_signals(r, ddN) for r in valN]
    tgt_sd = float(np.mean([np.std(h["e"]) for h in hv]))
    sigma2 = float(np.clip(sigma * (tgt_sd / max(np.mean(sds), 1e-6)), 0.02, 1.2))
    print(f"σ {sigma:.3f}->{sigma2:.3f}", flush=True)

    # ---- 공정시험 (양 도메인, 43 기계 재사용: BASE 대신 어댑터 주입) ----
    out = {}
    for exp, per_road in [("2024", 4), ("namsan", 2)]:
        roads, _, dd = load_roads(os.path.join(CACHE, f"env_roads_{exp}.npz"))
        roads = trim_roads(roads)
        if exp == "2024":
            s2 = np.array([r["subject"] for r in roads], "int64")
            _, _, te2 = gen_split(s2, seed=0)
            roads = [r for r, m in zip(roads, te2) if m]
        env = DrivingEnv([dict(r, e_ref=np.zeros_like(r["v_ref"])) for r in roads],
                         dd=dd, record=True, steer_gain=GAIN)
        T, offc = [], 0
        for k in range(len(roads)):
            for j in range(per_road):
                pol = p22.SpectralPolicy(bc, ch["fr"], ch["A"], sigma2, lib=ch["lib"],
                                         seed=7000 + k * 20 + j)
                pol.reset()
                traj, off = rollout(env, pol, k)
                offc += int(off)
                if len(traj) > 60:
                    T.append((traj, roads[k]))
        S = [p21.seg_features(p39.symmetric_signals(t, r, dd, GAIN)) for t, r in T]
        if exp == "2024":
            H = []
            for r in roads:
                for c in p21.chunk_signals(p20.human_signals(r, dd)):
                    H.append(p21.seg_features(c))
        else:
            H = [p21.seg_features(p20.human_signals(r, dd)) for r in roads]
        rngb = np.random.RandomState(1)
        nu = min(len(H), len(S))
        Hb = [H[i] for i in rngb.choice(len(H), nu, replace=False)]
        Sb = [S[i] for i in rngb.choice(len(S), nu, replace=False)]
        auc, imp = p43.fair_exam(Hb, Sb, ret_imp=True)
        top = np.argsort(-imp)[:3]
        out[exp] = dict(auc=float(auc), off=offc / max(len(roads) * per_road, 1),
                        importance=dict(zip(p43.FEATS, map(float, imp))))
        print(f"[{exp}] BC-네이티브 GBM AUC={auc:.3f} off={out[exp]['off']:.2f} | 상위: "
              + ", ".join(f"{p43.FEATS[i]}={imp[i]:.3f}" for i in top), flush=True)
    out["sigma"] = sigma2
    json.dump(out, open(os.path.join(REP, "bc_native.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved bc_native.json", flush=True)


if __name__ == "__main__":
    main()
