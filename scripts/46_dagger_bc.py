# -*- coding: utf-8 -*-
"""
46_dagger_bc.py  --  후보 A: DAgger 안정화 BC 하이브리드.

44의 게이트 즉사(BC 드리프트)가 증명한 전제조건 이행: 학생(BC)이 직접 몰며 방문한
상태를 특권 교사(PD, 사람 e_ref 추종)가 재라벨 → 복합오차 교정 데이터로 재학습 반복.
목표: 사람 유래 응답함수(온분포)를 유지한 채 닫힌루프 생존 → 공정 GBM에서 루프
서명이 RL 정책과 다른지 판정.

  python 46_dagger_bc.py
"""
import os, json, importlib
import numpy as np
import torch
import torch.nn as nn

from common import ART, REP, CACHE, gen_split
from driving_env import (DrivingEnv, load_roads, rollout, trim_roads,
                         make_expert_dataset, pd_action, OBS_DIM)

p20 = importlib.import_module("20_profile_eval")
p21 = importlib.import_module("21_validation")
p22 = importlib.import_module("22_v3_spectral")
p34 = importlib.import_module("34_virtual_cohort")
p39 = importlib.import_module("39_discriminator_audit")
p43 = importlib.import_module("43_gbm_anatomy")
p44 = importlib.import_module("44_bc_native")

np.random.seed(0); torch.manual_seed(0)
GAIN = 0.012


def fit(net, X, Y, epochs=12):
    opt = torch.optim.Adam(net.parameters(), lr=5e-4)
    lossf = nn.SmoothL1Loss()
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        for s in range(0, len(Xt), 1024):
            b = perm[s:s + 1024]
            opt.zero_grad()
            loss = lossf(net(Xt[b]), Yt[b]); loss.backward(); opt.step()
    net.eval()
    return net


def main():
    roads8, _, dd8 = load_roads(os.path.join(CACHE, "env_roads_multi8.npz"))
    roads8 = trim_roads(roads8)
    subj = np.array([r["subject"] for r in roads8], "int64")
    tr, va, te = gen_split(subj, seed=0)
    train_roads = [r for r, m in zip(roads8, tr) if m]
    val_roads = [r for r, m in zip(roads8, va) if m]

    import sys
    argv = sys.argv[1:]
    SMOOTH_W = int(argv[0]) if argv else 9
    INJ = "inj" in argv                                # 주입-인지 DAgger (46d)
    CURVY = "curvy" in argv                            # 남산급 곡률 DAgger (46e, 기각됨)
    ROUNDS = next((int(a[1:]) for a in argv if a.startswith("r") and a[1:].isdigit()), 2)
    DART = "dart" in argv                              # 수집시 행동노이즈(커버리지 확장)
    TAG = (f"_s{SMOOTH_W}" if SMOOTH_W != 9 else "") + ("_inj" if INJ else "") \
        + ("_curvy" if CURVY else "") + (f"_r{ROUNDS}" if ROUNDS != 2 else "") \
        + ("_dart" if DART else "")
    print(f"expert label smooth_w={SMOOTH_W}", flush=True)
    X0, Y0 = make_expert_dataset(train_roads, dd8, gain=GAIN, smooth_w=SMOOTH_W)
    net = nn.Sequential(nn.Linear(OBS_DIM, 64), nn.ReLU(),
                        nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2), nn.Tanh())
    sd_path = os.path.join(ART, "bc_native44.pt")
    if SMOOTH_W == 9 and os.path.exists(sd_path):
        net.load_state_dict(torch.load(sd_path))
        print("BC init from bc_native44.pt", flush=True)
    else:
        fit(net, X0, Y0, epochs=30)
    bc = p44.BCAdapter(net)

    # ---- DAgger 반복: 학생 주행 → PD 교사 재라벨 ----
    wlib = None
    if INJ:                                            # 평가 과제(주입) 그대로 재라벨
        wlib = []
        r24, _, dd24 = load_roads(os.path.join(CACHE, "env_roads_2024.npz"))
        r24t = trim_roads(r24)
        s24 = np.array([r["subject"] for r in r24t], "int64")
        t24, v24, _ = gen_split(s24, seed=0)
        for r in [r for r, m in zip(r24t, t24 | v24) if m]:
            for ch_ in p21.chunk_signals(p20.human_signals(r, dd24)):
                x = np.asarray(ch_["e"], np.float64); x -= x.mean()
                if len(x) >= 400 and x.std() > 1e-3:
                    wlib.append((x / x.std()).astype(np.float64))
        print(f"injection-aware DAgger: {len(wlib)} chunks", flush=True)
    dag_roads = list(train_roads)
    if CURVY:                                          # 남산(0.0106)급 곡률 커버
        rw, _, ddw = load_roads(os.path.join(CACHE, "env_roads_wangsuk.npz"))
        rw = trim_roads(rw)
        add = [dict(r, subject=int(r["subject"]) + 100) for r in rw
               if 0.008 < float(np.abs(r["curv"]).max()) <= 0.0105]
        dag_roads += add
        print(f"curvy DAgger roads +{len(add)}", flush=True)
    env_t = DrivingEnv(dag_roads, dd=dd8, record=False, steer_gain=GAIN,
                       wander_lib=wlib, wander_sigma=0.2 if INJ else 0.0)
    if INJ:
        env_t.wander_obs = True
    Xagg, Yagg = [X0], [Y0]
    for rnd in range(1, ROUNDS + 1):
        Xr, Yr = [], []
        for k in range(0, len(dag_roads), 2):            # 절반 도로 샘플
            obs, _ = env_t.reset(options={"road_idx": k})
            done = 0
            while not done:
                a_t = pd_action(env_t)                   # 특권 교사 (사람 e_ref 추종)
                a_s = bc.predict(obs)[0]
                if DART:
                    a_s = a_s + np.random.randn(2).astype(np.float32) * [0.08, 0.05]
                Xr.append(obs); Yr.append(a_t)
                obs, _, term, trunc, _ = env_t.step(a_s)
                done = term or trunc
        Xagg.append(np.asarray(Xr, np.float32)); Yagg.append(np.asarray(Yr, np.float32))
        fit(net, np.vstack(Xagg), np.vstack(Yagg), epochs=8)
        # 게이트: val 닫힌루프 (주입 없음)
        env_v = DrivingEnv(val_roads, dd=dd8, record=True, steer_gain=GAIN)
        offs = 0
        for k in range(len(val_roads)):
            _, off = rollout(env_v, lambda o, e: bc.predict(o)[0], k)
            offs += int(off)
        print(f"[DAgger r{rnd}] +{len(Xr):,} labels | val 이탈 {offs}/{len(val_roads)}",
              flush=True)
    torch.save(net.state_dict(), os.path.join(ART, f"bc_dagger46{TAG}.pt"))

    # ---- 남산 게이트 (청크 주입) + σ 보정 ----
    ch = p34.load_champion()
    cal = json.load(open(os.path.join(REP, "v3_library_2024_multi.json"), encoding="utf-8"))
    sigma = float(cal["sigma"])
    roadsN, _, ddN = load_roads(os.path.join(CACHE, "env_roads_namsan.npz"))
    roadsN = trim_roads(roadsN)
    sN = np.array([r["subject"] for r in roadsN], "int64")
    _, vaN, _ = gen_split(sN, seed=0)
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
    print(f"[남산 게이트] 이탈 {off_rate:.2f} SDLP {np.mean(sds):.3f}", flush=True)
    if off_rate > 0.3:
        print("GATE FAIL", flush=True)
        json.dump(dict(gate="FAIL", off=off_rate),
                  open(os.path.join(REP, "dagger_bc.json"), "w", encoding="utf-8"))
        return
    hv = [p20.human_signals(r, ddN) for r in valN]
    tgt_sd = float(np.mean([np.std(h["e"]) for h in hv]))
    sigma2 = float(np.clip(sigma * (tgt_sd / max(np.mean(sds), 1e-6)), 0.02, 1.2))

    # ---- 공정시험 양 도메인 ----
    out = {"gate_off": off_rate, "sigma": sigma2}
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
        out[exp] = dict(auc=float(auc), off=offc / max(len(roads) * per_road, 1))
        print(f"[{exp}] DAgger-BC GBM AUC={auc:.3f} off={out[exp]['off']:.2f} | 상위: "
              + ", ".join(f"{p43.FEATS[i]}={imp[i]:.3f}" for i in top), flush=True)
    json.dump(out, open(os.path.join(REP, f"dagger_bc{TAG}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"saved dagger_bc{TAG}.json", flush=True)


if __name__ == "__main__":
    main()
