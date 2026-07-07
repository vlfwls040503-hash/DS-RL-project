# -*- coding: utf-8 -*-
"""
56_pilot_scaling.py  --  파일럿 스케일링 법칙: 퓨샷 AUC vs 파일럿 인원 (5/10/19명).

목적: "몇 명의 파일럿 주행이면 0.55에 닿는가"를 실측 곡선으로 — 실험실의 추가 수집
결정에 정량 요구서 제공. 프로토콜은 54와 동일(te 10명 분리, 3시드 복제).

  python 56_pilot_scaling.py
"""
import os, json, importlib
import numpy as np
import torch
import torch.nn as nn

from common import ART, REP, CACHE, gen_split
from driving_env import DrivingEnv, load_roads, rollout, trim_roads, OBS_DIM

p20 = importlib.import_module("20_profile_eval")
p21 = importlib.import_module("21_validation")
p39 = importlib.import_module("39_discriminator_audit")
p43 = importlib.import_module("43_gbm_anatomy")
p44 = importlib.import_module("44_bc_native")
p51 = importlib.import_module("51_trackA_diffusion")
p52 = importlib.import_module("52_best_stack")
p54 = importlib.import_module("54_fewshot_namsan")

np.random.seed(0); torch.manual_seed(0)
GAIN = 0.012


def main():
    net = nn.Sequential(nn.Linear(OBS_DIM, 64), nn.ReLU(),
                        nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2), nn.Tanh())
    net.load_state_dict(torch.load(os.path.join(ART, "bc_dagger46_inj.pt")))
    net.eval()
    bc = p44.BCAdapter(net)

    rN, _, ddN = load_roads(os.path.join(CACHE, "env_roads_namsan.npz"))
    rN = trim_roads(rN)
    sN = np.array([r["subject"] for r in rN], "int64")
    trN, vaN, teN = gen_split(sN, seed=0)
    tr_r = [r for r, m in zip(rN, trN) if m]
    va_r = [r for r, m in zip(rN, vaN) if m]
    te_r = [r for r, m in zip(rN, teN) if m]
    tr_subj = sorted(set(r["subject"] for r in tr_r))

    m8, _, dd8 = load_roads(os.path.join(CACHE, "env_roads_multi8.npz"))
    m8 = trim_roads(m8)
    s8 = np.array([r["subject"] for r in m8], "int64")
    t8, v8, _ = gen_split(s8, seed=0)
    base_corpus = [r for r, m in zip(m8, t8 | v8) if m]
    for extra in ["icing", "underpass21"]:
        ex, _, _d = load_roads(os.path.join(CACHE, f"env_roads_{extra}.npz"))
        base_corpus += trim_roads(ex)

    Hte = [p21.seg_features(p20.human_signals(r, ddN)) for r in te_r]
    blind = [dict(r, e_ref=np.zeros_like(r["v_ref"])) for r in te_r]
    env = DrivingEnv(blind, dd=ddN, record=True, steer_gain=GAIN)
    env_v = DrivingEnv([dict(r, e_ref=np.zeros_like(r["v_ref"])) for r in va_r],
                       dd=ddN, record=True, steer_gain=GAIN)
    hv = [p20.human_signals(r, ddN) for r in va_r]
    tgt_sd = float(np.mean([np.std(h["e"]) for h in hv]))

    out = {}
    rngp = np.random.RandomState(7)
    for n_pilot in [5, 10, 19]:
        subs = tr_subj if n_pilot >= len(tr_subj) else \
            sorted(rngp.choice(tr_subj, n_pilot, replace=False).tolist())
        pr = [r for r in tr_r if r["subject"] in subs]
        p54.train_fewshot(base_corpus + pr, dd8)
        libN = []
        for r in pr:
            x = np.asarray(p20.human_signals(r, ddN)["e"], np.float64)
            x -= x.mean()
            if len(x) >= 200 and x.std() > 1e-3:
                libN.append((x / x.std()).astype(np.float64))

        class FS(p52.BestStackPolicy):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                d = torch.load(p54.CKPT_FS, map_location=p51.DEV, weights_only=False)
                self.net = p51.Denoiser().to(p51.DEV)
                self.net.load_state_dict(d["state"]); self.net.eval()
                self.mu, self.sd, self.r_sd = d["mu_c"], d["sd_c"], float(d["r_sd"])

        # σ 1노브(β3 τ0.8 고정 — 54 최적 레시피)
        s0 = 0.2
        for it in range(2):
            sds = []
            for k in range(len(va_r)):
                pol = FS(bc, beta=3.0, tau=0.8, sigma=s0, lib=libN, seed=90 + k)
                pol.reset()
                traj, _ = rollout(env_v, pol, k)
                if len(traj) > 60:
                    sds.append(float(np.std(
                        p39.symmetric_signals(traj, va_r[k], ddN, GAIN)["e"])))
            s0 = float(np.clip(s0 * tgt_sd / max(np.mean(sds), 1e-6), 0.03, 1.0))
        aucs = []
        for rep in range(3):
            T = []
            for k in range(len(blind)):
                for j in range(4):
                    pol = FS(bc, beta=3.0, tau=0.8, sigma=s0, lib=libN,
                             seed=7000 + k * 20 + j + rep * 100000)
                    pol.reset()
                    traj, _ = rollout(env, pol, k)
                    if len(traj) > 60:
                        T.append((traj, te_r[k]))
            S = [p21.seg_features(p39.symmetric_signals(t, r, ddN, GAIN)) for t, r in T]
            aucs.append(p43.fair_exam(list(Hte), S))
        out[n_pilot] = dict(reps=[float(a) for a in aucs],
                            mean=float(np.mean(aucs)), std=float(np.std(aucs)),
                            sigma=s0, n_chunks=len(libN))
        print(f"[pilot={n_pilot}] {[round(a,3) for a in aucs]} -> "
              f"{np.mean(aucs):.3f}±{np.std(aucs):.3f}", flush=True)
    json.dump(out, open(os.path.join(REP, "pilot_scaling.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved pilot_scaling.json", flush=True)


if __name__ == "__main__":
    main()
