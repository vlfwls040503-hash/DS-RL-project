# -*- coding: utf-8 -*-
"""
47_dagger_preemph.py  --  승자 계보 결합: 46d(주입-인지 DAgger-BC) + 42(목표 사전강조).

  python 47_dagger_preemph.py
"""
import os, json, importlib
import numpy as np
import torch
import torch.nn as nn

from common import ART, REP, CACHE, gen_split
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


def main():
    net = nn.Sequential(nn.Linear(OBS_DIM, 64), nn.ReLU(),
                        nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2), nn.Tanh())
    net.load_state_dict(torch.load(os.path.join(ART, "bc_dagger46_inj.pt")))
    net.eval()
    bc = p44.BCAdapter(net)
    ch = p34.load_champion()
    cal = json.load(open(os.path.join(REP, "v3_library_2024_multi.json"), encoding="utf-8"))
    sigma = float(cal["sigma"])

    def calibrate(val_roads, dd):
        """도메인별 (β, σ) 보정 — 질감·SDLP 목표는 그 도메인 val 사람 기준."""
        env_v = DrivingEnv([dict(r, e_ref=np.zeros_like(r["v_ref"])) for r in val_roads],
                           dd=dd, record=True, steer_gain=GAIN)
        hv = [p20.human_signals(r, dd) for r in val_roads]
        tgt_de, _, tgt_sd = p41.stats(hv)
        best = None
        for beta in [0.5, 1.0, 2.0, 3.0]:
            sds, des, offs = [], [], 0
            for k in range(len(val_roads)):
                pol = p42.PreemphPolicy(bc, ch["fr"], ch["A"], sigma, lib=ch["lib"],
                                        beta=beta, seed=61 + k)
                pol.reset()
                traj, off = rollout(env_v, pol, k)
                offs += int(off)
                if len(traj) > 60:
                    s = p39.symmetric_signals(traj, val_roads[k], dd, GAIN)
                    sds.append(float(np.std(s["e"])))
                    des.append(float(np.mean(np.abs(np.diff(s["e"])))))
            gap = abs(np.mean(des) - tgt_de) / tgt_de \
                + 0.5 * abs(np.mean(sds) - tgt_sd) / tgt_sd \
                + 100.0 * (offs / len(val_roads) > 0.15)
            print(f"    beta={beta}: |de|={np.mean(des):.4f}({tgt_de:.4f}) "
                  f"SDLP={np.mean(sds):.3f}({tgt_sd:.3f}) gap={gap:.3f}", flush=True)
            if best is None or gap < best[1]:
                best = (beta, gap, float(np.mean(sds)))
        b_b, _, sd_now = best
        s2 = float(np.clip(sigma * (tgt_sd / max(sd_now, 1e-6)), 0.02, 1.2))
        return b_b, s2

    out = {}
    for exp, per_road in [("2024", 4), ("namsan", 2)]:
        roads, _, dd = load_roads(os.path.join(CACHE, f"env_roads_{exp}.npz"))
        roads = trim_roads(roads)
        s2_ = np.array([r["subject"] for r in roads], "int64")
        tr2, va2, te2 = gen_split(s2_, seed=0)
        val2 = [r for r, m in zip(roads, va2) if m]
        print(f"[{exp}] 도메인 보정:", flush=True)
        b_b, sigma2 = calibrate(val2, dd)
        print(f"  -> beta={b_b} sigma={sigma2:.3f}", flush=True)
        out[f"{exp}_knobs"] = dict(beta=b_b, sigma=sigma2)
        if exp == "2024":
            roads = [r for r, m in zip(roads, te2) if m]
        env = DrivingEnv([dict(r, e_ref=np.zeros_like(r["v_ref"])) for r in roads],
                         dd=dd, record=True, steer_gain=GAIN)
        T, offc = [], 0
        for k in range(len(roads)):
            for j in range(per_road):
                pol = p42.PreemphPolicy(bc, ch["fr"], ch["A"], sigma2, lib=ch["lib"],
                                        beta=b_b, seed=7000 + k * 20 + j)
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
        print(f"[{exp}] 46d+42 GBM AUC={auc:.3f} off={out[exp]['off']:.2f} | 상위: "
              + ", ".join(f"{p43.FEATS[i]}={imp[i]:.3f}" for i in top), flush=True)
    json.dump(out, open(os.path.join(REP, "dagger_preemph.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved dagger_preemph.json", flush=True)


if __name__ == "__main__":
    main()
