# -*- coding: utf-8 -*-
"""
49_cnn_exam.py  --  감별자 고도화: 원시 CNN 판별자를 공식 스위트에 편입 (3층 시험).

목표 지시 이행: 시험 스위트 = 선형(참고) + GBM(주 시험) + **원시 CNN(최상급)**.
남산에서 ①RL 기준 구성 ②승자 구성(46d 응답+사전강조)의 CNN AUC를 공식 측정 —
매칭 진전이 원시 수준에서도 실재하는지 검증(문헌: NN-로짓 C2ST 계열).

  python 49_cnn_exam.py
"""
import os, json, importlib
import numpy as np
import torch
import torch.nn as nn

from common import ART, REP, CACHE, gen_split
from driving_env import DrivingEnv, load_roads, rollout, trim_roads, OBS_DIM

p20 = importlib.import_module("20_profile_eval")
p22 = importlib.import_module("22_v3_spectral")
p27 = importlib.import_module("27_raw_cnn_eval")
p34 = importlib.import_module("34_virtual_cohort")
p39 = importlib.import_module("39_discriminator_audit")
p42 = importlib.import_module("42_preemphasis")
p44 = importlib.import_module("44_bc_native")

np.random.seed(0); torch.manual_seed(0)
GAIN = 0.012


def sym_units(trajs, dd):
    return [p39.symmetric_signals(t, r, dd, GAIN) for t, r in trajs]


def cnn_auc(H_sigs, S_sigs, w=20, stride=10):
    Xs, ys, gs = [], [], []
    gid = 0
    for cls, units in [(0, H_sigs), (1, S_sigs)]:
        for u in units:
            for win in p27.unit_windows(u, w, stride):
                Xs.append(win); ys.append(cls); gs.append(gid)
            gid += 1
    return p27.cnn_group_cv(np.stack(Xs), np.array(ys), np.array(gs))


def main():
    roads, _, dd = load_roads(os.path.join(CACHE, "env_roads_namsan.npz"))
    roads = trim_roads(roads)
    ch = p34.load_champion(base="rl_multi.zip")
    cal = json.load(open(os.path.join(REP, "v3_library_2024_multi.json"), encoding="utf-8"))
    sigma = float(cal["sigma"])
    H = [p20.human_signals(r, dd) for r in roads]
    env = DrivingEnv([dict(r, e_ref=np.zeros_like(r["v_ref"])) for r in roads],
                     dd=dd, record=True, steer_gain=GAIN)

    def collect(mk_pol):
        T = []
        for k in range(len(roads)):
            for j in range(2):
                pol = mk_pol(7000 + k * 20 + j)
                pol.reset()
                traj, _ = rollout(env, pol, k)
                if len(traj) > 60:
                    T.append((traj, roads[k]))
        return T

    out = {}
    # ① RL 기준 구성 (v3.3)
    A = collect(lambda sd: p22.SpectralPolicy(ch["model"], ch["fr"], ch["A"], sigma,
                                              lib=ch["lib"], seed=sd))
    out["baseline_rl"] = cnn_auc(H, sym_units(A, dd))
    print(f"[CNN 200m] RL 기준: {out['baseline_rl']:.3f}", flush=True)
    # ② 승자 구성 (46d + 사전강조 β2)
    net = nn.Sequential(nn.Linear(OBS_DIM, 64), nn.ReLU(),
                        nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2), nn.Tanh())
    net.load_state_dict(torch.load(os.path.join(ART, "bc_dagger46_inj.pt")))
    net.eval()
    bc = p44.BCAdapter(net)
    knobs = json.load(open(os.path.join(REP, "dagger_preemph.json"), encoding="utf-8"))
    b_b = knobs.get("namsan_knobs", {}).get("beta", 2.0)
    s_b = knobs.get("namsan_knobs", {}).get("sigma", 0.163)
    B = collect(lambda sd: p42.PreemphPolicy(bc, ch["fr"], ch["A"], s_b,
                                             lib=ch["lib"], beta=b_b, seed=sd))
    out["winner_46d42"] = cnn_auc(H, sym_units(B, dd))
    print(f"[CNN 200m] 승자(46d+42): {out['winner_46d42']:.3f}", flush=True)
    json.dump(out, open(os.path.join(REP, "cnn_exam_namsan.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved cnn_exam_namsan.json", flush=True)


if __name__ == "__main__":
    main()
