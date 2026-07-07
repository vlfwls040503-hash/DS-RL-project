# -*- coding: utf-8 -*-
"""
55_replicate_records.py  --  공식 기록 전체 3복제 검증 (goal '검증까지 완벽하게' 이행).

퓨샷 0.669가 시드 분산(±0.04)의 행운값으로 판명 → 모든 공식 기록에 동일 잣대.
  ①홈 52스택(0.760) ②남산 제로샷 46d+42(0.817) ③남산 퓨샷 기준선(0.763)
각 3시드 평균±표준편차로 확정.

  python 55_replicate_records.py
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
p52 = importlib.import_module("52_best_stack")

np.random.seed(0); torch.manual_seed(0)
GAIN = 0.012


def main():
    net = nn.Sequential(nn.Linear(OBS_DIM, 64), nn.ReLU(),
                        nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2), nn.Tanh())
    net.load_state_dict(torch.load(os.path.join(ART, "bc_dagger46_inj.pt")))
    net.eval()
    bc = p44.BCAdapter(net)
    r24, _, dd24 = load_roads(os.path.join(CACHE, "env_roads_2024.npz"))
    r24 = trim_roads(r24)
    s24 = np.array([r["subject"] for r in r24], "int64")
    _, va24, te24 = gen_split(s24, seed=0)
    fitlib = []
    for r in [r for r, m in zip(r24, ~te24) if m]:
        for ch_ in p21.chunk_signals(p20.human_signals(r, dd24)):
            x = np.asarray(ch_["e"], np.float64); x -= x.mean()
            if len(x) >= 400 and x.std() > 1e-3:
                fitlib.append((x / x.std()).astype(np.float64))

    def reps(exam_roads, dd, per_road, mk_pol, H_units, tag):
        blind = [dict(r, e_ref=np.zeros_like(r["v_ref"])) for r in exam_roads]
        env = DrivingEnv(blind, dd=dd, record=True, steer_gain=GAIN)
        aucs = []
        for rep in range(3):
            T = []
            for k in range(len(blind)):
                for j in range(per_road):
                    pol = mk_pol(7000 + k * 20 + j + rep * 100000)
                    pol.reset()
                    traj, _ = rollout(env, pol, k)
                    if len(traj) > 60:
                        T.append((traj, exam_roads[k]))
            S = [p21.seg_features(p39.symmetric_signals(t, r, dd, GAIN)) for t, r in T]
            aucs.append(p43.fair_exam(list(H_units), S))
        m, sd_ = float(np.mean(aucs)), float(np.std(aucs))
        print(f"[{tag}] {[round(a,3) for a in aucs]} -> {m:.3f}±{sd_:.3f}", flush=True)
        return dict(reps=[float(a) for a in aucs], mean=m, std=sd_)

    out = {}
    # ① 홈 52스택 (τ0.3 σ0.164 β1)
    home_roads = [r for r, m in zip(r24, te24) if m]
    Hh = []
    for r in home_roads:
        for c in p21.chunk_signals(p20.human_signals(r, dd24)):
            Hh.append(p21.seg_features(c))
    out["home_52stack"] = reps(home_roads, dd24, 4,
                               lambda sd_: p52.BestStackPolicy(bc, beta=1.0, tau=0.3,
                                                               sigma=0.164, lib=fitlib,
                                                               seed=sd_), Hh, "홈 52스택")
    # ② 남산 제로샷 46d+42 근사 (β2 τ0 σ0.142, full-116)
    rN, _, ddN = load_roads(os.path.join(CACHE, "env_roads_namsan.npz"))
    rN = trim_roads(rN)
    HN = [p21.seg_features(p20.human_signals(r, ddN)) for r in rN]
    out["namsan_zeroshot"] = reps(rN, ddN, 2,
                                  lambda sd_: p52.BestStackPolicy(bc, beta=2.0, tau=0.0,
                                                                  sigma=0.142, lib=fitlib,
                                                                  seed=sd_), HN,
                                  "남산 제로샷")
    json.dump(out, open(os.path.join(REP, "replicate_records.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved replicate_records.json", flush=True)


if __name__ == "__main__":
    main()
