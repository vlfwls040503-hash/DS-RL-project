# -*- coding: utf-8 -*-
"""
53_knob_opt.py  --  트랙 B(합법판): (σ, τ, β) 노브를 val GBM AUC에 직접 최적화.

대리 모멘트가 아니라 시험 점수 자체를 목적함수로 좌표하강 탐색(블랙박스 —
판별자 기울기 미사용 = 적대 아님). val 과적합 방지: 최종 판정은 피험자-분리 test.

  python 53_knob_opt.py
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

    out = {}
    for exp, per_road, start in [("2024", 4, dict(sigma=0.164, tau=0.3, beta=1.0)),
                                 ("namsan", 2, dict(sigma=0.142, tau=0.3, beta=2.0))]:
        roads, _, dd = load_roads(os.path.join(CACHE, f"env_roads_{exp}.npz"))
        roads = trim_roads(roads)
        sx = np.array([r["subject"] for r in roads], "int64")
        trx, vax, tex = gen_split(sx, seed=0)
        val_r = [r for r, m in zip(roads, vax) if m]
        val_b = [dict(r, e_ref=np.zeros_like(r["v_ref"])) for r in val_r]
        env_v = DrivingEnv(val_b, dd=dd, record=True, steer_gain=GAIN)
        if exp == "2024":
            Hval = []
            for r in val_r:
                for c in p21.chunk_signals(p20.human_signals(r, dd)):
                    Hval.append(p21.seg_features(c))
        else:
            Hval = [p21.seg_features(p20.human_signals(r, dd)) for r in val_r]

        def val_auc(k):
            S = []
            for kk in range(len(val_r)):
                for j in range(2):
                    pol = p52.BestStackPolicy(bc, beta=k["beta"], tau=k["tau"],
                                              sigma=k["sigma"], lib=fitlib,
                                              seed=90 + kk * 7 + j)
                    pol.reset()
                    traj, _ = rollout(env_v, pol, kk)
                    if len(traj) > 60:
                        S.append(p21.seg_features(
                            p39.symmetric_signals(traj, val_r[kk], dd, GAIN)))
            return p43.fair_exam(list(Hval), S)

        cur = dict(start)
        cur_a = val_auc(cur)
        print(f"[{exp}] 시작 {cur} val AUC={cur_a:.3f}", flush=True)
        grids = dict(sigma=[0.7, 1.0, 1.4], tau=[0.5, 1.0, 1.7], beta=[0.7, 1.0, 1.5])
        for sweep in range(2):
            for kname, mults in grids.items():
                for m in mults:
                    if m == 1.0:
                        continue
                    cand = dict(cur)
                    cand[kname] = float(np.clip(cur[kname] * m, 0.02, 4.0))
                    a = val_auc(cand)
                    print(f"    {kname}={cand[kname]:.3f}: val {a:.3f}", flush=True)
                    if a < cur_a:
                        cur, cur_a = cand, a
        print(f"[{exp}] 최적 {cur} val AUC={cur_a:.3f}", flush=True)

        # ---- 피험자-분리 test 판정 ----
        exam_roads = roads if exp == "namsan" else [r for r, m in zip(roads, tex) if m]
        blind = [dict(r, e_ref=np.zeros_like(r["v_ref"])) for r in exam_roads]
        env = DrivingEnv(blind, dd=dd, record=True, steer_gain=GAIN)
        T, offc = [], 0
        for kk in range(len(blind)):
            for j in range(per_road):
                pol = p52.BestStackPolicy(bc, beta=cur["beta"], tau=cur["tau"],
                                          sigma=cur["sigma"], lib=fitlib,
                                          seed=7000 + kk * 20 + j)
                pol.reset()
                traj, off = rollout(env, pol, kk)
                offc += int(off)
                if len(traj) > 60:
                    T.append((traj, exam_roads[kk]))
        S = [p21.seg_features(p39.symmetric_signals(t, r, dd, GAIN)) for t, r in T]
        if exp == "2024":
            Hh = []
            for r in exam_roads:
                for c in p21.chunk_signals(p20.human_signals(r, dd)):
                    Hh.append(p21.seg_features(c))
        else:
            Hh = [p21.seg_features(p20.human_signals(r, dd)) for r in exam_roads]
        auc = p43.fair_exam(Hh, S)
        out[exp] = dict(knobs=cur, val_auc=float(cur_a), test_auc=float(auc),
                        off=offc / max(len(blind) * per_road, 1))
        print(f"[{exp}] TEST AUC={auc:.3f} (현직 {'0.817' if exp=='namsan' else '0.760'}) "
              f"off={out[exp]['off']:.2f}", flush=True)
    json.dump(out, open(os.path.join(REP, "knob_opt.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved knob_opt.json", flush=True)


if __name__ == "__main__":
    main()
