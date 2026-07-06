# -*- coding: utf-8 -*-
"""
39_discriminator_audit.py  --  AUC<0.55 캠페인 0단계: 시험관 감사.

목표 0.55를 좇기 전에 시험이 공정한지 확정한다. 혐의 3건:
  A) 채널 산출 비대칭: 사람 신호는 e의 미분으로 ψ/θ를 유도(human_signals),
     가상 신호는 시뮬 기록값 사용(rl_signals) — 측정 방법 차이가 지문으로 둔갑?
     → 가상도 사람과 동일하게 실현 e에서 유도한 "대칭 신호" 버전 비교.
  B) 판별기 강도: 선형 로지스틱 vs 비선형 GBM.
  C) 폴드 누수: 구간 단위 StratifiedKFold(현행) vs 유닛 단위 GroupKFold.
2×2×2 표로 공정 시험을 확정하고 캠페인 목표지표를 정의한다.

  python 39_discriminator_audit.py
"""
import os, json, importlib
import numpy as np

from common import REP, CACHE, gen_split
from driving_env import DrivingEnv, load_roads, rollout, trim_roads

p20 = importlib.import_module("20_profile_eval")
p21 = importlib.import_module("21_validation")
p22 = importlib.import_module("22_v3_spectral")
p34 = importlib.import_module("34_virtual_cohort")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score

np.random.seed(0)
GAIN, BASE = 0.012, "rl_multi.zip"
GRID = p20.GRID


def symmetric_signals(traj, road, dd, gain):
    """가상 궤적을 '사람과 동일한 방법'으로 신호화: 실현 e를 평활→미분 유도."""
    s0 = traj[:, 0]
    g = np.arange(s0[0], s0[-1], GRID)
    e = np.interp(g, s0, traj[:, 1])
    v = np.interp(g, s0, traj[:, 2])
    from driving_env import _smooth
    e = _smooth(np.asarray(e, np.float64))
    psi = np.gradient(e, GRID)
    curv_road = np.interp(g, np.arange(len(road["curv"])) * dd, road["curv"])
    kap = curv_road + np.gradient(psi, GRID)
    return dict(s=g, e=e, psi=psi, latv=v * psi, kappa=kap, lata=v * v * kap,
                theta=kap * p20.K2DEG)


def auc_variants(XH_units, XS_units):
    """유닛별 특징행렬 목록 → (분류기 × CV) 4조합 AUC."""
    X, y, grp = [], [], []
    gid = 0
    for cls, units in [(0, XH_units), (1, XS_units)]:
        for F in units:
            X.append(F); y += [cls] * len(F); grp += [gid] * len(F)
            gid += 1
    X = np.vstack(X); y = np.array(y); grp = np.array(grp)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    out = {}
    for cname, mk in [("logistic", lambda: LogisticRegression(max_iter=1000)),
                      ("gbm", lambda: HistGradientBoostingClassifier(max_iter=200,
                                                                     random_state=0))]:
        for cvname, splits in [("segCV", StratifiedKFold(5, shuffle=True, random_state=0)
                                .split(X, y)),
                               ("unitCV", GroupKFold(5).split(X, y, grp))]:
            ps = np.zeros(len(y))
            for tri, tei in splits:
                clf = mk(); clf.fit(X[tri], y[tri])
                ps[tei] = clf.predict_proba(X[tei])[:, 1]
            out[f"{cname}_{cvname}"] = float(roc_auc_score(y, ps))
    return out


def main():
    roads, _, dd = load_roads(os.path.join(CACHE, "env_roads_namsan.npz"))
    roads = trim_roads(roads)
    ch = p34.load_champion(base=BASE)
    cal = json.load(open(os.path.join(REP, "v3_library_2024_multi.json"), encoding="utf-8"))
    sigma = float(cal["sigma"])

    # 가상 롤아웃 수집 (제로샷 프로토콜, per_road 2)
    blind = [dict(r, e_ref=np.zeros_like(r["v_ref"])) for r in roads]
    env = DrivingEnv(blind, dd=dd, record=True, steer_gain=GAIN)
    trajs = []
    for k in range(len(blind)):
        for j in range(2):
            pol = p22.SpectralPolicy(ch["model"], ch["fr"], ch["A"], sigma,
                                     lib=ch["lib"], seed=7000 + k * 20 + j)
            pol.reset()
            traj, _ = rollout(env, pol, k)
            if len(traj) > 60:
                trajs.append((traj, roads[k]))
        if (k + 1) % 40 == 0:
            print(f"  rollout {k+1}/{len(blind)}", flush=True)

    H_units = [p21.seg_features(p20.human_signals(r, dd)) for r in roads]
    S_asis = [p21.seg_features(p20.rl_signals(t, gain=GAIN)) for t, r in trajs]
    S_symm = [p21.seg_features(symmetric_signals(t, r, dd, GAIN)) for t, r in trajs]

    # 밸런스 (유닛 수 맞춤)
    rngb = np.random.RandomState(1)
    nu = min(len(H_units), len(S_asis))
    idxH = rngb.choice(len(H_units), nu, replace=False)
    idxS = rngb.choice(len(S_asis), nu, replace=False)
    H = [H_units[i] for i in idxH]

    res = {}
    for tag, S_all in [("asis", S_asis), ("symmetric", S_symm)]:
        S = [S_all[i] for i in idxS]
        res[tag] = auc_variants(H, S)
        print(f"[{tag}] " + " | ".join(f"{k}={v:.3f}" for k, v in res[tag].items()),
              flush=True)

    # 셔플 대조 (유닛 라벨 셔플, 공정판=symmetric+gbm+unitCV)
    rs = np.random.RandomState(7)
    S = [S_symm[i] for i in idxS]
    order = rs.permutation(2 * nu)
    mixed = H + S
    Hs = [mixed[i] for i in order[:nu]]
    Ss = [mixed[i] for i in order[nu:]]
    null = auc_variants(Hs, Ss)
    res["shuffle_null"] = null
    print("[shuffle] " + " | ".join(f"{k}={v:.3f}" for k, v in null.items()), flush=True)

    json.dump(res, open(os.path.join(REP, "discriminator_audit.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved discriminator_audit.json", flush=True)


if __name__ == "__main__":
    main()
