# -*- coding: utf-8 -*-
"""
43_gbm_anatomy.py  --  GBM 0.869 부동의 해부: ①2024 안방 대조 ②GBM 중요도.

갈림길 진단: 공정시험(대칭+GBM+유닛CV)을 2024 안방(청크·도메인 일치)에서 측정.
  - 2024도 ~0.86 → 지문 = 추종 루프 서명 (청크 출처 무관) → 정책 레벨 수술
  - 2024 « 남산 → 도메인 불일치 → 원산지 청크(지하유출입·결빙) 투입
GBM permutation importance로 어떤 특징(결합)을 쓰는지 특정.

  python 43_gbm_anatomy.py
"""
import os, json, importlib
import numpy as np

from common import REP, CACHE, gen_split
from driving_env import DrivingEnv, load_roads, rollout, trim_roads

p20 = importlib.import_module("20_profile_eval")
p21 = importlib.import_module("21_validation")
p22 = importlib.import_module("22_v3_spectral")
p34 = importlib.import_module("34_virtual_cohort")
p39 = importlib.import_module("39_discriminator_audit")

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

np.random.seed(0)
GAIN, BASE = 0.012, "rl_multi.zip"
FEATS = ["std_e", "mean|de|", "std_latv", "std_lata", "SRR0.5", "SRR2", "RMSde", "max|latv|"]


def fair_exam(H_units, S_units, ret_imp=False):
    H_units = [F for F in H_units if len(F) > 0]     # 200m 미만 유닛 제거 후
    S_units = [F for F in S_units if len(F) > 0]     # 유닛 수 재균형 (퇴화 방지)
    nu = min(len(H_units), len(S_units))
    rng = np.random.RandomState(11)
    H_units = [H_units[i] for i in rng.choice(len(H_units), nu, replace=False)]
    S_units = [S_units[i] for i in rng.choice(len(S_units), nu, replace=False)]
    X, y, grp = [], [], []
    gid = 0
    for cls, units in [(0, H_units), (1, S_units)]:
        for F in units:
            X.append(F); y += [cls] * len(F); grp += [gid] * len(F)
            gid += 1
    X = np.vstack(X); y = np.array(y); grp = np.array(grp)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    ps = np.zeros(len(y))
    for tri, tei in GroupKFold(5).split(X, y, grp):
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=0)
        clf.fit(X[tri], y[tri])
        ps[tei] = clf.predict_proba(X[tei])[:, 1]
    auc = float(roc_auc_score(y, ps))
    if not ret_imp:
        return auc
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(X, y)
    imp = permutation_importance(clf, X, y, n_repeats=5, random_state=0,
                                 scoring="roc_auc")
    return auc, imp.importances_mean


def cohort(ch, roads_sub, dd, sigma, per_road, seed0):
    env = DrivingEnv([dict(r, e_ref=np.zeros_like(r["v_ref"])) for r in roads_sub],
                     dd=dd, record=True, steer_gain=GAIN)
    T = []
    for k in range(len(roads_sub)):
        for j in range(per_road):
            pol = p22.SpectralPolicy(ch["model"], ch["fr"], ch["A"], sigma,
                                     lib=ch["lib"], seed=seed0 + k * 20 + j)
            pol.reset()
            traj, _ = rollout(env, pol, k)
            if len(traj) > 60:
                T.append((traj, roads_sub[k]))
    return T


def main():
    ch = p34.load_champion(base=BASE)     # 청크=2024산 (안방 대조에 정합)
    cal = json.load(open(os.path.join(REP, "v3_library_2024_multi.json"), encoding="utf-8"))
    sigma = float(cal["sigma"])
    out = {}
    for exp, per_road in [("2024", 4), ("namsan", 2)]:
        roads, _, dd = load_roads(os.path.join(CACHE, f"env_roads_{exp}.npz"))
        roads = trim_roads(roads)
        if exp == "2024":                  # 안방: test 분할만 (v3.1 프로토콜과 정합)
            subj = np.array([r["subject"] for r in roads], "int64")
            _, _, te = gen_split(subj, seed=0)
            roads = [r for r, m in zip(roads, te) if m]
        T = cohort(ch, roads, dd, sigma, per_road, 7000)
        S = [p21.seg_features(p39.symmetric_signals(t, r, dd, GAIN)) for t, r in T]
        if exp == "2024":                  # 사람도 5.2km 청크 단위 (기존 프로토콜)
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
        auc, imp = fair_exam(Hb, Sb, ret_imp=True)
        out[exp] = dict(auc=float(auc),
                        importance=dict(zip(FEATS, map(float, imp))))
        top = np.argsort(-imp)[:4]
        print(f"[{exp}] 공정 GBM AUC={auc:.3f} | 중요도 상위: "
              + ", ".join(f"{FEATS[i]}={imp[i]:.3f}" for i in top), flush=True)
    json.dump(out, open(os.path.join(REP, "gbm_anatomy.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved gbm_anatomy.json", flush=True)


if __name__ == "__main__":
    main()
