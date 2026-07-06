# -*- coding: utf-8 -*-
"""
38_fingerprint.py  --  남산 v3.3 잔여 지문 해부 + 청크 신축(stretch) 노브.

가설: 남산 잔차(파장 905↔773, SRR 17.8↔11.9)는 학습 문제가 아니라 **청크 사투리**
— 주입하는 사람 청크가 2024(고속, 파장 940) 출신이라 리듬이 느림. 거리축 신축으로
목표 도메인 리듬에 맞출 수 있는 구조적 노브가 존재.

  1) 진단: 판별기(로지스틱)의 특징별 기여 + 단독 AUC 순위
  2) 신축 노브: 청크 w를 거리축으로 ×stretch 리샘플, val에서 (stretch, σ) 2노브 보정
  3) 판정: te C2ST가 0.656보다 내려가는가

  python 38_fingerprint.py
"""
import os, json, argparse, importlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from common import REP, CACHE, gen_split
from driving_env import DrivingEnv, load_roads, rollout, trim_roads

p20 = importlib.import_module("20_profile_eval")
p21 = importlib.import_module("21_validation")
p22 = importlib.import_module("22_v3_spectral")
p24 = importlib.import_module("24_gail_seg")
p34 = importlib.import_module("34_virtual_cohort")

for fam in ["Malgun Gothic", "NanumGothic", "Gulim"]:
    try:
        plt.rcParams["font.family"] = fam; break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(REP, "figs"); os.makedirs(FIG, exist_ok=True)
np.random.seed(0)
GAIN, BASE = 0.012, "rl_multi.zip"
FEATS = ["std_e", "mean|Δe|", "std_횡속", "std_횡가", "SRR0.5", "SRR2", "RMSΔe", "max|횡속|"]


class StretchPolicy(p22.SpectralPolicy):
    """청크 목표 w의 거리축 신축: stretch<1 = 파장 축소(리듬 촘촘)."""
    def __init__(self, *a, stretch=1.0, **kw):
        super().__init__(*a, **kw)
        self.stretch = float(stretch)

    def reset(self):
        super().reset()
        if self.stretch != 1.0:
            n = len(self.w)
            xi = np.arange(int(n * 1.0)) * self.stretch
            xi = xi[xi < n - 1]
            self.w = np.interp(xi, np.arange(n), self.w)
            # 재정규화 (신축이 std 살짝 바꿈)
            self.w = self.w / (self.w.std() + 1e-12)


def collect(ch, roads_sub, dd, sigma, stretch, per_road, seed0):
    env = DrivingEnv([dict(r, e_ref=np.zeros_like(r["v_ref"])) for r in roads_sub],
                     dd=dd, record=True, steer_gain=GAIN)
    S = []
    for k in range(len(roads_sub)):
        for j in range(per_road):
            pol = StretchPolicy(ch["model"], ch["fr"], ch["A"], sigma,
                                lib=ch["lib"], stretch=stretch, seed=seed0 + k * 20 + j)
            pol.reset()
            traj, _ = rollout(env, pol, k)
            if len(traj) > 60:
                S.append(p20.rl_signals(traj, gain=GAIN))
    return S


def c2st_full(H_sigs, S_sigs, ret_clf=False):
    XH = np.vstack([p21.seg_features(h) for h in H_sigs])
    XS = np.vstack([p21.seg_features(s) for s in S_sigs])
    rng = np.random.RandomState(2)
    nmin = min(len(XH), len(XS))
    X = np.vstack([XH[rng.choice(len(XH), nmin, replace=False)],
                   XS[rng.choice(len(XS), nmin, replace=False)]])
    y = np.concatenate([np.zeros(nmin), np.ones(nmin)])
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    auc = p24.cv_auc(Xs, y)
    if not ret_clf:
        return auc
    clf = LogisticRegression(max_iter=1000).fit(Xs, y)
    singles = []
    for i in range(Xs.shape[1]):
        singles.append(p24.cv_auc(Xs[:, [i]], y))
    return auc, clf.coef_[0], singles


def tex(sigs):
    return dict(sdlp=float(np.mean([np.std(s["e"]) for s in sigs])),
                wl=float(np.nanmean([p20.wavelength(s["e"]) for s in sigs])),
                srr=float(np.mean([p20.srr(s["theta"], 0.5) for s in sigs])))


def main():
    roads, _, dd = load_roads(os.path.join(CACHE, "env_roads_namsan.npz"))
    roads = trim_roads(roads)
    subj = np.array([r["subject"] for r in roads], "int64")
    tr, va, te = gen_split(subj, seed=0)
    val_roads = [r for r, m in zip(roads, va) if m]
    te_roads = [r for r, m in zip(roads, te | tr) if m]     # 진단은 넓게(비교일관 위해 아래 동일)
    ch = p34.load_champion(base=BASE)
    cal = json.load(open(os.path.join(REP, "v3_library_2024_multi.json"), encoding="utf-8"))
    sigma0 = float(cal["sigma"])

    H_all = [p20.human_signals(r, dd) for r in te_roads]
    hv = [p20.human_signals(r, dd) for r in val_roads]
    wl_h_val = float(np.nanmean([p20.wavelength(h["e"]) for h in hv]))
    std_h_val = float(np.mean([np.std(h["e"]) for h in hv]))

    # ---- 1) 진단: 현 v3.3(stretch=1)의 특징별 지문 ----
    S0 = collect(ch, te_roads, dd, sigma0, 1.0, per_road=2, seed0=5000)
    auc0, coefs, singles = c2st_full(H_all, S0, ret_clf=True)
    t0 = tex(S0); th = tex(H_all)
    print(f"[진단] 기준 AUC={auc0:.3f} | 사람 wl {th['wl']:.0f} SRR {th['srr']:.1f} "
          f"/ 가상 wl {t0['wl']:.0f} SRR {t0['srr']:.1f}", flush=True)
    order = np.argsort(-np.abs(coefs))
    for i in order:
        print(f"  {FEATS[i]:10s} 계수 {coefs[i]:+.2f} | 단독 AUC {singles[i]:.3f}", flush=True)

    # ---- 2) (stretch, σ) 2노브 val 보정 ----
    best = (1.0, sigma0, 1e9)
    for st in [0.70, 0.82, 1.0]:
        Sv = collect(ch, val_roads, dd, sigma0, st, per_road=2, seed0=41)
        tv = tex(Sv)
        s_adj = float(np.clip(sigma0 * (std_h_val / max(tv["sdlp"], 1e-6)), 0.02, 1.2))
        Sv2 = collect(ch, val_roads, dd, s_adj, st, per_road=2, seed0=43)
        tv2 = tex(Sv2)
        gap = abs(tv2["wl"] - wl_h_val) / wl_h_val + abs(tv2["sdlp"] - std_h_val) / std_h_val
        print(f"  [val] stretch={st}: wl {tv2['wl']:.0f}(목표 {wl_h_val:.0f}) "
              f"SDLP {tv2['sdlp']:.3f}(목표 {std_h_val:.3f}) SRR {tv2['srr']:.1f} gap={gap:.3f}",
              flush=True)
        if gap < best[2]:
            best = (st, s_adj, gap)
    st_b, sig_b, _ = best
    print(f"선택: stretch={st_b} sigma={sig_b:.3f}", flush=True)

    # ---- 3) 판정: te AUC ----
    S1 = collect(ch, te_roads, dd, sig_b, st_b, per_road=2, seed0=5000)
    auc1 = c2st_full(H_all, S1)
    t1 = tex(S1)
    print(f"[판정] AUC {auc0:.3f} -> {auc1:.3f} | wl {t0['wl']:.0f}->{t1['wl']:.0f} "
          f"(사람 {th['wl']:.0f}) SRR {t0['srr']:.1f}->{t1['srr']:.1f} ({th['srr']:.1f}) "
          f"SDLP {t1['sdlp']:.3f} ({th['sdlp']:.3f})", flush=True)

    json.dump(dict(auc_before=float(auc0), auc_after=float(auc1),
                   stretch=st_b, sigma=sig_b,
                   coefs=dict(zip(FEATS, map(float, coefs))),
                   singles=dict(zip(FEATS, map(float, singles))),
                   tex_h=th, tex_before=t0, tex_after=t1),
              open(os.path.join(REP, "fingerprint_namsan.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved fingerprint_namsan.json", flush=True)


if __name__ == "__main__":
    main()
