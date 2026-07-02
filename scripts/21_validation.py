# -*- coding: utf-8 -*-
"""
21_validation.py  --  정량 검증 스위트: "사람 vs 합성이 비슷하다"를 3층으로 검정.

  ① C2ST(기계 판별자): 200m 구간 특징으로 사람/합성 분류기(로지스틱) 학습,
     5-fold CV AUC + 라벨순열 p.  AUC≈0.5 = 구별불가(가장 강한 단일 증거).
  ② 동등성(TOST식): 도로단위 지표(SDLP·LPM·SRR·주파장) 평균차의 부트스트랩 95% CI가
     사전선언 마진(±0.5×사람SD) 안이면 '동등'.  (차이검정 순열 p도 참고로 병기.)
  ③ 조건효과 보존: 지상/지하 SDLP 차이의 Cohen's d — 사람 vs 합성.

  python 21_validation.py --exp 2024 --per_road 3
"""
import os, json, argparse
import numpy as np
import torch
import importlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from common import ART, REP, CACHE, gen_split
from driving_env import DrivingEnv, load_roads, rollout, trim_roads

p20 = importlib.import_module("20_profile_eval")

for fam in ["Malgun Gothic", "NanumGothic", "Gulim"]:
    try:
        plt.rcParams["font.family"] = fam; break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(REP, "figs"); os.makedirs(FIG, exist_ok=True)
np.random.seed(0); torch.manual_seed(0)

SEG_PTS = 20            # 200 m at GRID=10
CHUNK_PTS = 520         # 5.2 km — 사람 도로를 합성 롤아웃과 같은 길이 단위로 청크
                        # (길이 불일치는 특히 주파장 추정을 불공정하게 만듦 + n 확보)
GRID = p20.GRID


def chunk_signals(sig, pts=CHUNK_PTS, min_pts=400):
    out = []
    n = len(sig["e"])
    for a in range(0, n, pts):
        b = min(a + pts, n)
        if b - a >= min_pts:
            out.append({k: (v[a:b] if isinstance(v, np.ndarray) else v) for k, v in sig.items()})
    return out


# ---------------- segment features (판별자 입력) ----------------
def seg_features(sig):
    """per-200m-segment features from a signal dict (human_signals/rl_signals)."""
    F = []
    n = len(sig["e"])
    for a in range(0, n - SEG_PTS, SEG_PTS):
        sl = slice(a, a + SEG_PTS)
        e, latv, lata, th = sig["e"][sl], sig["latv"][sl], sig["lata"][sl], sig["theta"][sl]
        de = np.diff(e)
        F.append([np.std(e), np.mean(np.abs(de)), np.std(latv), np.std(lata),
                  p20.srr(th, 0.5), p20.srr(th, 2.0),
                  float(np.sqrt(np.mean(de ** 2))), np.max(np.abs(latv))])
    return np.array(F, dtype="float64")


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                 / max(len(a) + len(b) - 2, 1))
    return float((a.mean() - b.mean()) / max(sp, 1e-12))


def perm_pvalue(a, b, n_perm=10000, rng=None):
    """two-sided permutation test on mean difference."""
    rng = rng or np.random.RandomState(0)
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs = abs(a.mean() - b.mean())
    z = np.concatenate([a, b]); na = len(a); cnt = 0
    for _ in range(n_perm):
        rng.shuffle(z)
        if abs(z[:na].mean() - z[na:].mean()) >= obs:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def boot_ci_diff(a, b, n_boot=10000, rng=None):
    """bootstrap 95% CI of mean(a)-mean(b)."""
    rng = rng or np.random.RandomState(1)
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = [np.mean(rng.choice(a, len(a))) - np.mean(rng.choice(b, len(b))) for _ in range(n_boot)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="2024")
    ap.add_argument("--per_road", type=int, default=3)
    args = ap.parse_args()
    exp = args.exp

    cal = json.load(open(os.path.join(REP, f"profile_eval_{exp}.json"), encoding="utf-8"))
    roads, subject, dd = load_roads(os.path.join(CACHE, f"env_roads_{exp}.npz"))
    roads = trim_roads(roads)
    subject = np.array([r["subject"] for r in roads], "int64")
    tr, va, te = gen_split(subject, seed=0)
    fit_roads = [r for r, m in zip(roads, tr | va) if m]
    test_roads = [r for r, m in zip(roads, te) if m]
    model = PPO.load(os.path.join(ART, f"rl_{exp}.zip"), device="cpu")
    print(f"[{exp}] driver: e_tau={cal['e_tau']:.0f} e_sigma={cal['e_sigma']:.3f} "
          f"e_lpf={cal['e_lpf']:.0f}  test roads={len(test_roads)}", flush=True)

    # trait pool (동일 부트스트랩)
    pool = [dict(sdlp=float(np.std(r["e_ref"])), lpm=float(np.mean(r["e_ref"])),
                 v=float(np.mean(r["v_ref"])), cond=int(r.get("cond", 0))) for r in fit_roads]
    sdlp_pool = float(np.mean([t["sdlp"] for t in pool]))
    v_pool = float(np.mean([t["v"] for t in pool]))
    by_cond = {}
    for t in pool:
        by_cond.setdefault(t["cond"], []).append(t)
    rng = np.random.RandomState(0)

    # ---- rollouts (b=0 초기화 수정 반영) + 신호 수집 ----
    env = DrivingEnv(test_roads, dd=dd, record=True)
    H_sig, H_units, h_unit_cond, S_sig, s_meta = [], [], [], [], []
    for k, road in enumerate(test_roads):
        hs = p20.human_signals(road, dd)
        H_sig.append(hs)
        for ch in chunk_signals(hs):                 # 사람도 5.2km 단위(합성과 동일 길이)
            H_units.append(ch); h_unit_cond.append(int(road.get("cond", 0)))
        cand = by_cond.get(int(road.get("cond", 0)), pool)
        for j in range(args.per_road):
            t = cand[rng.randint(len(cand))]
            pol = p20.HumanlikePolicy(model, e_tau=cal["e_tau"], e_lpf=cal["e_lpf"],
                                      e_sigma=float(np.clip(cal["e_sigma"] * t["sdlp"] / sdlp_pool, 0.03, 1.2)),
                                      b_bias=t["lpm"], v_scale=t["v"] / v_pool,
                                      seed=2000 + k * 10 + j)
            pol.reset()
            traj, off = rollout(env, pol, k)
            if len(traj) > 60:
                S_sig.append(p20.rl_signals(traj))
                s_meta.append(dict(road=k, cond=int(road.get("cond", 0)), off=bool(off)))
        print(f"  road {k+1}/{len(test_roads)}", flush=True)
    off_rate = float(np.mean([m["off"] for m in s_meta]))

    # ================= ① C2ST 기계 판별자 =================
    XH = np.vstack([seg_features(h) for h in H_sig])
    XS = np.vstack([seg_features(s) for s in S_sig])
    # 사람 구간이 훨씬 많음(33km vs 5.6km) → 균형 서브샘플
    rng2 = np.random.RandomState(2)
    nmin = min(len(XH), len(XS))
    XH_s = XH[rng2.choice(len(XH), nmin, replace=False)]
    XS_s = XS[rng2.choice(len(XS), nmin, replace=False)]
    X = np.vstack([XH_s, XS_s])
    y = np.concatenate([np.zeros(nmin), np.ones(nmin)])
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)

    def cv_auc(Xa, ya, seed=0):
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        ps = np.zeros(len(ya))
        for tr_i, te_i in skf.split(Xa, ya):
            clf = LogisticRegression(max_iter=1000)
            clf.fit(Xa[tr_i], ya[tr_i])
            ps[te_i] = clf.predict_proba(Xa[te_i])[:, 1]
        return float(roc_auc_score(ya, ps))

    auc = cv_auc(X, y)
    null_aucs = []
    for i in range(50):                      # 라벨순열 null 분포
        yp = y.copy(); np.random.RandomState(100 + i).shuffle(yp)
        null_aucs.append(cv_auc(X, yp, seed=i))
    p_auc = float((np.sum(np.array(null_aucs) >= auc) + 1) / (len(null_aucs) + 1))
    print(f"C2ST: AUC={auc:.3f} (null {np.mean(null_aucs):.3f}±{np.std(null_aucs):.3f}, p={p_auc:.3f}) "
          f"n={nmin}+{nmin} segments", flush=True)

    # ================= ② 동등성 (도로/주행 단위 지표) =================
    def road_metrics(sig):
        return dict(sdlp=float(np.std(sig["e"])), lpm=float(np.mean(sig["e"])),
                    srr=float(p20.srr(sig["theta"], 0.5)),
                    wl=float(p20.wavelength(sig["e"])))
    MH = [road_metrics(h) for h in H_units]          # 사람: 5.2km 청크 단위 (n 확보 + 길이 공정)
    MS = [road_metrics(s) for s in S_sig]
    equiv = {}
    for m in ["sdlp", "lpm", "srr", "wl"]:
        a = np.array([x[m] for x in MS]); b = np.array([x[m] for x in MH])
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        margin = 0.5 * float(b.std(ddof=1))
        lo, hi = boot_ci_diff(a, b)
        p_diff = perm_pvalue(a, b, 5000)
        equiv[m] = dict(mean_h=float(b.mean()), mean_s=float(a.mean()),
                        diff=float(a.mean() - b.mean()), ci=[lo, hi], margin=margin,
                        equivalent=bool(lo > -margin and hi < margin), p_diff=p_diff)
        print(f"  {m:5s} diff={equiv[m]['diff']:+.3f} CI[{lo:+.3f},{hi:+.3f}] "
              f"margin=±{margin:.3f} -> {'동등' if equiv[m]['equivalent'] else '동등 아님'} "
              f"(차이검정 p={p_diff:.3f})", flush=True)

    # ================= ③ 조건효과 보존 (지상/지하 SDLP) =================
    hc = {c: [float(np.std(h["e"])) for h, cc in zip(H_units, h_unit_cond) if cc == c]
          for c in (0, 1)}
    sc = {c: [np.std(s["e"]) for s, m in zip(S_sig, s_meta) if m["cond"] == c] for c in (0, 1)}
    d_h = cohens_d(hc[1], hc[0]) if hc[0] and hc[1] else float("nan")
    d_s = cohens_d(sc[1], sc[0]) if sc[0] and sc[1] else float("nan")
    print(f"조건효과(지하-지상 SDLP): 사람 d={d_h:+.2f}  합성 d={d_s:+.2f}", flush=True)

    # ================= figures =================
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    ax.hist(null_aucs, bins=15, alpha=.6, color="#888780", label="순열 null AUC")
    ax.axvline(auc, color="#7F77DD", lw=2, label=f"실제 AUC={auc:.3f}")
    ax.axvline(0.5, color="#185FA5", ls=":", lw=1.5, label="구별불가(0.5)")
    ax.set_xlabel("AUC"); ax.set_title(f"① C2ST 판별자 (p={p_auc:.3f})"); ax.legend(fontsize=8)
    ax = axes[1]
    names = ["SDLP", "LPM", "SRR", "주파장"]
    keys = ["sdlp", "lpm", "srr", "wl"]
    for i, (nm, k) in enumerate(zip(names, keys)):
        e = equiv[k]; mrg = e["margin"]
        no = (e["diff"]) / mrg if mrg > 0 else 0  # normalized
        lo, hi = e["ci"][0] / mrg, e["ci"][1] / mrg
        col = "#1D9E75" if e["equivalent"] else "#D85A30"
        ax.plot([lo, hi], [i, i], color=col, lw=3)
        ax.plot([no], [i], "o", color=col)
    ax.axvline(-1, ls="--", color="#888780"); ax.axvline(1, ls="--", color="#888780")
    ax.axvline(0, ls=":", color="#B4B2A9")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("평균차 / 동등성 마진 (±1 안이면 동등)")
    ax.set_title("② 동등성 검정 (부트스트랩 95% CI)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"fig_val_{exp}.png"), dpi=120); plt.close(fig)

    # ================= save + report =================
    out = dict(exp=exp, off_rate=off_rate, n_synth=len(S_sig),
               c2st=dict(auc=auc, null_mean=float(np.mean(null_aucs)),
                         null_std=float(np.std(null_aucs)), p=p_auc, n_seg=int(nmin)),
               equivalence=equiv, condition_effect=dict(d_human=d_h, d_synth=d_s))
    json.dump(out, open(os.path.join(REP, f"validation_{exp}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    L = ["\n\n---\n\n## 9. 정량 검증 (정성 판단 → 통계 검정)\n",
         f"합성 주행 {len(S_sig)}회(이탈율 {off_rate:.2f}) vs 사람 test 도로 {len(test_roads)}개 "
         f"(**5.2km 청크 {len(H_units)}단위** — 합성 롤아웃과 동일 길이로 공정 비교 + 표본 확보).\n",
         f"### ① 기계 판별자 C2ST — \"분류기도 못 가르나\"\n",
         f"- 200m 구간 특징 8종, 로지스틱 5-fold CV: **AUC = {auc:.3f}** "
         f"(순열 null {np.mean(null_aucs):.3f}±{np.std(null_aucs):.3f}, p={p_auc:.3f}, "
         f"구간 {nmin}+{nmin})",
         "- AUC 0.5=구별불가, 1.0=완전구별. p<0.05면 '구별 가능하다'는 유의한 증거.\n",
         "### ② 동등성 검정 (마진 ±0.5×사람SD, 사전선언)\n",
         "| 지표 | 사람 | 합성 | 평균차 [95% CI] | 마진 | 판정 | 차이검정 p |",
         "|---|---|---|---|---|---|---|"]
    for nm, k in zip(names, keys):
        e = equiv[k]
        L.append(f"| {nm} | {e['mean_h']:.3f} | {e['mean_s']:.3f} | "
                 f"{e['diff']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}] | ±{e['margin']:.3f} | "
                 f"{'**동등**' if e['equivalent'] else '동등 아님'} | {e['p_diff']:.3f} |")
    L += [f"\n### ③ 조건효과 보존 (지하−지상 SDLP, Cohen's d)\n",
          f"- 사람 d = {d_h:+.2f}, 합성 d = {d_s:+.2f} → 방향 "
          f"{'일치' if np.sign(d_h) == np.sign(d_s) else '불일치'}\n",
          f"![검증](figs/fig_val_{exp}.png)\n",
          "**주의**: 도로 표본 n이 작아(test 10개) 동등성 검정력은 제한적 — CI가 마진을 벗어나면 "
          "'동등 입증 실패'이지 '다름 입증'이 아님. C2ST는 구간 단위라 표본이 커서 가장 민감한 검정.\n"]
    open(os.path.join(REP, "report_rl.md"), "a", encoding="utf-8").write("\n".join(L))
    print("wrote section 9 + fig + validation json", flush=True)


if __name__ == "__main__":
    main()
