# -*- coding: utf-8 -*-
"""
14_eval_gen.py  --  within-experiment validation of the CVAE generator.
Generates trajectories (z~N(0,I) over held-out geometry) and compares the *distribution*
of behavior summaries to held-out reality. Wasserstein/KS, z-space PCA, kinematics, condition.

  python 14_eval_gen.py --smoke
  python 14_eval_gen.py --exp 2024
"""
import os, json, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (ART, REP, CACHE, GEN_DD, build_smoke_dataset, gen_split,
                    wasserstein1d, ks_stat)
from datasets import Scaler
from models import CVAE

DEV = "cuda" if torch.cuda.is_available() else "cpu"
for fam in ["Malgun Gothic", "NanumGothic", "Gulim"]:
    try:
        plt.rcParams["font.family"] = fam; break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(REP, "figs"); os.makedirs(FIG, exist_ok=True)


def load_ckpt(exp):
    ck = torch.load(os.path.join(ART, f"cvae_{exp}.pt"), map_location=DEV, weights_only=False)
    m = CVAE(beh_dim=ck["beh_dim"], geo_dim=ck["geo_dim"], z_dim=ck["z_dim"],
             stochastic=ck.get("stochastic", False), stoch_dim=ck.get("stoch_dim"))
    m.load_state_dict(ck["state"]); m.to(DEV).eval()
    return m, Scaler.from_dict(ck["geo_scaler"]), Scaler.from_dict(ck["beh_scaler"]), ck


def load_gen(exp):
    p = os.path.join(CACHE, f"dataset_gen_{exp}.npz")
    if not os.path.exists(p) and exp == "smoke":
        build_smoke_dataset(p)
    d = np.load(p, allow_pickle=True)
    dd = float(d["dd"]) if "dd" in d else GEN_DD
    return (d["X_geo"].astype("float32"), d["Y_beh"].astype("float32"),
            d["win_subject"].astype("int64"), d["win_run"].astype("int64"),
            d["win_cond"].astype("int64"), dd)


def summaries(beh, dd):
    """beh (N,W,2)=[offset,speed]. Returns dict of per-window arrays + raw accel/jerk."""
    off, spd = beh[:, :, 0], beh[:, :, 1]
    dvdx = np.gradient(spd, dd, axis=1)
    accel = spd * dvdx                       # m/s^2 (a = v dv/dx)
    jerk = spd * np.gradient(accel, dd, axis=1)
    return dict(sdlp=off.std(axis=1), mean_speed=spd.mean(axis=1), speed_std=spd.std(axis=1),
                accel=accel.reshape(-1), jerk=jerk.reshape(-1))


def pca2(X):
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def decode_samples(model, gs, bs, Xg_te, z_dim, chunk=512):
    out = []
    with torch.no_grad():
        for s in range(0, len(Xg_te), chunk):
            geo = torch.from_numpy(gs.transform(Xg_te[s:s + chunk])).to(DEV)
            z = torch.randn(geo.shape[0], z_dim, device=DEV)
            rec = model.decode_sample(z, geo).cpu().numpy()
            out.append(bs.inverse(rec))
    return np.concatenate(out)


def encode_mu(model, gs, bs, Xg_te, Yb_te, chunk=512):
    out = []
    with torch.no_grad():
        for s in range(0, len(Xg_te), chunk):
            geo = torch.from_numpy(gs.transform(Xg_te[s:s + chunk])).to(DEV)
            beh = torch.from_numpy(bs.transform(Yb_te[s:s + chunk])).to(DEV)
            mu, _ = model.encode(beh, geo)
            out.append(mu.cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="smoke")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    exp = "smoke" if args.smoke else args.exp

    model, gs, bs, ck = load_ckpt(exp)
    Xg, Yb, wsub, wrun, wcond, dd = load_gen(exp)
    _, _, te = gen_split(wsub, seed=0)
    Xte, Yte, cond_te = Xg[te], Yb[te], wcond[te]
    print(f"[{exp}] test windows={len(Xte)}  geo_dim={ck['geo_dim']}  z_dim={ck['z_dim']}", flush=True)

    gen = decode_samples(model, gs, bs, Xte, ck["z_dim"])     # (Nte,W,2) original units
    real_s, gen_s = summaries(Yte, dd), summaries(gen, dd)

    # ---- distribution distances ----
    dist = {}
    for k in ["sdlp", "mean_speed", "speed_std"]:
        dist[k] = dict(wasserstein=wasserstein1d(gen_s[k], real_s[k]),
                       ks=ks_stat(gen_s[k], real_s[k]),
                       real_mean=float(np.mean(real_s[k])), gen_mean=float(np.mean(gen_s[k])))

    # ---- kinematic plausibility (vs real range) ----
    def viol(a, thr):
        return float(np.mean(np.abs(a) > thr))
    kin = dict(accel_viol_gen=viol(gen_s["accel"], 4.0), accel_viol_real=viol(real_s["accel"], 4.0),
               jerk_viol_gen=viol(gen_s["jerk"], 5.0), jerk_viol_real=viol(real_s["jerk"], 5.0))

    # ---- z-space ----
    mu = encode_mu(model, gs, bs, Xte, Yte)
    zc = pca2(mu) if mu.shape[1] >= 2 else np.column_stack([mu[:, 0], np.zeros(len(mu))])
    corr_sdlp = float(np.corrcoef(zc[:, 0], real_s["sdlp"])[0, 1])
    corr_spd = float(np.corrcoef(zc[:, 0], real_s["mean_speed"])[0, 1])

    # ---- condition effect (if >1 group in test) ----
    cond_tbl = {}
    ug = sorted(set(cond_te.tolist()))
    if len(ug) > 1:
        for g in ug:
            mg = cond_te == g
            cond_tbl[int(g)] = dict(real_sdlp=float(real_s["sdlp"][mg].mean()),
                                    gen_sdlp=float(gen_s["sdlp"][mg].mean()),
                                    real_speed=float(real_s["mean_speed"][mg].mean()),
                                    gen_speed=float(gen_s["mean_speed"][mg].mean()), n=int(mg.sum()))

    # ================= figures =================
    # 1) z-space
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sc = ax.scatter(zc[:, 0], zc[:, 1], c=real_s["sdlp"], cmap="viridis", s=18)
    plt.colorbar(sc, label="실제 SDLP"); ax.set_xlabel("z-PC1"); ax.set_ylabel("z-PC2")
    ax.set_title(f"z-space (PC1~SDLP r={corr_sdlp:.2f}, ~speed r={corr_spd:.2f})")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"fig_gen_zspace_{exp}.png"), dpi=120); plt.close(fig)

    # 2) distribution overlays
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, k, t in zip(axes, ["sdlp", "mean_speed", "speed_std"], ["SDLP", "평균속도(m/s)", "속도std"]):
        lo, hi = np.percentile(np.concatenate([real_s[k], gen_s[k]]), [1, 99])
        b = np.linspace(lo, hi, 40)
        ax.hist(real_s[k], bins=b, alpha=.5, density=True, label="실제", color="#185FA5")
        ax.hist(gen_s[k], bins=b, alpha=.5, density=True, label="생성", color="#D85A30")
        ax.set_title(f"{t}  W1={dist[k]['wasserstein']:.3g}"); ax.legend()
    fig.suptitle(f"분포 일치: 생성 vs 실제 ({exp})"); fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"fig_gen_dist_{exp}.png"), dpi=120); plt.close(fig)

    # 3) example trajectories (offset vs distance)
    fig, ax = plt.subplots(figsize=(11, 4))
    xm = np.arange(Yte.shape[1]) * dd
    for i in range(min(4, len(Yte))):
        ax.plot(xm, Yte[i, :, 0], color="#185FA5", lw=1.0, alpha=.8, label="실제" if i == 0 else None)
        ax.plot(xm, gen[i, :, 0], color="#D85A30", lw=1.0, alpha=.8, ls="--", label="생성" if i == 0 else None)
    ax.set_xlabel("거리(m)"); ax.set_ylabel("차선 offset(m)"); ax.legend()
    ax.set_title(f"생성 궤적 예시 (offset, {exp})"); fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"fig_gen_traj_{exp}.png"), dpi=120); plt.close(fig)

    # 4) kinematics
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, k, t, thr in zip(axes, ["accel", "jerk"], ["종가속도(m/s²)", "저크(m/s³)"], [4.0, 5.0]):
        lo, hi = np.percentile(np.concatenate([real_s[k], gen_s[k]]), [1, 99])
        b = np.linspace(lo, hi, 50)
        ax.hist(real_s[k], bins=b, alpha=.5, density=True, label="실제", color="#185FA5")
        ax.hist(gen_s[k], bins=b, alpha=.5, density=True, label="생성", color="#D85A30")
        ax.axvline(thr, ls=":", color="#888780"); ax.axvline(-thr, ls=":", color="#888780")
        ax.set_title(t); ax.legend()
    fig.suptitle(f"운동학 그럴듯함 ({exp})"); fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"fig_gen_kinematics_{exp}.png"), dpi=120); plt.close(fig)

    # 5) condition
    if cond_tbl:
        fig, ax = plt.subplots(figsize=(6, 4)); gs_ = sorted(cond_tbl); w = 0.38
        x = np.arange(len(gs_))
        ax.bar(x - w/2, [cond_tbl[g]["real_sdlp"] for g in gs_], w, label="실제", color="#185FA5")
        ax.bar(x + w/2, [cond_tbl[g]["gen_sdlp"] for g in gs_], w, label="생성", color="#D85A30")
        ax.set_xticks(x); ax.set_xticklabels([f"cond{g}" for g in gs_]); ax.set_ylabel("SDLP")
        ax.set_title(f"조건별 SDLP: 생성 vs 실제 ({exp})"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f"fig_gen_condition_{exp}.png"), dpi=120); plt.close(fig)

    summ = dict(exp=exp, n_test=int(len(Xte)), distribution=dist, kinematics=kin,
                zspace=dict(corr_pc1_sdlp=corr_sdlp, corr_pc1_speed=corr_spd),
                condition=cond_tbl, final_val_KL=ck.get("args", {}).get("beta"))
    json.dump(summ, open(os.path.join(REP, f"eval_gen_{exp}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # ================= report (within-experiment section) =================
    L = [f"# 운전행태 생성모델(CVAE) - 검증 리포트 ({'SMOKE 자체검증' if exp=='smoke' else exp})\n",
         "> distance 재인덱싱 궤적에 대한 조건부 CVAE. prior z~N(0,I)에서 합성 주행 생성, "
         "held-out 피실험자와 *분포* 비교.\n",
         "## 1. within-experiment 검증\n",
         f"- 테스트 윈도: {len(Xte)} | geo차원 {ck['geo_dim']} | z차원 {ck['z_dim']}",
         f"- posterior collapse 점검: 학습 KL>0 이어야 z가 쓰임 (metrics_cvae_{exp}.json 참고)\n",
         "### 분포 일치 (생성 vs 실제, 작을수록 좋음)\n",
         "| 지표 | Wasserstein | KS | 실제평균 | 생성평균 |", "|---|---|---|---|---|"]
    for k, t in [("sdlp", "SDLP"), ("mean_speed", "평균속도"), ("speed_std", "속도std")]:
        dk = dist[k]
        L.append(f"| {t} | {dk['wasserstein']:.4g} | {dk['ks']:.3f} | {dk['real_mean']:.3g} | {dk['gen_mean']:.3g} |")
    L.append(f"\n### z-space (스타일 분리)\n- z-PC1 ↔ SDLP 상관 **{corr_sdlp:.2f}**, ↔ 평균속도 **{corr_spd:.2f}** "
             "(|상관|이 크면 z가 의미있는 스타일축을 잡음)")
    L.append(f"\n### 운동학 그럴듯함 (임계 초과 비율)\n- |가속|>4 m/s²: 생성 {kin['accel_viol_gen']:.3f} / 실제 {kin['accel_viol_real']:.3f}")
    L.append(f"- |저크|>5 m/s³: 생성 {kin['jerk_viol_gen']:.3f} / 실제 {kin['jerk_viol_real']:.3f}")
    if cond_tbl:
        L.append("\n### 조건효과 재현 (SDLP)\n| cond | 실제 SDLP | 생성 SDLP | n |\n|---|---|---|---|")
        for g in sorted(cond_tbl):
            c = cond_tbl[g]; L.append(f"| {g} | {c['real_sdlp']:.4g} | {c['gen_sdlp']:.4g} | {c['n']} |")
    L.append("\n### 그림\n")
    for fn, cap in [(f"fig_gen_zspace_{exp}.png", "z-space (스타일)"),
                    (f"fig_gen_dist_{exp}.png", "분포 일치"),
                    (f"fig_gen_traj_{exp}.png", "생성 궤적 예시"),
                    (f"fig_gen_kinematics_{exp}.png", "운동학"),
                    (f"fig_gen_condition_{exp}.png", "조건별 SDLP")]:
        if os.path.exists(os.path.join(FIG, fn)):
            L.append(f"**{cap}**\n\n![{cap}](figs/{fn})\n")
    L.append("\n## 한계 (정직하게)\n")
    L.append("- 생성 다양성은 **학습 데이터 manifold에 묶임** — 데이터에 없던 운전스타일은 생성 못 함.")
    L.append("- 입력이 **god's-eye 도로기하** — 조명·시야·심리 등 *지각 요인은 모델 밖*.")
    L.append("- 열린 분포비교 기준. (교차실험 전이는 15_cross_validate 참고)\n")
    open(os.path.join(REP, "report_generative.md"), "w", encoding="utf-8").write("\n".join(L))

    print(f"분포 W1: " + ", ".join(f"{k}={dist[k]['wasserstein']:.3g}" for k in dist))
    print(f"z-PC1↔SDLP r={corr_sdlp:.2f} | 운동학 |a|>4 생성{kin['accel_viol_gen']:.3f}/실제{kin['accel_viol_real']:.3f}")
    print("wrote report_generative.md + figs")


if __name__ == "__main__":
    main()
