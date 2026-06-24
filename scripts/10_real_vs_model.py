# -*- coding: utf-8 -*-
"""
10_real_vs_model.py  --  "실제 사람 vs BC모델" fidelity comparison (2024 data).
Uses the deterministic point model (the 'mean-line' BC) on held-out test subjects.
Open-loop 1-step prediction fidelity. Reads cache/preds_2024_all.npz.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import REP, CACHE, ACTIONS

FIG = os.path.join(REP, "figs"); os.makedirs(FIG, exist_ok=True)
for fam in ["Malgun Gothic", "NanumGothic", "Gulim"]:
    try:
        plt.rcParams["font.family"] = fam; break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False
ACT_KO = {"steering": "조향", "throttle": "가속", "brake": "브레이크"}


def r2(t, p):
    return 1 - np.sum((t - p) ** 2) / (np.sum((t - t.mean()) ** 2) + 1e-12)


def mae(t, p):
    return float(np.mean(np.abs(t - p)))


def main():
    pp = os.path.join(CACHE, "preds_2024_all.npz")
    if not os.path.exists(pp):
        raise SystemExit("preds_2024_all.npz 없음 — arm1(all+both) 완료 후 실행")
    P = np.load(pp, allow_pickle=True)
    y = P["y"]; pred = P["mu_point"]            # BC model = deterministic 'mean-line'
    ends = P["ends"]; cond = P["cond"]; subject = P["subject"]
    d = np.load(os.path.join(CACHE, "dataset_2024.npz"), allow_pickle=True)
    run_of_end = d["run_id"].astype("int64")[ends]
    act_use = [a for a in ACTIONS if y[:, ACTIONS.index(a)].std() > 1e-4]

    # 1) predicted vs real hexbin
    fig, axes = plt.subplots(1, len(act_use), figsize=(5.5*len(act_use), 4.2))
    for ax, a in zip(np.atleast_1d(axes), act_use):
        j = ACTIONS.index(a); t, q = y[:, j], pred[:, j]
        ax.hexbin(t, q, gridsize=45, mincnt=1, cmap="viridis", bins="log")
        lo, hi = np.percentile(t, [0.5, 99.5]); ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel("실제(사람)"); ax.set_ylabel("BC모델 예측")
        ax.set_title(f"{ACT_KO[a]}  R²={r2(t, q):.3f}  MAE={mae(t, q):.4f}")
    fig.suptitle("실제 vs BC모델 (held-out 피실험자, 1-스텝)"); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_rvm_scatter.png"), dpi=120); plt.close(fig)

    # 2) marginal distribution overlay
    fig, axes = plt.subplots(1, len(act_use), figsize=(5.5*len(act_use), 3.6))
    for ax, a in zip(np.atleast_1d(axes), act_use):
        j = ACTIONS.index(a); t, q = y[:, j], pred[:, j]
        lo, hi = np.percentile(np.concatenate([t, q]), [0.5, 99.5])
        bins = np.linspace(lo, hi, 60)
        ax.hist(t, bins=bins, alpha=.5, density=True, label="실제(사람)", color="#185FA5")
        ax.hist(q, bins=bins, alpha=.5, density=True, label="BC모델", color="#D85A30")
        ax.set_title(f"{ACT_KO[a]} 분포"); ax.legend()
    fig.suptitle("행동 분포: 실제 vs BC모델"); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_rvm_dist.png"), dpi=120); plt.close(fig)

    # 3) timeseries overlay (one long test run)
    runs, cnts = np.unique(run_of_end, return_counts=True)
    pick = runs[np.argmax(cnts)]
    sel = run_of_end == pick; order = np.argsort(ends[sel])
    yy = y[sel][order]; qq = pred[sel][order]
    cnd = "지하" if cond[sel][0] == 1 else "지상"; subj = int(subject[sel][0])
    sl = slice(0, min(900, len(yy)))
    fig, axes = plt.subplots(len(act_use), 1, figsize=(12, 3.1*len(act_use)), sharex=True)
    for ax, a in zip(np.atleast_1d(axes), act_use):
        j = ACTIONS.index(a); t = np.arange(len(yy))[sl]
        ax.plot(t, yy[sl, j], color="#185FA5", lw=1.3, label="실제(사람)")
        ax.plot(t, qq[sl, j], color="#D85A30", lw=1.0, alpha=.85, label="BC모델")
        ax.set_ylabel(ACT_KO[a]); ax.legend(loc="upper right", fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("프레임 (~19Hz)")
    fig.suptitle(f"실제 vs BC모델 시계열 (피실험자{subj} {cnd}, run#{int(pick)})")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_rvm_timeseries.png"), dpi=120); plt.close(fig)

    # 4) fidelity by condition (지상/지하)
    bycond = {}
    fig, ax = plt.subplots(figsize=(6.5, 4)); w = 0.38; xx = np.arange(len(act_use))
    for ci, (nm, c) in enumerate([("지상", 0), ("지하", 1)]):
        r2s = []
        for a in act_use:
            j = ACTIONS.index(a); m = cond == c
            r2s.append(r2(y[m, j], pred[m, j]))
        bycond[nm] = {a: float(v) for a, v in zip(act_use, r2s)}
        ax.bar(xx + (ci-0.5)*w, r2s, w, label=nm, color=["#1D9E75", "#185FA5"][ci])
        for k, v in enumerate(r2s):
            ax.text(xx[k]+(ci-0.5)*w, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xx); ax.set_xticklabels([ACT_KO[a] for a in act_use]); ax.set_ylim(0, 1.05)
    ax.set_ylabel("R²"); ax.set_title("충실도: 지상 vs 지하"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_rvm_bycond.png"), dpi=120); plt.close(fig)

    # 5) per-subject fidelity distribution (steering)
    j = ACTIONS.index("steering"); subs = np.unique(subject)
    persubj = []
    for s in subs:
        m = subject == s
        if m.sum() > 100:
            persubj.append(r2(y[m, j], pred[m, j]))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.boxplot([persubj], vert=True, labels=["조향"])
    ax.scatter(np.ones(len(persubj)) + np.random.uniform(-0.05, 0.05, len(persubj)),
               persubj, alpha=.6, color="#185FA5", s=18)
    ax.set_ylabel("피실험자별 R²"); ax.set_title("피실험자별 충실도 분포 (조향)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_rvm_bysubject.png"), dpi=120); plt.close(fig)

    # ---- report ----
    overall = {a: dict(r2=float(r2(y[:, ACTIONS.index(a)], pred[:, ACTIONS.index(a)])),
                       mae=mae(y[:, ACTIONS.index(a)], pred[:, ACTIONS.index(a)])) for a in act_use}
    json.dump(dict(overall=overall, by_condition=bycond,
                   per_subject_steering=dict(median=float(np.median(persubj)),
                                             min=float(np.min(persubj)), max=float(np.max(persubj)))),
              open(os.path.join(REP, "real_vs_model.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    L = ["# 실제 사람 vs BC모델 — 충실도 비교 (2024 지하고속도로)\n",
         "> held-out 피실험자(학습 미사용)에서 **결정론 BC(평균선) 모델**이 사람 주행을 얼마나 재현하는가. "
         "열린 루프(1-스텝) 기준.\n",
         "## 전체 정확도\n", "| 행동 | R² | MAE |", "|---|---|---|"]
    for a in act_use:
        L.append(f"| {ACT_KO[a]} | {overall[a]['r2']:.3f} | {overall[a]['mae']:.4f} |")
    L.append("\n## 지상 vs 지하 충실도 (R²)\n| 행동 | 지상 | 지하 |\n|---|---|---|")
    for a in act_use:
        L.append(f"| {ACT_KO[a]} | {bycond['지상'][a]:.3f} | {bycond['지하'][a]:.3f} |")
    L.append(f"\n- 피실험자별 조향 R² 중앙값 {np.median(persubj):.3f} "
             f"(범위 {np.min(persubj):.3f}~{np.max(persubj):.3f}) → 사람별 편차.")
    L.append("\n## 그림\n")
    for fn, cap in [("fig_rvm_scatter.png", "예측 vs 실제 산점도"),
                    ("fig_rvm_dist.png", "행동 분포 일치"),
                    ("fig_rvm_timeseries.png", "시계열 오버레이"),
                    ("fig_rvm_bycond.png", "지상 vs 지하 충실도"),
                    ("fig_rvm_bysubject.png", "피실험자별 충실도")]:
        L.append(f"**{cap}**\n\n![{cap}](figs/{fn})\n")
    L.append("## 주의\n")
    L.append("- **열린 루프 1-스텝 충실도**다 (사람의 실제 상태를 매 순간 주고 다음 조작 예측). "
             "자율 누적주행(닫힌 루프)과는 다름.")
    L.append("- 조향 R²가 높은 데엔 *자차상태 연속성* 기여가 큼(ablation 참조) — "
             "순수 도로반응 충실도는 그보다 낮음.")
    open(os.path.join(REP, "report_real_vs_model.md"), "w", encoding="utf-8").write("\n".join(L))
    print("=== real vs BC model (test) ===")
    for a in act_use:
        print(f"  {a:9s} R2={overall[a]['r2']:.3f} MAE={overall[a]['mae']:.4f} "
              f"| 지상 {bycond['지상'][a]:.3f} 지하 {bycond['지하'][a]:.3f}")
    print("figs + report_real_vs_model.md written")


if __name__ == "__main__":
    main()
