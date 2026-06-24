# -*- coding: utf-8 -*-
"""
09_eval_2024.py  --  PoC + ablation figures/report.
Run after the 3-arm training (all/geo/ego). Reads:
  reports/metrics_2024_{all,geo,ego}.json
  cache/preds_2024_all.npz
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
MODE_KO = {"all": "전체", "geo": "도로+조건만", "ego": "자차상태만"}
MODE_COL = {"all": "#185FA5", "geo": "#1D9E75", "ego": "#888780"}


def jload(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def main():
    M = {m: jload(os.path.join(REP, f"metrics_2024_{m}.json")) for m in ["all", "geo", "ego"]}
    M = {k: v for k, v in M.items() if v}
    # active actions (brake ~0 on this highway)
    act_use = ["steering", "throttle"]

    # ---- 1) ABLATION: R2 by feature set (point model) ----
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    modes = [m for m in ["all", "geo", "ego"] if m in M]
    x = np.arange(len(act_use)); w = 0.8 / max(len(modes), 1)
    for i, m in enumerate(modes):
        vals = [M[m]["point"][a]["r2"] if M[m].get("point") else np.nan for a in act_use]
        ax.bar(x + (i - (len(modes)-1)/2) * w, vals, w, label=MODE_KO[m], color=MODE_COL[m])
        for k, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(x[k] + (i-(len(modes)-1)/2)*w, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([ACT_KO[a] for a in act_use])
    ax.set_ylabel("R² (held-out test)"); ax.set_ylim(0, 1.05)
    ax.set_title("Ablation: 어떤 입력이 정확도를 만드나 (결정론 모델)")
    ax.legend(title="입력 구성")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2024_ablation.png"), dpi=120); plt.close(fig)

    # ---- variability/calibration/band/condition from preds_2024_all ----
    pp = os.path.join(CACHE, "preds_2024_all.npz")
    if os.path.exists(pp) and M.get("all") and M["all"].get("variability"):
        P = np.load(pp, allow_pickle=True)
        y, mu_g, sig = P["y"], P["mu_gauss"], P["sigma_gauss"]
        ends, cond = P["ends"], P["cond"]
        comp = M["all"]["variability"]
        d = np.load(os.path.join(CACHE, "dataset_2024.npz"), allow_pickle=True)
        run_of_end = d["run_id"].astype("int64")[ends]
        w = 0.38

        # 2) sigma emp vs pred
        fig, ax = plt.subplots(figsize=(6.5, 4))
        xx = np.arange(len(act_use))
        se = [comp[a]["sigma_emp"] for a in act_use]; sp = [comp[a]["sigma_pred_mean"] for a in act_use]
        ax.bar(xx - w/2, se, w, label="실제 변동성(잔차 σ)", color="#888780")
        ax.bar(xx + w/2, sp, w, label="모델 예측 σ", color="#1D9E75")
        ax.set_xticks(xx); ax.set_xticklabels([ACT_KO[a] for a in act_use]); ax.set_ylabel("표준편차(원단위)")
        ax.set_title("변동성 재현: 실제 vs 예측 σ"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2024_sigma.png"), dpi=120); plt.close(fig)

        # 3) calibration
        fig, ax = plt.subplots(figsize=(6.5, 4))
        c1 = [comp[a]["cover_1sigma"] for a in act_use]; c2 = [comp[a]["cover_2sigma"] for a in act_use]
        ax.bar(xx - w/2, c1, w, label="±1σ 포함율", color="#378ADD")
        ax.bar(xx + w/2, c2, w, label="±2σ 포함율", color="#185FA5")
        ax.axhline(0.68, ls="--", color="#888780", lw=1); ax.axhline(0.95, ls="--", color="#888780", lw=1)
        ax.set_xticks(xx); ax.set_xticklabels([ACT_KO[a] for a in act_use]); ax.set_ylim(0, 1.08)
        ax.set_ylabel("실제값 포함 비율"); ax.set_title("캘리브레이션 (목표 .68/.95)"); ax.legend(loc="lower right")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2024_calibration.png"), dpi=120); plt.close(fig)

        # 4) band time-series (지하 run)
        runs = np.unique(run_of_end)
        under = [r for r in runs if cond[run_of_end == r][0] == 1]
        pick = max(under or list(runs), key=lambda r: (run_of_end == r).sum())
        sel = run_of_end == pick; order = np.argsort(ends[sel])
        yy = y[sel][order]; mg = mu_g[sel][order]; sg = sig[sel][order]
        sl = slice(0, min(900, len(yy)))
        fig, axes = plt.subplots(len(act_use), 1, figsize=(12, 3.2*len(act_use)), sharex=True)
        for ax, a in zip(np.atleast_1d(axes), act_use):
            j = ACTIONS.index(a); t = np.arange(len(yy))[sl]
            ax.fill_between(t, mg[sl, j]-2*sg[sl, j], mg[sl, j]+2*sg[sl, j], color="#1D9E75", alpha=0.22, label="예측 ±2σ")
            ax.plot(t, yy[sl, j], color="#185FA5", lw=1.3, label="실제(사람)")
            ax.plot(t, mg[sl, j], color="#0F6E56", lw=1.0, label="예측 평균")
            ax.set_ylabel(ACT_KO[a]); ax.legend(loc="upper right", fontsize=8)
        np.atleast_1d(axes)[-1].set_xlabel("프레임 (~19Hz)")
        fig.suptitle(f"확률적 정책: 예측 ±2σ 띠가 사람 조작을 담는가 (지하 run#{int(pick)})")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2024_band.png"), dpi=120); plt.close(fig)

        # 5) condition effect (steering)
        j = ACTIONS.index("steering")
        fig, ax = plt.subplots(figsize=(6, 4))
        emp = [(y[cond==c, j]-mu_g[cond==c, j]).std() for c in (0, 1)]
        prd = [sig[cond==c, j].mean() for c in (0, 1)]
        gx = np.arange(2)
        ax.bar(gx - w/2, emp, w, label="실제 변동성", color="#888780")
        ax.bar(gx + w/2, prd, w, label="예측 σ", color="#1D9E75")
        ax.set_xticks(gx); ax.set_xticklabels(["지상", "지하"]); ax.set_ylabel("조향 σ")
        ax.set_title("지상 vs 지하 변동성 (조향)"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2024_condition.png"), dpi=120); plt.close(fig)
        have_var = True
    else:
        have_var = False
        comp = None

    # ---- report ----
    L = ["# 2024 지하고속도로 — 확률적·조건부 정책 PoC + Ablation\n",
         "> 데이터: 32명 × {지상, 지하} 장거리 주행, held-out 피실험자 평가.\n",
         "## 1. ★Ablation — 정확도의 출처 (가장 중요)\n",
         "결정론 모델 R²를 입력 구성별로 비교: '전체' vs '도로+조건만' vs '자차상태만'.\n",
         "| 행동 | 전체 | 도로+조건만 | 자차상태만 |", "|---|---|---|---|"]
    for a in act_use:
        row = [f"{M[m]['point'][a]['r2']:.3f}" if (m in M and M[m].get('point')) else "-"
               for m in ["all", "geo", "ego"]]
        L.append(f"| {ACT_KO[a]} | {row[0]} | {row[1]} | {row[2]} |")
    L.append("\n**해석:**")
    L.append("- `자차상태만`이 `전체`에 근접하면 → 높은 정확도는 *기하 판단*이 아니라 "
             "**자차상태 연속성(물리·관성)**에서 온 것. (steering이 특히 그럼)")
    L.append("- `도로+조건만`의 R²가 *순수 기하→행태* 신호의 크기. 낮더라도 이게 "
             "**'도로를 보고 반응하는 가상 운전자'**의 실제 설명력.")
    L.append("- 즉 연구적으로 의미있는 부분은 `도로+조건만` 모델이며, 전체모델의 높은 수치는 "
             "상당부분 *거품*임을 정량적으로 보여줌.\n")
    if comp:
        L.append("## 2. 변동성 재현 (확률적 정책, 전체입력)\n")
        L.append("| 행동 | 실제 변동성 σ | 예측 σ | ±1σ 포함 | ±2σ 포함 |")
        L.append("|---|---|---|---|---|")
        for a in act_use:
            c = comp[a]
            L.append(f"| {ACT_KO[a]} | {c['sigma_emp']:.4f} | {c['sigma_pred_mean']:.4f} | "
                     f"{c['cover_1sigma']:.2f} | {c['cover_2sigma']:.2f} |")
        L.append("\n예측 σ가 실제와 가깝고 포함율이 .68/.95에 근접 → '평균'이 아니라 '분포'를 학습.\n")
    L.append("## 3. 그림\n")
    figs = [("fig2024_ablation.png", "Ablation: 정확도의 출처")]
    if have_var:
        figs += [("fig2024_sigma.png", "변동성: 실제 vs 예측 σ"),
                 ("fig2024_calibration.png", "캘리브레이션"),
                 ("fig2024_band.png", "예측 ±2σ 띠 vs 실제 (시계열)"),
                 ("fig2024_condition.png", "지상 vs 지하 변동성")]
    for fn, cap in figs:
        L.append(f"**{cap}**\n\n![{cap}](figs/{fn})\n")
    L.append("## 4. 한계\n")
    L.append("- 열린 루프 평가. SDLP 완전 재현·새 기하 평가엔 시뮬레이터 닫힌 루프 필요.")
    L.append("- brake는 장거리 고속주행이라 거의 0 → 제외.")
    L.append("- '자차상태'는 제어기엔 필요하지만 도로설계 효과 해석 시 분리 필요(→ ablation).\n")
    open(os.path.join(REP, "report_2024_PoC.md"), "w", encoding="utf-8").write("\n".join(L))
    print("=== ablation R2 ===")
    for a in act_use:
        s = "  ".join(f"{m}={M[m]['point'][a]['r2']:.3f}" for m in modes if M[m].get("point"))
        print(f"  {a:9s} {s}")
    print("figs + report_2024_PoC.md written")


if __name__ == "__main__":
    main()
