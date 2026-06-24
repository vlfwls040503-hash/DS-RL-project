# -*- coding: utf-8 -*-
"""
05_report.py  --  assemble reports/report_BC.md from the JSON artifacts + figures.
Run after 04_eval_report.py.
"""
import os, json
from common import REP, ACTIONS

ACT_KO = {"steering": "조향(steering)", "throttle": "가속(throttle)", "brake": "브레이크(brake)"}


def jload(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def fmt_metrics_table(models, evals):
    lines = ["| 모델 | 행동 | MAE | RMSE | R² |", "|---|---|---|---|---|"]
    for mname in models:
        ev = evals[mname]["test"]
        for a in ACTIONS:
            m = ev[a]
            lines.append(f"| {mname} | {ACT_KO[a]} | {m['mae']:.5f} | {m['rmse']:.5f} | {m['r2']:.3f} |")
    return "\n".join(lines)


def main():
    expl = jload(os.path.join(REP, "explore_summary.json"))
    ev = jload(os.path.join(REP, "eval_summary.json"))
    mlp = jload(os.path.join(REP, "metrics_mlp.json"))
    gru = jload(os.path.join(REP, "metrics_gru.json"))
    if ev is None:
        raise SystemExit("eval_summary.json 없음 — 먼저 04_eval_report.py 실행")

    models = [m for m in ev.keys() if not m.startswith("_")]
    best = ev.get("_best_model")

    L = []
    L.append("# 드라이빙 시뮬레이터 데이터 기반 모방학습(Behavioral Cloning) 결과\n")
    L.append("> 전체 주행 제어(조향+가속+브레이크)를 인간 주행 로그로부터 모방하는 에이전트.\n")

    # TL;DR
    L.append("## 1. 한 줄 요약\n")
    bev = ev[best]["test"]
    L.append(f"- 가능합니다. **{best.upper()}** 모델이 처음 보는 피실험자(held-out)에서 "
             f"조향 R²=**{bev['steering']['r2']:.3f}**, 가속 R²=**{bev['throttle']['r2']:.3f}**, "
             f"브레이크 R²=**{bev['brake']['r2']:.3f}** 달성.")
    L.append("- 조향은 사실상 완벽하게 인간 행태를 재현, 가속은 양호, 브레이크는 데이터상 "
             "제동이 희소해 난도가 높음.\n")

    # Data
    L.append("## 2. 데이터 개요\n")
    if expl:
        L.append(f"- 본주행 파일 **{expl['n_files']}개**, 피실험자 **{expl['n_subjects']}명**, "
                 f"자차(uv) 총 **{expl['total_uv_rows']:,}행** (샘플링 ~{expl['approx_hz']:.1f}Hz)")
        L.append(f"- 시나리오별 행수(S1~S4): {expl['per_scenario_uv_rows']}")
        L.append(f"- 주행모드: 전부 Manual(순수 인간 제어) → 모방학습에 이상적\n")

    # Problem
    L.append("## 3. 문제 정의\n")
    L.append("- **행동(예측 대상, 3차원)**: 조향, 가속(0~1), 브레이크(0~1)")
    L.append("- **상태(입력, 24차원)**: 속도, 종/횡 가속도, yaw rate, 차선곡률, 차선·도로중심 offset, "
             "SDLP, 좌/우 경계거리, 차로/차도폭, 도로 종/횡경사, 좌/우 차선침범, "
             "앞차거리·TTC(가공: 캡+플래그+1/TTC), 시나리오 one-hot(S1~S4)")
    L.append("- **데이터 분할: 피실험자 단위** (train/val/test에 서로 다른 사람) → "
             "*새로운 운전자에 대한 일반화*를 측정. 이것이 '피실험자 수가 적다'는 문제의 핵심 검증.")
    L.append(f"  - val 피실험자: {mlp['val_subjects']}, test 피실험자: {mlp['test_subjects']}")
    L.append(f"  - 행수: train {mlp['rows']['train']:,} / val {mlp['rows']['val']:,} / "
             f"test {mlp['rows']['test']:,}\n")

    # Models
    L.append("## 4. 모델\n")
    L.append("- **MLP**: 단일 프레임 상태 → 행동. LayerNorm+GELU+Dropout, SmoothL1 손실, "
             "입력·타깃 z-정규화(타깃 정규화로 3개 행동의 손실 기여 균형).")
    if gru:
        w = gru["args"]["window"]; s = gru["args"]["stride"]
        L.append(f"- **GRU**: 최근 {w}프레임(stride {s}, ~{w*s/44:.1f}s 맥락) 시퀀스 → 마지막 시점 행동. "
                 "시간맥락으로 가속/브레이크 개선을 노림.")
    L.append("")

    # Results
    L.append("## 5. 결과 (held-out 테스트 피실험자, 원단위)\n")
    L.append(fmt_metrics_table(models, ev))
    # baseline reference
    if mlp and "baseline" in mlp:
        b = mlp["baseline"]["predict_train_mean"]
        L.append("\n참고 — '평균값 예측' 단순 기준선 MAE: "
                 + ", ".join(f"{ACT_KO[a]} {b[a]['mae']:.5f}" for a in ACTIONS))
    L.append(f"\n→ 종합 최적 모델: **{best.upper()}**\n")

    # per scenario (best)
    L.append("### 시나리오(격벽조건)별 오차 — 최적 모델\n")
    ps = ev[best].get("per_scenario", {})
    if ps:
        L.append("| 시나리오 | 조향 MAE | 가속 MAE | 브레이크 MAE |")
        L.append("|---|---|---|---|")
        for sc in sorted(ps.keys(), key=lambda x: int(x)):
            d = ps[sc]
            L.append(f"| S{sc} | {d['steering']['mae']:.5f} | {d['throttle']['mae']:.5f} "
                     f"| {d['brake']['mae']:.5f} |")
    L.append("")

    # figures
    L.append("## 6. 그림\n")
    for fn, cap in [("fig_train_curves.png", "학습 곡선"),
                    ("fig_scatter.png", "예측 vs 실제 (행동별)"),
                    ("fig_action_hist.png", "행동 분포: 실제 vs 예측"),
                    ("fig_per_scenario_mae.png", "시나리오별 MAE"),
                    ("fig_timeseries.png", "테스트 주행 1개 구간 시계열")]:
        L.append(f"**{cap}**\n\n![{cap}](figs/{fn})\n")

    # interpretation
    L.append("## 7. 해석\n")
    L.append("- **조향**: 차선 offset·곡률 등 기하 정보만으로 인간 조향이 거의 결정적이라 R²≈1.0. "
             "차선유지 행태 재현은 신뢰도 높음.")
    L.append("- **가속**: 추세는 잘 따르나 인간의 계단식 페달 조작을 평활화하는 경향.")
    L.append("- **브레이크**: 평균 0.013으로 매우 희소(고속도로 터널). 돌발적 제동이라 단일프레임 "
             "예측이 어렵고 R²가 낮음. 시간맥락(GRU)·클래스 가중·이벤트 검출이 개선 여지.\n")

    # limitations
    L.append("## 8. 한계와 주의 (중요)\n")
    L.append("- **이건 '열린 루프(open-loop)' 1-스텝 예측 성능**입니다. 실제로 시뮬레이터에 "
             "에이전트를 태워 *스스로 누적 주행*시키면 작은 오차가 쌓여 분포가 벗어나는 "
             "**covariate shift / compounding error** 문제가 생깁니다. BC의 근본 한계.")
    L.append("- 따라서 위 R²가 높다고 곧바로 '자율주행이 된다'는 의미는 아닙니다. "
             "닫힌 루프(closed-loop) 검증이 반드시 필요.")
    L.append("- 브레이크 희소성, 단일 속도제한(50)·단일 차로 등 시나리오 다양성 제한.")
    L.append("- 29명·동일 코스라 도메인이 좁음 → 다른 도로/속도역에는 외삽 주의.\n")

    # next steps
    L.append("## 9. 다음 단계 (권장 순서)\n")
    L.append("1. **시퀀스 모델 강화**: GRU/Transformer 윈도우 확대, 브레이크 이벤트 가중 손실.")
    L.append("2. **DAgger / 닫힌 루프**: 시뮬레이터(SCANeR)를 프로그램 제어로 연결해 에이전트가 "
             "주행→전문가 보정 라벨 수집을 반복. compounding error 직접 완화.")
    L.append("3. **오프라인 RL(CQL/IQL)**: 보상(안전 TTC·차선이탈 + 효율 속도 + 승차감 급가감속)을 "
             "설계하면 *추가 실험 없이* 인간보다 나은 정책 학습 가능. — '피실험자 부족' 문제의 정공법.")
    L.append("4. **데이터 증강/도메인 확장**: 더 다양한 코스·속도·교통조건으로 일반화 향상.\n")

    # repro
    L.append("## 10. 재현 방법\n")
    L.append("```bash")
    L.append("cd D:\\driving_bc\\scripts")
    L.append("python 01_explore.py          # 데이터 탐색")
    L.append("python 02_build_dataset.py    # cache/dataset.npz 생성")
    L.append("python 03_train.py --model mlp")
    L.append("python 03_train.py --model gru --batch 1024 --window 24 --stride 2")
    L.append("python 04_eval_report.py      # 지표·그림")
    L.append("python 05_report.py           # 이 리포트 생성")
    L.append("```")
    L.append("\n**산출물**: `artifacts/bc_*.pt`(모델), `cache/dataset.npz`(데이터), "
             "`reports/figs/*`(그림), `reports/*.json`(지표).\n")

    out = os.path.join(REP, "report_BC.md")
    open(out, "w", encoding="utf-8").write("\n".join(L))
    print("wrote", out)


if __name__ == "__main__":
    main()
