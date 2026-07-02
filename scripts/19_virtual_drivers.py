# -*- coding: utf-8 -*-
"""
19_virtual_drivers.py  --  synthetic driver POPULATION v2 (v2.4 HumanlikePolicy 기반).

각 합성 운전자 i는 실제 train/val 도로에서 조건(지상/지하)별 결합 부트스트랩한 특성:
  e_sigma_i : 의도-방황 크기  (도로 SDLP에 비례; v2.4는 미터 단위라 직접 매핑)
  b_i       : 차선 편향(LPM)  -> HumanlikePolicy.b_bias
  γ_i       : 속도 취향       -> HumanlikePolicy.v_scale (PD가 γ·v_ref 추종)
(e_tau, e_lpf 는 20_profile_eval의 2D 보정값을 그대로 사용 — profile_eval json에서 로드)

  python 19_virtual_drivers.py --exp 2024 --per_road 3
"""
import os, json, argparse, importlib
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from common import ART, REP, CACHE, gen_split, wasserstein1d
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


def road_traits(roads):
    return [dict(sdlp=float(np.std(r["e_ref"])), lpm=float(np.mean(r["e_ref"])),
                 v=float(np.mean(r["v_ref"])), cond=int(r.get("cond", 0))) for r in roads]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="2024")
    ap.add_argument("--per_road", type=int, default=3)
    args = ap.parse_args()
    exp = args.exp

    cal = json.load(open(os.path.join(REP, f"profile_eval_{exp}.json"), encoding="utf-8"))
    e_tau, e_sig0, e_lpf = cal["e_tau"], cal["e_sigma"], cal["e_lpf"]
    print(f"[{exp}] calibrated driver: e_tau={e_tau:.0f}m e_sigma={e_sig0:.3f}m e_lpf={e_lpf:.0f}m", flush=True)

    roads, subject, dd = load_roads(os.path.join(CACHE, f"env_roads_{exp}.npz"))
    roads = trim_roads(roads)
    subject = np.array([r["subject"] for r in roads], "int64")
    tr, va, te = gen_split(subject, seed=0)
    fit_roads = [r for r, m in zip(roads, tr | va) if m]
    test_roads = [r for r, m in zip(roads, te) if m]
    model = PPO.load(os.path.join(ART, f"rl_{exp}.zip"), device="cpu")

    pool = road_traits(fit_roads)
    sdlp_pool = float(np.mean([t["sdlp"] for t in pool]))
    v_pool = float(np.mean([t["v"] for t in pool]))
    by_cond = {}
    for t in pool:
        by_cond.setdefault(t["cond"], []).append(t)
    rng = np.random.RandomState(0)

    env = DrivingEnv(test_roads, dd=dd, record=True)
    rows = []
    for k, road in enumerate(test_roads):
        cand = by_cond.get(int(road.get("cond", 0)), pool)
        for j in range(args.per_road):
            t = cand[rng.randint(len(cand))]
            pol = p20.HumanlikePolicy(
                model, e_tau=e_tau, e_lpf=e_lpf,
                e_sigma=float(np.clip(e_sig0 * t["sdlp"] / sdlp_pool, 0.03, 1.2)),
                b_bias=t["lpm"], v_scale=t["v"] / v_pool,
                seed=1000 + k * 10 + j)
            pol.reset()
            traj, off = rollout(env, pol, k)
            if len(traj) > 50:
                sg = p20.rl_signals(traj)
                rows.append(dict(road=k, off=bool(off),
                                 sdlp=float(sg["e"].std()), lpm=float(sg["e"].mean()),
                                 v=float(np.mean(traj[:, 2])),
                                 srr=float(p20.srr(sg["theta"], 0.5))))
        print(f"  road {k+1}/{len(test_roads)} done", flush=True)

    hum = road_traits(test_roads)
    hum_srr = [float(p20.srr(p20.human_signals(r, dd)["theta"], 0.5)) for r in test_roads]
    H = {m: np.array([t[m] for t in hum]) for m in ["sdlp", "lpm", "v"]}
    H["srr"] = np.array(hum_srr)
    S = {m: np.array([r[m] for r in rows if not r["off"]]) for m in ["sdlp", "lpm", "v", "srr"]}
    off_rate = float(np.mean([r["off"] for r in rows]))
    W = {m: float(wasserstein1d(S[m], H[m])) for m in ["sdlp", "lpm", "v", "srr"]}
    print(f"population: n={len(rows)} offroad={off_rate:.2f}", flush=True)
    for m in ["sdlp", "lpm", "v", "srr"]:
        print(f"  {m:5s} human {H[m].mean():.3f}±{H[m].std():.3f} | synth {S[m].mean():.3f}±{S[m].std():.3f} "
              f"| W1={W[m]:.3f}", flush=True)

    # ---- figures ----
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, m, t in zip(axes, ["sdlp", "lpm", "v", "srr"],
                        ["SDLP(m)", "LPM(m)", "평균속도(m/s)", "SRR0.5(/km)"]):
        lo = min(H[m].min(), S[m].min()); hi = max(H[m].max(), S[m].max())
        pad = 0.08 * (hi - lo + 1e-6); b = np.linspace(lo - pad, hi + pad, 20)
        ax.hist(H[m], bins=b, alpha=.55, density=True, label="사람(test)", color="#185FA5")
        ax.hist(S[m], bins=b, alpha=.55, density=True, label="합성", color="#7F77DD")
        ax.set_title(f"{t}  W1={W[m]:.3f}", fontsize=10); ax.legend(fontsize=8)
    fig.suptitle(f"합성 운전자 군집 v2 (v2.4 드라이버, {exp})")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"fig_vd_population_{exp}.png"), dpi=120); plt.close(fig)

    road0 = test_roads[0]
    grid = np.arange(len(road0["e_ref"])) * dd
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(grid, road0["e_ref"], color="#185FA5", lw=1.2, alpha=.7, label="사람(실제)")
    cols = ["#7F77DD", "#1D9E75", "#D85A30"]
    cand = by_cond.get(int(road0.get("cond", 0)), pool)
    for j in range(3):
        t = cand[rng.randint(len(cand))]
        pol = p20.HumanlikePolicy(model, e_tau=e_tau, e_lpf=e_lpf,
                                  e_sigma=float(np.clip(e_sig0 * t["sdlp"] / sdlp_pool, 0.03, 1.2)),
                                  b_bias=t["lpm"], v_scale=t["v"] / v_pool, seed=77 + j)
        pol.reset()
        traj, _ = rollout(env, pol, 0)
        ax.plot(traj[:, 0], traj[:, 1], color=cols[j], lw=1.0, alpha=.9,
                label=f"합성{j+1} (SDLP타깃 {t['sdlp']:.2f}, 편향 {t['lpm']:+.2f})")
    ax.set_xlabel("거리(m)"); ax.set_ylabel("차선 offset(m)"); ax.legend(fontsize=8)
    ax.set_title(f"같은 도로, 서로 다른 합성 운전자 3명 (v2, {exp})")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"fig_vd_drivers_{exp}.png"), dpi=120); plt.close(fig)

    json.dump(dict(exp=exp, n_rollouts=len(rows), off_rate=off_rate, per_road=args.per_road,
                   driver=dict(e_tau=e_tau, e_sigma0=e_sig0, e_lpf=e_lpf),
                   human={m: [float(H[m].mean()), float(H[m].std())] for m in H},
                   synth={m: [float(S[m].mean()), float(S[m].std())] for m in S}, w1=W),
              open(os.path.join(REP, f"virtual_drivers_{exp}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    L = ["\n\n---\n\n## 8. 합성 운전자 군집 v2 (v2.4 드라이버 기반)\n",
         f"특성(SDLP→의도방황 크기, LPM→차선편향 b, 속도→γ)을 조건별 결합 부트스트랩, "
         f"운전자 {args.per_road}명/도로 × test {len(test_roads)}개 = {len(rows)} 주행 "
         f"(이탈율 {off_rate:.2f}).\n",
         "| 지표 | 사람 mean±std | 합성 mean±std | W1 |", "|---|---|---|---|"]
    for m, t in [("sdlp", "SDLP"), ("lpm", "LPM"), ("v", "속도"), ("srr", "SRR_0.5")]:
        L.append(f"| {t} | {H[m].mean():.3f}±{H[m].std():.3f} | {S[m].mean():.3f}±{S[m].std():.3f} | {W[m]:.3f} |")
    L += [f"\n![군집 분포](figs/fig_vd_population_{exp}.png)\n",
          f"![합성 운전자 예시](figs/fig_vd_drivers_{exp}.png)\n",
          "- v1 대비: 속도는 PD 컨트롤러가 γ·v_ref를 직접 추종(속도편향 보정 불필요), "
          "SDLP 특성은 미터 단위로 직접 주입(α 역변환 제거), SRR까지 군집 지표에 포함.\n"]
    open(os.path.join(REP, "report_rl.md"), "a", encoding="utf-8").write("\n".join(L))
    print("wrote section 8 + figs + virtual_drivers json", flush=True)


if __name__ == "__main__":
    main()
