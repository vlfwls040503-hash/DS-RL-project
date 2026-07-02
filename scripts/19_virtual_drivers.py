# -*- coding: utf-8 -*-
"""
19_virtual_drivers.py  --  synthetic driver POPULATION on the closed-loop env.

Motivation: RL-σ reproduces one driver's SDLP (single α). Real subjects differ in
weave (SDLP), lane bias (LPM) and speed preference. Here each synthetic driver i
gets traits (α_i, b_i, γ_i) bootstrapped JOINTLY from real train/val roads of the
SAME condition (지상/지하), then drives held-out test roads:
  α_i : steering-noise temperature  (from road SDLP via the α->SDLP calibration curve)
  b_i : lane-position bias          (obs e is shifted -> agent stabilizes at e≈b_i)
  γ_i : speed-preference ratio      (obs v_ref scaled -> agent cruises at γ_i·v_ref)
Output: population-level SDLP/LPM/speed distributions vs human test roads (W1),
figures, report section appended to report_rl.md.

  python 19_virtual_drivers.py --exp 2024 --per_road 3
"""
import os, json, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from common import ART, REP, CACHE, gen_split, wasserstein1d
from driving_env import DrivingEnv, load_roads, rollout, trim_roads

for fam in ["Malgun Gothic", "NanumGothic", "Gulim"]:
    try:
        plt.rcParams["font.family"] = fam; break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(REP, "figs"); os.makedirs(FIG, exist_ok=True)
np.random.seed(0); torch.manual_seed(0)

IDX_VREF, IDX_E = 1, 2      # observation layout (driving_env.build_obs)


def road_traits(roads):
    """Per-road human traits: (sdlp, lpm, v_mean, cond)."""
    return [dict(sdlp=float(np.std(r["e_ref"])), lpm=float(np.mean(r["e_ref"])),
                 v=float(np.mean(r["v_ref"])), cond=int(r.get("cond", 0))) for r in roads]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="2024")
    ap.add_argument("--per_road", type=int, default=3, help="synthetic drivers per test road")
    args = ap.parse_args()
    exp = args.exp

    roads, subject, dd = load_roads(os.path.join(CACHE, f"env_roads_{exp}.npz"))
    roads = trim_roads(roads)
    subject = np.array([r["subject"] for r in roads], "int64")
    tr, va, te = gen_split(subject, seed=0)
    fit_roads = [r for r, m in zip(roads, tr | va) if m]        # trait pool (train+val)
    test_roads = [r for r, m in zip(roads, te) if m]
    val_roads = [r for r, m in zip(roads, va) if m] or fit_roads[:8]
    print(f"[{exp}] trait-pool roads={len(fit_roads)}  test roads={len(test_roads)}", flush=True)

    model = PPO.load(os.path.join(ART, f"rl_{exp}.zip"), device="cpu")
    orig_logstd = model.policy.log_std.data.clone()

    def set_temp(alpha):
        ls = orig_logstd.clone()
        ls[0] = ls[0] + float(np.log(alpha)); ls[1] = float(np.log(0.01))
        model.policy.log_std.data = ls

    # ---- α -> closed-loop SDLP calibration curve (on val roads) ----
    alphas = [0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
    env_cal = DrivingEnv(val_roads, dd=dd, record=True)
    curve = []
    for a in alphas:
        set_temp(a)
        vals = []
        for k in range(len(val_roads)):
            traj, _ = rollout(env_cal, lambda o, e: model.predict(o, deterministic=False)[0], k)
            if len(traj) > 10:
                vals.append(float(traj[:, 1].std()))
        curve.append(float(np.mean(vals)))
    print("calibration: alpha", alphas, "-> SDLP", np.round(curve, 3).tolist(), flush=True)

    def sdlp_to_alpha(s):
        return float(np.clip(np.interp(s, curve, alphas), alphas[0], alphas[-1]))

    # ---- trait pool (joint bootstrap, condition-matched) ----
    pool = road_traits(fit_roads)
    v_pop = float(np.mean([t["v"] for t in pool]))
    by_cond = {}
    for t in pool:
        by_cond.setdefault(t["cond"], []).append(t)
    rng = np.random.RandomState(0)

    # ---- roll the synthetic population over test roads ----
    env = DrivingEnv(test_roads, dd=dd, record=True)
    rows = []
    for k, road in enumerate(test_roads):
        cand = by_cond.get(int(road.get("cond", 0)), pool)
        for j in range(args.per_road):
            t = cand[rng.randint(len(cand))]                  # joint (sdlp,lpm,v) sample
            alpha_i, b_i, gam_i = sdlp_to_alpha(t["sdlp"]), t["lpm"], t["v"] / v_pop
            set_temp(alpha_i)

            def policy(obs, e, b=b_i, g=gam_i):
                o = obs.copy(); o[IDX_E] -= b; o[IDX_VREF] *= g
                return model.predict(o, deterministic=False)[0]

            traj, off = rollout(env, policy, k)
            if len(traj) > 10:
                rows.append(dict(road=k, off=bool(off), alpha=alpha_i, b=b_i, gam=gam_i,
                                 sdlp=float(traj[:, 1].std()), lpm=float(traj[:, 1].mean()),
                                 v=float(traj[:, 2].mean())))
        print(f"  road {k+1}/{len(test_roads)} done", flush=True)
    model.policy.log_std.data = orig_logstd

    # ---- population vs human-test distributions ----
    hum = road_traits(test_roads)
    H = {m: np.array([t[m] for t in hum]) for m in ["sdlp", "lpm", "v"]}
    S = {m: np.array([r[m] for r in rows if not r["off"]]) for m in ["sdlp", "lpm", "v"]}
    off_rate = float(np.mean([r["off"] for r in rows]))
    W = {m: float(wasserstein1d(S[m], H[m])) for m in ["sdlp", "lpm", "v"]}
    print(f"population: n={len(rows)} offroad={off_rate:.2f}", flush=True)
    for m in ["sdlp", "lpm", "v"]:
        print(f"  {m:5s} human mean={H[m].mean():.3f} std={H[m].std():.3f} | "
              f"synth mean={S[m].mean():.3f} std={S[m].std():.3f} | W1={W[m]:.3f}", flush=True)

    # ---- figures ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, m, t in zip(axes, ["sdlp", "lpm", "v"], ["SDLP(m)", "LPM(m)", "평균속도(m/s)"]):
        lo = min(H[m].min(), S[m].min()); hi = max(H[m].max(), S[m].max())
        pad = 0.08 * (hi - lo + 1e-6); b = np.linspace(lo - pad, hi + pad, 22)
        ax.hist(H[m], bins=b, alpha=.55, density=True, label="사람(test)", color="#185FA5")
        ax.hist(S[m], bins=b, alpha=.55, density=True, label="합성 운전자", color="#7F77DD")
        ax.set_title(f"{t}  W1={W[m]:.3f}"); ax.legend()
    fig.suptitle(f"합성 운전자 군집 vs 사람 — 운전자 간 분산까지 ({exp})")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"fig_vd_population_{exp}.png"), dpi=120); plt.close(fig)

    # example: 3 different synthetic drivers on one road
    k0 = 0; road0 = test_roads[k0]
    grid = np.arange(len(road0["e_ref"])) * dd
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(grid, road0["e_ref"], color="#185FA5", lw=1.2, alpha=.7, label="사람(실제)")
    cols = ["#7F77DD", "#1D9E75", "#D85A30"]
    cand = by_cond.get(int(road0.get("cond", 0)), pool)
    for j in range(3):
        t = cand[rng.randint(len(cand))]
        set_temp(sdlp_to_alpha(t["sdlp"]))
        b_i, g_i = t["lpm"], t["v"] / v_pop
        traj, _ = rollout(env, lambda o, e, b=b_i, g=g_i:
                          model.predict(_mod(o, b, g), deterministic=False)[0], k0)
        ax.plot(traj[:, 0], traj[:, 1], color=cols[j], lw=1.0, alpha=.9,
                label=f"합성 운전자{j+1} (SDLP타깃 {t['sdlp']:.2f})")
    model.policy.log_std.data = orig_logstd
    ax.set_xlabel("거리(m)"); ax.set_ylabel("차선 offset(m)"); ax.legend(fontsize=8)
    ax.set_title(f"같은 도로, 서로 다른 합성 운전자 3명 ({exp})")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"fig_vd_drivers_{exp}.png"), dpi=120); plt.close(fig)

    # ---- save + report append ----
    json.dump(dict(exp=exp, n_rollouts=len(rows), off_rate=off_rate,
                   per_road=args.per_road, curve=dict(alphas=alphas, sdlp=curve),
                   human={m: [float(H[m].mean()), float(H[m].std())] for m in H},
                   synth={m: [float(S[m].mean()), float(S[m].std())] for m in S},
                   w1=W),
              open(os.path.join(REP, f"virtual_drivers_{exp}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    L = ["\n\n---\n\n## 5. 합성 운전자 군집 (운전자 간 분산 재현)\n",
         f"실제 train/val 도로의 (SDLP, LPM, 속도) 특성을 **조건(지상/지하)별 결합 부트스트랩**으로 뽑아 "
         f"합성 운전자 {args.per_road}명/도로를 test 도로에 주행시켰다 (총 {len(rows)} 주행, 이탈율 {off_rate:.2f}).\n",
         "| 지표 | 사람 mean±std | 합성군집 mean±std | W1 |", "|---|---|---|---|"]
    for m, t in [("sdlp", "SDLP"), ("lpm", "LPM"), ("v", "속도")]:
        L.append(f"| {t} | {H[m].mean():.3f}±{H[m].std():.3f} | {S[m].mean():.3f}±{S[m].std():.3f} | {W[m]:.3f} |")
    L += [f"\n![군집 분포](figs/fig_vd_population_{exp}.png)\n",
          f"![합성 운전자 예시](figs/fig_vd_drivers_{exp}.png)\n",
          "- 특성 3종을 *같은 도로에서 결합 샘플* → 특성 간 상관 보존. 조건별 샘플링으로 지상/지하 "
          "SDLP 차이는 **보정으로 재현**(창발 아님 — god's-eye 입력의 한계를 명시).",
          "- 이것이 '합성 피실험자' v1: 통계적 검정력 보강·설계안 사전평가용 가상 모집단.\n"]
    open(os.path.join(REP, "report_rl.md"), "a", encoding="utf-8").write("\n".join(L))
    print("appended section 5 -> report_rl.md ; figs + virtual_drivers json written", flush=True)


def _mod(o, b, g):
    o = o.copy(); o[IDX_E] -= b; o[IDX_VREF] *= g
    return o


if __name__ == "__main__":
    main()
