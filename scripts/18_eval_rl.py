# -*- coding: utf-8 -*-
"""
18_eval_rl.py  --  closed-loop evaluation on held-out roads.

Compares three closed-loop drivers on the SAME surrogate env / test roads:
  PD  : privileged ref-tracking controller (env validity reference)
  BC  : env-native behavioral cloning (obs->action supervised on train roads)
  RL  : PPO agent (17_train_rl)
Metrics per road: completion(offroad), SDLP, LPM, mean speed, |e-e_ref| RMSE, jerk viol.
Also distribution match (Wasserstein) vs human refs.  Korean report + figures.

  python 18_eval_rl.py --smoke
  python 18_eval_rl.py --exp 2024
"""
import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from common import (ART, REP, CACHE, GEN_DD, gen_split, wasserstein1d,
                    RL_DT, RL_A_MAX)
from driving_env import (DrivingEnv, load_roads, rollout, pd_action,
                         make_expert_dataset, trim_roads, OBS_DIM)

for fam in ["Malgun Gothic", "NanumGothic", "Gulim"]:
    try:
        plt.rcParams["font.family"] = fam; break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(REP, "figs"); os.makedirs(FIG, exist_ok=True)
torch.manual_seed(0); np.random.seed(0)


# ----------------------------- env-native BC baseline -----------------------------
def train_env_bc(train_roads, dd, epochs=30, lr=1e-3):
    X, Y = make_expert_dataset(train_roads, dd)
    net = nn.Sequential(nn.Linear(OBS_DIM, 64), nn.ReLU(),
                        nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2), nn.Tanh())
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.SmoothL1Loss()
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for s in range(0, n, 1024):
            b = perm[s:s + 1024]
            opt.zero_grad()
            loss = lossf(net(Xt[b]), Yt[b])
            loss.backward(); opt.step()
            tot += loss.item() * len(b)
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  [env-BC] ep{ep:3d} loss={tot/n:.4f}", flush=True)
    net.eval()
    return net


# ----------------------------- rollout metrics -----------------------------
def eval_driver(roads, dd, policy_fn, name):
    env = DrivingEnv(roads, dd=dd, record=True)
    rows = []
    for k in range(len(roads)):
        traj, off = rollout(env, policy_fn, k)
        r = roads[k]
        if len(traj) < 10:
            rows.append(dict(off=True, sdlp=np.nan, lpm=np.nan, v=np.nan, rmse=np.nan, jviol=np.nan))
            continue
        grid = np.arange(len(r["e_ref"])) * dd
        e_ref_i = np.interp(traj[:, 0], grid, r["e_ref"])
        jerk = np.diff(traj[:, 3]) / RL_DT
        rows.append(dict(off=bool(off),
                         sdlp=float(traj[:, 1].std()), lpm=float(traj[:, 1].mean()),
                         v=float(traj[:, 2].mean()),
                         rmse=float(np.sqrt(np.mean((traj[:, 1] - e_ref_i) ** 2))),
                         jviol=float(np.mean(np.abs(jerk) > 5.0))))
    off_rate = float(np.mean([x["off"] for x in rows]))
    agg = {m: float(np.nanmean([x[m] for x in rows])) for m in ["sdlp", "lpm", "v", "rmse", "jviol"]}
    agg["off_rate"] = off_rate
    print(f"  [{name}] offroad {off_rate:.2f} | SDLP {agg['sdlp']:.3f} | RMSE_e {agg['rmse']:.3f} "
          f"| v {agg['v']:.1f} | jerk_viol {agg['jviol']:.3f}", flush=True)
    return rows, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="smoke")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    exp = "smoke" if args.smoke else args.exp

    roads, subject, dd = load_roads(os.path.join(CACHE, f"env_roads_{exp}.npz"))
    roads = trim_roads(roads)                       # cruising regime only (same as 17)
    subject = np.array([r["subject"] for r in roads], "int64")
    tr, va, te = gen_split(subject, seed=0)
    train_roads = [r for r, m in zip(roads, tr) if m]
    test_roads = [r for r, m in zip(roads, te) if m] or [r for r, m in zip(roads, va) if m]
    print(f"[{exp}] test roads={len(test_roads)}", flush=True)

    # human reference stats on the same roads
    hum_sdlp = np.array([float(np.std(r["e_ref"])) for r in test_roads])
    hum_v = np.array([float(np.mean(r["v_ref"])) for r in test_roads])

    # drivers
    print("PD (env validity):", flush=True)
    pd_rows, pd_agg = eval_driver(test_roads, dd, lambda o, e: pd_action(e), "PD")

    print("env-native BC baseline:", flush=True)
    bcnet = train_env_bc(train_roads, dd)
    def bc_policy(obs, env):
        with torch.no_grad():
            return bcnet(torch.from_numpy(obs).unsqueeze(0)).numpy()[0]
    bc_rows, bc_agg = eval_driver(test_roads, dd, bc_policy, "BC")

    print("RL (PPO):", flush=True)
    model = PPO.load(os.path.join(ART, f"rl_{exp}.zip"), device="cpu")
    rl_rows, rl_agg = eval_driver(test_roads, dd,
                                  lambda o, e: model.predict(o, deterministic=True)[0], "RL")

    # ---- RL-σ: policy-noise temperature α calibrated on VAL roads to match human SDLP ----
    # (deterministic eval collapses variability; the policy's own action noise, filtered by
    #  the closed-loop dynamics, produces smooth human-like weaving — we only tune its scale.)
    val_roads = [r for r, m in zip(roads, va) if m] or test_roads
    hum_sdlp_val = float(np.mean([np.std(r["e_ref"]) for r in val_roads]))
    orig_logstd = model.policy.log_std.data.clone()

    def set_temp(alpha):
        """Scale ONLY the steering-channel noise; accel stays ~deterministic (protects jerk —
        the CVAE lesson: longitudinal noise wrecks kinematic plausibility)."""
        ls = orig_logstd.clone()
        ls[0] = ls[0] + float(np.log(alpha))
        ls[1] = float(np.log(0.01))
        model.policy.log_std.data = ls

    def mean_sdlp(alpha, rset):
        set_temp(alpha)
        env_ = DrivingEnv(rset, dd=dd, record=True)
        vals = []
        for k in range(len(rset)):
            traj, _ = rollout(env_, lambda o, e: model.predict(o, deterministic=False)[0], k)
            if len(traj) > 10:
                vals.append(float(traj[:, 1].std()))
        return float(np.mean(vals)) if vals else float("nan")

    alphas = [0.4, 0.6, 0.8, 1.0, 1.2]
    curve = [mean_sdlp(a, val_roads) for a in alphas]
    alpha_star = float(np.clip(np.interp(hum_sdlp_val, curve, alphas), alphas[0], alphas[-1]))
    print(f"  [cal] val human SDLP={hum_sdlp_val:.3f}  curve={np.round(curve,3).tolist()}  "
          f"alpha*={alpha_star:.2f}", flush=True)

    print("RL-s (calibrated stochastic):", flush=True)
    set_temp(alpha_star)
    rls_rows, rls_agg = eval_driver(test_roads, dd,
                                    lambda o, e: model.predict(o, deterministic=False)[0], "RL-s")
    model.policy.log_std.data = orig_logstd

    # distribution match vs human (only completed roads)
    def w1_vs_human(rows, key, hum):
        vals = np.array([x[key] for x in rows if not x["off"]], float)
        return float(wasserstein1d(vals, hum)) if len(vals) else float("nan")
    dist = {name: dict(sdlp_w1=w1_vs_human(rows, "sdlp", hum_sdlp),
                       v_w1=w1_vs_human(rows, "v", hum_v))
            for name, rows in [("PD", pd_rows), ("BC", bc_rows), ("RL", rl_rows),
                               ("RL-s", rls_rows)]}

    # ------------------------------ figures ------------------------------
    # 1) trajectory overlay on one test road
    env = DrivingEnv(test_roads, dd=dd, record=True)
    k0 = 0
    r0 = test_roads[k0]
    grid0 = np.arange(len(r0["e_ref"])) * dd
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(grid0, r0["e_ref"], color="#185FA5", lw=1.4, label="사람(실제)")
    for nm, fn, col in [("RL", lambda o, e: model.predict(o, deterministic=True)[0], "#1D9E75"),
                        ("BC", bc_policy, "#D85A30")]:
        traj, off = rollout(env, fn, k0)
        ax.plot(traj[:, 0], traj[:, 1], color=col, lw=1.1, alpha=.9,
                label=f"{nm}{' (이탈)' if off else ''}")
    hw = float(np.mean(r0["lane_w"])) / 2
    ax.axhline(hw, ls=":", color="#888780"); ax.axhline(-hw, ls=":", color="#888780")
    ax.set_xlabel("거리(m)"); ax.set_ylabel("차선 offset(m)"); ax.legend()
    ax.set_title(f"닫힌루프 궤적: 사람 vs RL vs BC (test road, {exp})")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"fig_rl_traj_{exp}.png"), dpi=120); plt.close(fig)

    # 2) off-road rate bar
    fig, ax = plt.subplots(figsize=(5.5, 4))
    names = ["PD", "BC", "RL", "RL-s"]
    rates = [pd_agg["off_rate"], bc_agg["off_rate"], rl_agg["off_rate"], rls_agg["off_rate"]]
    ax.bar(names, rates, color=["#888780", "#D85A30", "#1D9E75", "#7F77DD"])
    for i, v in enumerate(rates):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    ax.set_ylabel("이탈율"); ax.set_ylim(0, 1.05); ax.set_title(f"닫힌루프 이탈율 ({exp})")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"fig_rl_offroad_{exp}.png"), dpi=120); plt.close(fig)

    # 3) SDLP distribution: human vs RL (completed only)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    rl_sdlp = [x["sdlp"] for x in rl_rows if not x["off"]]
    rls_sdlp = [x["sdlp"] for x in rls_rows if not x["off"]]
    bins = np.linspace(0, max(float(hum_sdlp.max()), max(rl_sdlp + rls_sdlp or [0.3])) * 1.1, 25)
    ax.hist(hum_sdlp, bins=bins, alpha=.5, density=True, label="사람", color="#185FA5")
    if rl_sdlp:
        ax.hist(rl_sdlp, bins=bins, alpha=.5, density=True, label="RL(결정론)", color="#1D9E75")
    if rls_sdlp:
        ax.hist(rls_sdlp, bins=bins, alpha=.5, density=True,
                label=f"RL-σ (α*={alpha_star:.2f})", color="#7F77DD")
    ax.set_xlabel("SDLP (m)"); ax.legend(); ax.set_title(f"SDLP 분포: 사람 vs RL ({exp})")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"fig_rl_sdlp_{exp}.png"), dpi=120); plt.close(fig)

    # ------------------------------ report ------------------------------
    summ = dict(exp=exp, n_test_roads=len(test_roads),
                PD=pd_agg, BC=bc_agg, RL=rl_agg, RLs=rls_agg,
                alpha_star=alpha_star, dist_vs_human=dist,
                human=dict(sdlp_mean=float(hum_sdlp.mean()), v_mean=float(hum_v.mean())))
    json.dump(summ, open(os.path.join(REP, f"eval_rl_{exp}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    L = [f"# RL 주행 에이전트 — 닫힌루프 평가 ({'SMOKE 자체검증' if exp=='smoke' else exp})\n",
         "> 간이 운동학 시뮬(파이썬 Gymnasium)에서 **에이전트가 직접 누적 주행**. "
         "held-out 피실험자 도로에서 PD(검증기준)·BC(모방)·RL(PPO) 비교.\n",
         "## 1. 환경 타당성 (PD가 사람 궤적을 추종하는가)\n",
         f"- PD 이탈율 **{pd_agg['off_rate']:.2f}**, 추종 RMSE **{pd_agg['rmse']:.3f} m** → "
         "시뮬이 기준궤적을 재현 가능함을 확인 후 비교 진행.\n",
         "## 2. 닫힌루프 비교 (test 도로)\n",
         "RL-σ = 정책 자체 행동노이즈의 온도 α를 **val 도로에서 사람 SDLP에 맞게 보정**한 확률적 평가 "
         f"(α*={alpha_star:.2f}, **조향 채널에만 적용** — 가속은 결정론 유지로 저크 보호). "
         "노이즈가 차량동역학(저역필터)을 통과해 *부드러운* 흔들림을 만든다.\n",
         "| 지표 | 사람 | PD | BC | RL(결정론) | RL-σ(보정) |", "|---|---|---|---|---|---|",
         f"| 이탈율 | 0 | {pd_agg['off_rate']:.2f} | {bc_agg['off_rate']:.2f} | {rl_agg['off_rate']:.2f} | **{rls_agg['off_rate']:.2f}** |",
         f"| SDLP(m) | **{hum_sdlp.mean():.3f}** | {pd_agg['sdlp']:.3f} | {bc_agg['sdlp']:.3f} | {rl_agg['sdlp']:.3f} | **{rls_agg['sdlp']:.3f}** |",
         f"| 평균속도(m/s) | {hum_v.mean():.1f} | {pd_agg['v']:.1f} | {bc_agg['v']:.1f} | {rl_agg['v']:.1f} | {rls_agg['v']:.1f} |",
         f"| RMSE(e-e_ref) | — | {pd_agg['rmse']:.3f} | {bc_agg['rmse']:.3f} | {rl_agg['rmse']:.3f} | {rls_agg['rmse']:.3f} |",
         f"| 저크위반율 | — | {pd_agg['jviol']:.3f} | {bc_agg['jviol']:.3f} | {rl_agg['jviol']:.3f} | {rls_agg['jviol']:.3f} |",
         f"\n주의 — **개인추종 RMSE의 바닥**: e_ref(개인 궤적)를 관측에서 숨긴 설계에서, 특정 개인과의 "
         f"RMSE는 그 사람 고유 흔들림(SDLP≈{hum_sdlp.mean():.2f}m) 아래로 원리적으로 내려갈 수 없다. "
         "따라서 인간유사성의 주지표는 RMSE가 아니라 **분포일치(SDLP·속도 W1)**다.",
         "\n분포일치(Wasserstein, 완주 도로만): "
         + ", ".join(f"{n} SDLP-W1={dist[n]['sdlp_w1']:.3g}/속도-W1={dist[n]['v_w1']:.3g}" for n in names),
         "\n## 3. 그림\n"]
    for fn, cap in [(f"fig_rl_traj_{exp}.png", "닫힌루프 궤적 오버레이"),
                    (f"fig_rl_offroad_{exp}.png", "이탈율 비교"),
                    (f"fig_rl_sdlp_{exp}.png", "SDLP 분포 (사람 vs RL)")]:
        L.append(f"**{cap}**\n\n![{cap}](figs/{fn})\n")
    L += ["## 4. 정직한 한계\n",
          "- **간이 운동학 시뮬** 기반: 절대 물리충실도가 아니라 *상대 비교·안정성 검증*용 "
          "(PD 타당성 검증 통과가 전제).",
          "- BC 기준선은 **env-native BC**(같은 관측→행동을 지도학습): 원 GRU BC와 특징 인터페이스가 "
          "달라 직접 이식이 불공정하므로, *같은 인터페이스에서 모방 vs RL*을 비교한 것.",
          "- human-like 보상은 특정 사람 궤적 추종 → 행태 다양성은 제한(후속: 분포기반/GAIL).",
          "- 실제 시뮬레이터(SCANeR) 닫힌루프 검증은 프로그램 제어 확보 시 별도 필요.\n"]
    open(os.path.join(REP, "report_rl.md"), "w", encoding="utf-8").write("\n".join(L))
    print("wrote report_rl.md + figs + eval_rl_%s.json" % exp, flush=True)


if __name__ == "__main__":
    main()
