# -*- coding: utf-8 -*-
"""
31_speed_eval.py  --  B2 게이트: 다속도 정책의 속도 행동 검증.

핵심 질문: v_ref 채널이 드디어 살아났는가?
  ① γ 반응성 — 관측의 v_ref를 γ배로 스케일하면 실현 속도가 단조·비례로 따라오는가
    (2024 단일속도 정책은 채널 무시로 실패했던 시험)
  ② 속도 W1 — held-out 도로에서 실현 도로평균속도 분포 vs 사람
  ③ 기준선 — 단일속도 학습 rl_2024_wide를 왕숙에 이식했을 때의 속도 추종 오차 대비

  python 31_speed_eval.py
"""
import os, json, importlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from common import REP, ART, CACHE, gen_split, wasserstein1d
from driving_env import DrivingEnv, load_roads, trim_roads

p20 = importlib.import_module("20_profile_eval")

for fam in ["Malgun Gothic", "NanumGothic", "Gulim"]:
    try:
        plt.rcParams["font.family"] = fam; break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(REP, "figs"); os.makedirs(FIG, exist_ok=True)
GAIN_MULTI, GAIN_W = 0.012, 0.012
GAMMAS = [0.7, 0.85, 1.0, 1.15]


def run_det(model, env, k, gamma=1.0):
    obs, _ = env.reset(options={"road_idx": k})
    env.vref_scale = gamma
    i0 = min(int(env.s / env.dd), len(env.road["curv"]) - 1)
    from driving_env import build_obs
    obs = build_obs(env.road, i0, env.v, env.e, env.psi, gamma)
    done, off = False, False
    while not done:
        a = model.predict(obs, deterministic=True)[0]
        obs, _, term, trunc, info = env.step(a)
        off = off or info["offroad"]
        done = term or trunc
    traj = np.asarray(env.traj, np.float32)
    return traj, off


def main():
    roads, _, dd = load_roads(os.path.join(CACHE, "env_roads_multi.npz"))
    roads = trim_roads(roads)
    subj = np.array([r["subject"] for r in roads], "int64")
    tr, va, te = gen_split(subj, seed=0)
    test = [r for r, m in zip(roads, te) if m]
    test_w = [r for r in test if r["subject"] > 100]      # 왕숙
    test_h = [r for r in test if r["subject"] <= 100]     # 2024
    print(f"test roads: wangsuk {len(test_w)} / 2024 {len(test_h)}", flush=True)
    model = PPO.load(os.path.join(ART, "rl_multi.zip"), device="cpu")

    out = {}
    # ---- ① γ 반응성 ----
    resp = {}
    for name, rs in [("wangsuk", test_w), ("2024", test_h)]:
        env = DrivingEnv(rs, dd=dd, record=True, steer_gain=GAIN_MULTI)
        rows = []
        for g in GAMMAS:
            vs, tgt, offs = [], [], 0
            for k in range(len(rs)):
                traj, off = run_det(model, env, k, gamma=g)
                offs += int(off)
                if len(traj) > 60:
                    vs.append(float(np.mean(traj[:, 2])))
                    tgt.append(g * float(np.mean(rs[k]["v_ref"])))
            rows.append(dict(gamma=g, v=float(np.mean(vs)), target=float(np.mean(tgt)),
                             off=offs / len(rs)))
            print(f"  [{name}] γ={g}: v={np.mean(vs)*3.6:.0f} "
                  f"(목표 {np.mean(tgt)*3.6:.0f} km/h) off={offs}/{len(rs)}", flush=True)
        resp[name] = rows
        vseq = [r["v"] for r in rows]
        mono = bool(all(vseq[i] < vseq[i + 1] for i in range(len(vseq) - 1)))
        gain_ratio = (vseq[-1] - vseq[0]) / max(rows[-1]["target"] - rows[0]["target"], 1e-9)
        out[f"resp_{name}"] = dict(rows=rows, monotone=mono, gain_ratio=float(gain_ratio))
        print(f"  [{name}] 단조={mono} 반응비={gain_ratio:.2f} (1.0=완전추종)", flush=True)

    # ---- ② 속도 W1 (γ=1) ----
    for name, rs in [("wangsuk", test_w), ("2024", test_h)]:
        env = DrivingEnv(rs, dd=dd, record=True, steer_gain=GAIN_MULTI)
        v_pol, v_hum = [], []
        for k in range(len(rs)):
            traj, _ = run_det(model, env, k, gamma=1.0)
            if len(traj) > 60:
                v_pol.append(float(np.mean(traj[:, 2])))
                v_hum.append(float(np.mean(rs[k]["v_ref"])))
        w1 = wasserstein1d(np.array(v_hum), np.array(v_pol))
        mae = float(np.mean(np.abs(np.array(v_hum) - np.array(v_pol))))
        out[f"w1_{name}"] = dict(w1=float(w1), mae=mae,
                                 v_h=float(np.mean(v_hum)), v_p=float(np.mean(v_pol)))
        print(f"[{name}] 속도 W1={w1:.2f} MAE={mae:.2f} m/s "
              f"(사람 {np.mean(v_hum)*3.6:.0f} / 정책 {np.mean(v_pol)*3.6:.0f} km/h)", flush=True)

    # ---- ③ 기준선: 단일속도 정책을 왕숙에 (자기 게인 0.012, 커버 도로만) ----
    base = PPO.load(os.path.join(ART, "rl_2024_wide.zip"), device="cpu")
    rs = [r for r in test_w if float(np.abs(r["curv"]).max()) <= GAIN_W]
    if rs:
        env = DrivingEnv(rs, dd=dd, record=True, steer_gain=GAIN_W)
        errs = []
        for k in range(len(rs)):
            traj, _ = run_det(base, env, k, gamma=1.0)
            if len(traj) > 60:
                errs.append(abs(float(np.mean(traj[:, 2])) - float(np.mean(rs[k]["v_ref"]))))
        out["baseline_2024wide_on_wangsuk"] = dict(n=len(rs), mae=float(np.mean(errs)))
        print(f"[기준선 rl_2024_wide→왕숙] 속도 MAE={np.mean(errs):.2f} m/s (n={len(rs)})", flush=True)

    # ---- 그림 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    for name, c in [("wangsuk", "#7F77DD"), ("2024", "#888780")]:
        rows = resp[name]
        ax.plot([r["target"] * 3.6 for r in rows], [r["v"] * 3.6 for r in rows],
                "o-", color=c, label=f"{name} 실현속도")
    lim = ax.get_xlim()
    ax.plot(lim, lim, ":", color="#185FA5", label="완전 추종")
    ax.set_xlabel("γ·v_ref (km/h)"); ax.set_ylabel("실현 속도 (km/h)")
    ax.legend(); ax.set_title("γ 반응성 (v_ref 채널 생존 시험)")
    ax = axes[1]
    env = DrivingEnv(test_w, dd=dd, record=True, steer_gain=GAIN_MULTI)
    vh, vp = [], []
    for k in range(len(test_w)):
        traj, _ = run_det(model, env, k, gamma=1.0)
        if len(traj) > 60:
            vh.append(np.mean(test_w[k]["v_ref"]) * 3.6); vp.append(np.mean(traj[:, 2]) * 3.6)
    ax.scatter(vh, vp, s=18, color="#7F77DD")
    lim = [min(vh + vp) - 3, max(vh + vp) + 3]
    ax.plot(lim, lim, ":", color="#185FA5")
    ax.set_xlabel("사람 도로평균속도 (km/h)"); ax.set_ylabel("정책 (km/h)")
    ax.set_title("왕숙 test: 도로별 속도 추종")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_speed_multi.png"), dpi=120)
    plt.close(fig)
    json.dump(out, open(os.path.join(REP, "speed_multi.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved fig_speed_multi.png + speed_multi.json", flush=True)


if __name__ == "__main__":
    main()
