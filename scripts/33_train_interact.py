# -*- coding: utf-8 -*-
"""
33_train_interact.py  --  C2: 선행차 로그재생 환경에서 차간 상호작용 학습.

env 확장: 관측 +[차간거리/100, 상대속도/10] (OBS_DIM+2). 선행차는 실제 fv 로그의
시간-위치 트랙 재생(리더는 자차와 무관하게 제 갈 길 — 로그재생의 정직한 한계).
보상: 기존 human-like 항 그대로 (사람 v_ref 자체가 그 리더를 따라간 기록이므로
차간 유지 의도를 내포) + 근접 페널티(gap<5m).
평가: 차두시간(THW) 분포 W1 vs 사람, 추돌(<2m)율, 속도 W1, 이탈율.

  python 33_train_interact.py --timesteps 600000
"""
import os, json, argparse, importlib
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from common import ART, REP, gen_split, RL_DT, wasserstein1d
from driving_env import DrivingEnv, OBS_DIM

b32 = importlib.import_module("32_build_interact")

for fam in ["Malgun Gothic", "NanumGothic", "Gulim"]:
    try:
        plt.rcParams["font.family"] = fam; break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(REP, "figs"); os.makedirs(FIG, exist_ok=True)
np.random.seed(0)
GAIN = 0.048
K_MAX = 0.02           # 여유율 법칙(권한=최대곡률×2.4)을 합류부 곡률 체급에 적용.
                       # v1의 0.05 실패는 여유 25배(과잉), 왕숙 붕괴는 1.0배(결핍) —
                       # 2.4배가 성립 조건이라는 법칙의 일반성 시험이기도 함.
GAP_FAR, PROX_M, PROX_PEN, CRASH_M = 200.0, 5.0, 5.0, 2.0


class LeaderEnv(gym.Wrapper):
    """DrivingEnv + 선행차 로그재생 관측/페널티."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM + 2,), np.float32)

    def _lead(self):
        base = self.env.unwrapped
        r = base.road
        t = base.steps * RL_DT
        if t >= r["leader_t"][-1]:
            return GAP_FAR, 0.0
        s_fv = float(np.interp(t, r["leader_t"], r["leader_s"]))
        v_fv = float(np.interp(t, r["leader_t"], r["leader_v"]))
        gap = s_fv - base.s
        if gap <= 0 or gap > GAP_FAR:          # 리더가 뒤/이탈 → 자유주행 취급
            return GAP_FAR, 0.0
        return gap, v_fv - base.v

    def _aug(self, obs):
        gap, relv = self._lead()
        self.last_gap = gap
        return np.concatenate([obs, [gap / 100.0, relv / 10.0]]).astype(np.float32)

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        return self._aug(obs), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        obs = self._aug(obs)
        info["gap"] = self.last_gap
        if self.last_gap < PROX_M:
            r -= PROX_PEN
        return obs, r, term, trunc, info


def rollout_lead(env, model, k, deterministic=True):
    obs, _ = env.reset(options={"road_idx": k})
    done, off, gaps, vs = False, False, [], []
    while not done:
        a = model.predict(obs, deterministic=deterministic)[0]
        obs, _, term, trunc, info = env.step(a)
        off = off or info["offroad"]
        gaps.append(info["gap"]); vs.append(env.env.unwrapped.v)
        done = term or trunc
    return np.array(gaps), np.array(vs), off


def human_thw(road, dd):
    """사람 THW: v_ref로 시간 적분해 자차 (t,s) 재구성 → 리더와 대조."""
    n = len(road["v_ref"])
    v = np.maximum(np.asarray(road["v_ref"], float), 0.5)
    t = np.concatenate([[0.0], np.cumsum(dd / v[:-1])])
    s = np.arange(n) * dd
    m = t < road["leader_t"][-1]
    if m.sum() < 50:
        return None
    s_fv = np.interp(t[m], road["leader_t"], road["leader_s"])
    gap = s_fv - s[m]
    ok = (gap > 0) & (gap < GAP_FAR)
    if ok.sum() < 50:
        return None
    return float(np.median(gap[ok] / v[m][ok]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=600_000)
    args = ap.parse_args()

    roads, dd = b32.load_interact()
    roads = [r for r in roads if float(np.abs(r["curv"]).max()) <= K_MAX
             and np.mean(np.asarray(r["v_ref"]) >= 5.0) > 0.8]
    subj = np.array([r["subject"] for r in roads], "int64")
    tr, va, te = gen_split(subj, seed=0)
    train_roads = [r for r, m in zip(roads, tr) if m]
    test_roads = [r for r, m in zip(roads, te) if m]
    print(f"interact roads(margin filter): {len(roads)} | train {len(train_roads)} "
          f"test {len(test_roads)}", flush=True)

    mon = Monitor(LeaderEnv(DrivingEnv(train_roads, dd=dd, random_start=True, seed=0,
                                       steer_gain=GAIN)))
    env = VecNormalize(DummyVecEnv([lambda: mon]), norm_obs=False, norm_reward=True,
                       gamma=0.995)
    model = PPO("MlpPolicy", env, device="cpu", seed=0, verbose=0,
                n_steps=2048, batch_size=256, learning_rate=3e-4, gamma=0.995,
                gae_lambda=0.95, ent_coef=1e-3, policy_kwargs=dict(net_arch=[64, 64]))
    import time as _t
    t0 = _t.time()
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    print(f"trained {args.timesteps:,} in {_t.time()-t0:.0f}s", flush=True)
    model.save(os.path.join(ART, "rl_interact.zip"))

    # ---- 평가: THW 분포 / 추돌 / 속도 / 이탈 ----
    env_t = LeaderEnv(DrivingEnv(test_roads, dd=dd, record=True, steer_gain=GAIN))
    thw_h, thw_p, crashes, offs, vW = [], [], 0, 0, []
    for k, road in enumerate(test_roads):
        th = human_thw(road, dd)
        gaps, vs, off = rollout_lead(env_t, model, k)
        offs += int(off)
        m = (gaps > 0) & (gaps < GAP_FAR) & (vs > 0.5)
        if th is not None and m.sum() > 50:
            thw_h.append(th)
            thw_p.append(float(np.median(gaps[m] / vs[m])))
        crashes += int((gaps < CRASH_M).any())
        vW.append((float(np.mean(vs)), float(np.mean(road["v_ref"]))))
    thw_h, thw_p = np.array(thw_h), np.array(thw_p)
    w1_thw = wasserstein1d(thw_h, thw_p)
    vp, vh = np.array([a for a, b in vW]), np.array([b for a, b in vW])
    res = dict(n_test=len(test_roads), n_thw=len(thw_h),
               thw_h_med=float(np.median(thw_h)), thw_p_med=float(np.median(thw_p)),
               w1_thw=float(w1_thw), crash_rate=crashes / len(test_roads),
               off_rate=offs / len(test_roads),
               v_mae=float(np.mean(np.abs(vp - vh))))
    print(f"THW 중앙값 사람 {res['thw_h_med']:.2f}s / 정책 {res['thw_p_med']:.2f}s "
          f"| W1={w1_thw:.2f}s | 추돌율 {res['crash_rate']:.2f} | 이탈율 {res['off_rate']:.2f} "
          f"| 속도 MAE {res['v_mae']:.2f} m/s", flush=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 6, 25)
    ax.hist(thw_h, bins=bins, alpha=0.6, label="사람", color="#185FA5")
    ax.hist(thw_p, bins=bins, alpha=0.6, label="정책", color="#7F77DD")
    ax.set_xlabel("도로별 중앙 차두시간 THW (s)"); ax.set_ylabel("도로 수")
    ax.legend(); ax.set_title(f"차간 상호작용: THW 분포 (W1={w1_thw:.2f}s)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_interact.png"), dpi=120)
    plt.close(fig)
    json.dump(res, open(os.path.join(REP, "interact.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved rl_interact.zip + interact.json + fig_interact.png", flush=True)


if __name__ == "__main__":
    main()
