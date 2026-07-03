# -*- coding: utf-8 -*-
"""
17_train_rl.py  --  train a PPO driving agent in the surrogate closed-loop env.

  python 17_train_rl.py --smoke
  python 17_train_rl.py --exp 2024 --timesteps 400000

Subject-level split (common.gen_split on per-road subjects): agent trains on train-roads
only; 18_eval_rl evaluates on held-out test roads.
Saves: artifacts/rl_{exp}.zip, reports/metrics_rl_{exp}.json
"""
import os, json, time, argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from common import ART, REP, CACHE, gen_split
from driving_env import DrivingEnv, load_roads, rollout, trim_roads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="smoke")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--timesteps", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gain", type=float, default=None,
                    help="조향권한(action 1.0=이 곡률). 미지정시 RL_STEER_GAIN")
    ap.add_argument("--tag", default="", help="저장 접미사 (rl_{exp}{tag}.zip)")
    args = ap.parse_args()
    from common import RL_STEER_GAIN
    gain = args.gain if args.gain is not None else RL_STEER_GAIN
    exp = "smoke" if args.smoke else args.exp
    if args.smoke:
        args.timesteps = min(args.timesteps, 8_000)

    roads, subject, dd = load_roads(os.path.join(CACHE, f"env_roads_{exp}.npz"))
    roads = trim_roads(roads)                       # cruising regime only
    subject = np.array([r["subject"] for r in roads], "int64")
    tr, va, te = gen_split(subject, seed=0)
    train_roads = [r for r, m in zip(roads, tr) if m]
    val_roads = [r for r, m in zip(roads, va) if m]
    print(f"[{exp}] roads train={len(train_roads)} val={len(val_roads)} test={int(te.sum())} "
          f"(subjects total {len(set(subject.tolist()))})", flush=True)

    mon = Monitor(DrivingEnv(train_roads, dd=dd, random_start=True, seed=args.seed,
                             steer_gain=gain))
    print(f"steer_gain={gain}", flush=True)
    # v4c: reward/return normalization (norm_obs=False — obs는 이미 수동 정규화;
    # 보상 정규화는 학습에만 작용하므로 eval 스크립트들의 predict 경로는 무변경)
    env = VecNormalize(DummyVecEnv([lambda: mon]), norm_obs=False, norm_reward=True,
                       gamma=0.995)
    model = PPO("MlpPolicy", env, device="cpu", seed=args.seed, verbose=0,
                n_steps=2048, batch_size=256, learning_rate=3e-4, gamma=0.995,
                gae_lambda=0.95, ent_coef=1e-3,
                policy_kwargs=dict(net_arch=[64, 64]))
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    dt = time.time() - t0
    print(f"trained {args.timesteps:,} steps in {dt:.0f}s ({args.timesteps/max(dt,1):.0f} fps)", flush=True)
    model.save(os.path.join(ART, f"rl_{exp}{args.tag}.zip"))

    # quick val rollout (deterministic)
    ev = DrivingEnv(val_roads or train_roads, dd=dd, record=True, steer_gain=gain)
    rets, offs, sdlps = [], 0, []
    for k in range(len(ev.roads)):
        traj, off = rollout(ev, lambda o, e: model.predict(o, deterministic=True)[0], k)
        offs += int(off)
        if len(traj):
            sdlps.append(float(traj[:, 1].std()))
    ep = mon.get_episode_rewards()          # raw (unnormalized) episode rewards from Monitor
    rep = dict(exp=exp, timesteps=args.timesteps, seconds=dt, steer_gain=gain,
               n_train_roads=len(train_roads), n_val_roads=len(ev.roads),
               train_ep_reward_first10=float(np.mean(ep[:10])) if len(ep) >= 10 else None,
               train_ep_reward_last10=float(np.mean(ep[-10:])) if len(ep) >= 10 else None,
               val_offroad=offs, val_offroad_rate=offs / len(ev.roads),
               val_sdlp_mean=float(np.mean(sdlps)) if sdlps else None)
    json.dump(rep, open(os.path.join(REP, f"metrics_rl_{exp}{args.tag}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"episode reward: first10={rep['train_ep_reward_first10']} -> last10={rep['train_ep_reward_last10']}")
    print(f"val: offroad {offs}/{len(ev.roads)}  SDLP mean={rep['val_sdlp_mean']}")
    print("saved -> artifacts/rl_%s.zip + metrics_rl_%s.json" % (exp, exp), flush=True)


if __name__ == "__main__":
    main()
