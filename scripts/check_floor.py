# -*- coding: utf-8 -*-
"""게이트 검사: 남산 코너링 SDLP 바닥이 조향 기반 교체(rl_2024_wide→rl_multi)로 내려가는가."""
import os, importlib
import numpy as np
from stable_baselines3 import PPO
from common import ART, CACHE
from driving_env import DrivingEnv, load_roads, rollout, trim_roads

p20 = importlib.import_module("20_profile_eval")
p22 = importlib.import_module("22_v3_spectral")
p34 = importlib.import_module("34_virtual_cohort")

ch = p34.load_champion()
roads, _, dd = load_roads(os.path.join(CACHE, "env_roads_namsan.npz"))
roads = trim_roads(roads)[:6]
for r in roads:
    r["e_ref"] = np.zeros_like(r["v_ref"])
env = DrivingEnv(roads, dd=dd, record=True, steer_gain=0.012)

for name in ["rl_2024_wide.zip", "rl_multi.zip"]:
    model = PPO.load(os.path.join(ART, name), device="cpu")
    for sig in [0.02, 0.232]:
        vals, offs = [], 0
        for k in range(len(roads)):
            pol = p22.SpectralPolicy(model, ch["fr"], ch["A"], sig, lib=ch["lib"], seed=k)
            pol.reset()
            traj, off = rollout(env, pol, k)
            offs += int(off)
            if len(traj) > 60:
                vals.append(float(np.std(p20.rl_signals(traj, gain=0.012)["e"])))
        print(f"{name:20s} sigma={sig}: SDLP {np.mean(vals):.3f} off {offs}/{len(roads)}",
              flush=True)
