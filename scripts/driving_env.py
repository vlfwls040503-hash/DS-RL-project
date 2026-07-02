# -*- coding: utf-8 -*-
"""
driving_env.py  --  surrogate closed-loop driving environment (kinematic, distance-grid roads).

Road dict (float32 arrays on the dd distance grid, from common.reindex_run or make_smoke_roads):
  curv, slope, lane_w, cw, e_ref, v_ref  (+ subject, cond)
State: s (m along road), v (m/s), e (lane offset m, +left), psi (heading error rad).
Action: [steer, accel] in [-1,1]^2 -> kappa_cmd = RL_STEER_GAIN*steer, a = RL_A_MAX*accel.
Reward (human-like): -w_e*(e-e_ref)^2 - w_v*(v-v_ref)^2 - w_j*jerk^2 - w_a*a^2 + alive
                     - offroad penalty (terminates).

Self-test:  python driving_env.py   (PD controller tracks the human ref on smoke roads;
                                     validates the surrogate sim before any RL.)
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from common import (GEN_DD, RL_DT, RL_V_MAX, RL_A_MAX, RL_STEER_GAIN, RL_LOOKAHEAD,
                    RL_LOOKAHEAD_GRID, RL_MARGIN, RL_MAX_STEPS,
                    RL_W_E, RL_W_V, RL_W_J, RL_W_A, RL_W_DS, RL_ALIVE, RL_OFFROAD_PEN,
                    RL_OFFROAD_STEP, RL_RATE_ACTION, RL_SRATE_MAX, make_smoke_roads)

# obs: [v, v_ref, e, psi, lane_halfwidth, slope] + lookahead curv + (rate mode) current steer
OBS_DIM = 6 + RL_LOOKAHEAD + (1 if RL_RATE_ACTION else 0)
# NOTE: v_ref (speed intent) IS observed — a real driver knows the target speed.
#       e_ref stays HIDDEN: lateral behavior must be inferred from geometry.
#       v4b rate mode: current steering appended (integrator state must be observable).


def build_obs(road, i, v, e, psi, vref_scale=1.0, steer=0.0):
    """Shared observation builder (env + expert-dataset use the SAME formula)."""
    M = len(road["curv"])
    i = min(max(i, 0), M - 1)
    base = [v / 30.0, road["v_ref"][i] * vref_scale / 30.0, e, psi * 5.0,
            road["lane_w"][i] / 2.0, road["slope"][i] * 20.0]
    idx = np.clip(i + RL_LOOKAHEAD_GRID * np.arange(1, RL_LOOKAHEAD + 1), 0, M - 1)
    tail = [steer] if RL_RATE_ACTION else []
    return np.concatenate([base, road["curv"][idx] * 200.0, tail]).astype(np.float32)


class DrivingEnv(gym.Env):
    """Kinematic lane-frame driving env over a set of recorded roads."""
    metadata = {"render_modes": []}

    def __init__(self, roads, dd=GEN_DD, record=False, random_start=False, seed=0):
        super().__init__()
        assert len(roads) > 0
        self.roads, self.dd = roads, float(dd)
        self.record, self.random_start = record, random_start
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self._rng = np.random.RandomState(seed)

    # ---------------------------------------------------------------- reset
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        idx = None
        if options and "road_idx" in options:
            idx = int(options["road_idx"])
        if idx is None:
            idx = self._rng.randint(len(self.roads))
        self.road = self.roads[idx]
        self.road_idx = idx
        M = len(self.road["curv"])
        self.s = 0.0
        if self.random_start:                       # long roads: train on random segments
            max_start = max(0.0, (M - 1) * self.dd - RL_MAX_STEPS * RL_DT * 30.0)
            if max_start > 0:
                self.s = float(self._rng.uniform(0.0, max_start))
        i0 = min(int(self.s / self.dd), M - 1)
        self.e = float(self.road["e_ref"][i0])
        self.v = float(max(self.road["v_ref"][i0], 0.1))
        self.psi = 0.0
        self.vref_scale = 1.0
        if self.random_start:      # state randomization -> policy learns recovery (robustness)
            self.e += float(self._rng.uniform(-0.15, 0.15))
            self.v = float(np.clip(self.v * self._rng.uniform(0.7, 1.25), 0.5, RL_V_MAX))
            self.psi = float(self._rng.uniform(-0.01, 0.01))
            # NOTE: v_ref-scale augmentation was tried here (U(0.85,1.15)) to make the policy
            # obey the v_ref channel, but it destabilized the equilibrium (lateral noise curve
            # collapsed, speed tracking broke) -> reverted. Proper fix: multi-speed-regime
            # training data (e.g. merge), not augmentation. vref_scale stays 1.0.
        self.prev_a = 0.0
        self.prev_steer = 0.0
        # v4b integrator state: start matched to road curvature (no initial transient)
        self.steer_state = float(np.clip(self.road["curv"][i0] / RL_STEER_GAIN, -1.0, 1.0))
        self.steps = 0
        self.was_offroad = False
        self.traj = []
        return build_obs(self.road, i0, self.v, self.e, self.psi, self.vref_scale,
                         self.steer_state), {}

    # ----------------------------------------------------------------- step
    def step(self, action):
        if RL_RATE_ACTION:            # v4b: action[0] = steering RATE; env integrates
            rate = float(np.clip(action[0], -1.0, 1.0))
            self.steer_state = float(np.clip(self.steer_state + rate * RL_SRATE_MAX * RL_DT,
                                             -1.0, 1.0))
            steer = self.steer_state
        else:
            steer = float(np.clip(action[0], -1.0, 1.0))
        a = float(np.clip(action[1], -1.0, 1.0)) * RL_A_MAX
        kappa_cmd = steer * RL_STEER_GAIN
        r = self.road
        M = len(r["curv"])
        i = min(int(self.s / self.dd), M - 1)
        kr = float(r["curv"][i])

        # kinematic lane-frame update
        self.psi += self.v * (kappa_cmd - kr) * RL_DT
        self.e += self.v * np.sin(self.psi) * RL_DT
        self.v = float(np.clip(self.v + a * RL_DT, 0.0, RL_V_MAX))
        self.s += self.v * np.cos(self.psi) * RL_DT
        self.steps += 1

        e_ref = float(r["e_ref"][i]); v_ref = float(r["v_ref"][i]) * self.vref_scale
        jerk = (a - self.prev_a) / RL_DT
        rew = (-RL_W_E * (self.e - e_ref) ** 2 - RL_W_V * (self.v - v_ref) ** 2
               - RL_W_J * jerk ** 2 - RL_W_A * a ** 2
               - RL_W_DS * (steer - self.prev_steer) ** 2 + RL_ALIVE)
        self.prev_a = a
        self.prev_steer = steer

        half = float(r["lane_w"][i]) / 2.0
        bound = half + RL_MARGIN
        offroad = bool(abs(self.e) > bound)
        if offroad:                              # no-escape: clamp back + penalize, keep going
            if not self.was_offroad:
                rew -= RL_OFFROAD_PEN            # one-time on entering
            rew -= RL_OFFROAD_STEP
            self.e = float(np.sign(self.e) * bound)
            self.psi *= 0.5                      # damp heading to allow recovery
        self.was_offroad = offroad
        terminated = False
        truncated = bool(self.s >= (M - 1) * self.dd or self.steps >= RL_MAX_STEPS)
        if self.record:   # (s, e, v, a, psi, steer) — psi/steer feed the multi-signal profile eval
            self.traj.append((self.s, self.e, self.v, a, self.psi, steer))
        i2 = min(int(self.s / self.dd), M - 1)
        obs = build_obs(r, i2, self.v, self.e, self.psi, self.vref_scale,
                        self.steer_state if RL_RATE_ACTION else steer)
        return obs, float(rew), terminated, truncated, dict(e_ref=e_ref, v_ref=v_ref, offroad=offroad)


# ======================================================================
# Road cache I/O (ragged roads stored as concatenated arrays + ptr)
# ======================================================================
ROAD_CHANS = ["curv", "slope", "lane_w", "cw", "e_ref", "v_ref"]


def save_roads(path, roads, dd=GEN_DD):
    ptr = np.cumsum([0] + [len(r["curv"]) for r in roads]).astype("int64")
    data = {c: np.concatenate([np.asarray(r[c], np.float32) for r in roads]) for c in ROAD_CHANS}
    np.savez_compressed(path, ptr=ptr, dd=dd,
                        subject=np.array([r.get("subject", 0) for r in roads], "int64"),
                        cond=np.array([r.get("cond", 0) for r in roads], "int8"), **data)


def load_roads(path):
    d = np.load(path)
    ptr = d["ptr"]
    roads = []
    for k in range(len(ptr) - 1):
        a, b = int(ptr[k]), int(ptr[k + 1])
        r = {c: d[c][a:b] for c in ROAD_CHANS}
        r["subject"] = int(d["subject"][k]); r["cond"] = int(d["cond"][k])
        roads.append(r)
    return roads, d["subject"].astype("int64"), float(d["dd"])


def trim_roads(roads, v_min=5.0, min_pts=200):
    """Cut low-speed launch/stop segments (real runs start from standstill) so that
    training/eval/human comparison all happen in the cruising regime."""
    out = []
    for r in roads:
        ok = np.where(np.asarray(r["v_ref"]) >= v_min)[0]
        if len(ok) < min_pts:
            continue
        a, b = int(ok[0]), int(ok[-1]) + 1
        r2 = {c: np.ascontiguousarray(r[c][a:b]) for c in ROAD_CHANS}
        r2["subject"], r2["cond"] = r["subject"], r["cond"]
        out.append(r2)
    return out


# ======================================================================
# PD controller (privileged: sees the human ref) — env validity check
# ======================================================================
def pd_action(env, k_e=0.10, psi_max=0.12, k_psi=0.6, k_v=1.0, preview=4):
    r = env.road
    M = len(r["curv"])
    i = min(int(env.s / env.dd), M - 1)
    j = min(i + preview, M - 1)
    e_ref = float(r["e_ref"][j]); v_ref = float(r["v_ref"][i])
    kr = float(r["curv"][i])
    psi_des = float(np.clip(k_e * (e_ref - env.e), -psi_max, psi_max))
    kappa_cmd = kr + k_psi * (psi_des - env.psi)
    steer_tgt = float(np.clip(kappa_cmd / RL_STEER_GAIN, -1.0, 1.0))
    if RL_RATE_ACTION:                 # convert target angle into a rate command
        cmd = float(np.clip((steer_tgt - env.steer_state) / (RL_SRATE_MAX * RL_DT), -1.0, 1.0))
    else:
        cmd = steer_tgt
    acc = float(np.clip(k_v * (v_ref - env.v) / RL_A_MAX, -1.0, 1.0))
    return np.array([cmd, acc], np.float32)


def rollout(env, policy_fn, road_idx):
    """Roll one episode. policy_fn(obs, env)->action. Env must have record=True.
    Returns (traj [T,4]=(s,e,v,a), offroad:bool)."""
    obs, _ = env.reset(options={"road_idx": road_idx})
    done, off = False, False
    while not done:
        obs, _, term, trunc, info = env.step(policy_fn(obs, env))
        off = off or info["offroad"]
        done = term or trunc
    return np.asarray(env.traj, np.float32), off


# ======================================================================
# Env-native expert dataset (for the closed-loop BC baseline in 18)
# ======================================================================
def _smooth(x, w=9):
    k = np.ones(w) / w
    return np.convolve(np.pad(x, (w // 2, w // 2), mode="edge"), k, mode="valid")[:len(x)]


def make_expert_dataset(roads, dd=GEN_DD):
    """Derive env-native expert (obs, action) pairs from human refs.
    psi_h = de/ds; kappa_h = curv + d²e/ds²; a_h = v dv/ds."""
    X, Y = [], []
    for r in roads:
        e = _smooth(np.asarray(r["e_ref"], np.float64))
        v = _smooth(np.asarray(r["v_ref"], np.float64))
        psi_h = np.gradient(e, dd)
        kappa_h = np.asarray(r["curv"], np.float64) + np.gradient(psi_h, dd)
        a_h = v * np.gradient(v, dd)
        steer_h = np.clip(kappa_h / RL_STEER_GAIN, -1, 1)
        acc_h = np.clip(a_h / RL_A_MAX, -1, 1)
        if RL_RATE_ACTION:
            rate_h = np.clip(np.gradient(steer_h, dd) * v / RL_SRATE_MAX, -1, 1)
        for i in range(0, len(e), 2):
            if RL_RATE_ACTION:
                X.append(build_obs(r, i, v[i], e[i], psi_h[i], steer=steer_h[i]))
                Y.append([rate_h[i], acc_h[i]])
            else:
                X.append(build_obs(r, i, v[i], e[i], psi_h[i]))
                Y.append([steer_h[i], acc_h[i]])
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)


# ======================================================================
# Self-test: PD tracks human refs on smoke roads (sim validity)
# ======================================================================
if __name__ == "__main__":
    roads = make_smoke_roads(n=6, seed=0)
    env = DrivingEnv(roads, record=True)
    print("PD self-test on smoke roads (surrogate-sim validity):")
    rmses, offs = [], 0
    for k in range(len(roads)):
        traj, off = rollout(env, lambda o, e: pd_action(e), k)
        r = roads[k]
        grid = np.arange(len(r["e_ref"])) * GEN_DD
        e_ref_i = np.interp(traj[:, 0], grid, r["e_ref"])
        v_ref_i = np.interp(traj[:, 0], grid, r["v_ref"])
        rmse_e = float(np.sqrt(np.mean((traj[:, 1] - e_ref_i) ** 2)))
        rmse_v = float(np.sqrt(np.mean((traj[:, 2] - v_ref_i) ** 2)))
        sdlp_pd = float(traj[:, 1].std()); sdlp_h = float(np.std(r["e_ref"]))
        rmses.append(rmse_e); offs += int(off)
        print(f"  road{k}: steps={len(traj):4d} offroad={off}  RMSE_e={rmse_e:.3f}m "
              f"RMSE_v={rmse_v:.2f}  SDLP pd={sdlp_pd:.3f} vs human={sdlp_h:.3f}")
    ok = (max(rmses) < 0.15) and (offs == 0)
    print(f"=> mean RMSE_e={np.mean(rmses):.3f}m, offroad={offs}/{len(roads)}  "
          f"{'VALID' if ok else 'NEEDS GAIN TUNING'}")
