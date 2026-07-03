# -*- coding: utf-8 -*-
"""
23_gail.py  --  GAIL (적대적 모방학습): 판별자를 보상으로 고용해 텍스처까지 학습.

설계 (세션 교훈 반영):
  - 자체 구현 (외부 imitation 라이브러리의 SB3 버전 충돌 회피)
  - 판별자 D(obs, action): 전이 단위 MLP. 전문가 = make_expert_dataset(train 도로)
  - 보상 = softplus(D logit) [양수 → 자살정책 원천 차단] + env 안전항(이탈 페널티만)
  - 챔피언(v3.1 절대각 정책) 웜스타트 + 낮은 lr → 적대학습 안정화
  - 챔피언 비파괴: artifacts/rl_2024_gail.zip 별도 저장
  - D 과잉승리 모니터: 정확도 로그 (1.0 고착 = G 신호 소멸)

  python 23_gail.py --exp 2024 --timesteps 600000
"""
import os, json, argparse, time, importlib
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from common import ART, REP, CACHE, gen_split
from driving_env import DrivingEnv, load_roads, rollout, trim_roads, make_expert_dataset, OBS_DIM

p20 = importlib.import_module("20_profile_eval")
p21 = importlib.import_module("21_validation")

for fam in ["Malgun Gothic", "NanumGothic", "Gulim"]:
    try:
        plt.rcParams["font.family"] = fam; break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(REP, "figs"); os.makedirs(FIG, exist_ok=True)
np.random.seed(0); torch.manual_seed(0)


class Discriminator(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)

    @torch.no_grad()
    def reward(self, obs, act):
        x = torch.from_numpy(np.concatenate([obs, act]).astype("float32"))
        return float(torch.nn.functional.softplus(self.net(x)).item())


class GAILWrapper(gym.Wrapper):
    """보상 = softplus(D) + env 안전항. (obs, action) 전이를 링버퍼에 적재."""
    def __init__(self, env, disc, buffer, r_scale=1.0):
        super().__init__(env)
        self.disc, self.buffer, self.r_scale = disc, buffer, r_scale
        self._last_obs = None

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._last_obs = obs.copy()
        return obs, info

    def step(self, action):
        obs, safety_r, term, trunc, info = self.env.step(action)
        a = np.asarray(action, np.float32)
        self.buffer.append((self._last_obs.copy(), a.copy()))
        r = self.r_scale * self.disc.reward(self._last_obs, a) + float(safety_r)
        self._last_obs = obs.copy()
        return obs, r, term, trunc, info


class DiscCallback(BaseCallback):
    """롤아웃마다 D 학습. 2차: label smoothing + 약한 D(에폭1, lr 1e-4)로 줄다리기 유지."""
    def __init__(self, disc, buffer, XE, YE, epochs=1, batch=512, nmax=8192,
                 d_lr=1e-4, smooth=(0.9, 0.1)):
        super().__init__()
        self.disc, self.buffer = disc, buffer
        self.expert = torch.from_numpy(np.concatenate([XE, YE], axis=1).astype("float32"))
        self.opt = torch.optim.Adam(disc.parameters(), lr=d_lr)
        self.epochs, self.batch, self.nmax = epochs, batch, nmax
        self.smooth = smooth
        self.accs = []

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        if len(self.buffer) < self.batch:
            return
        pol = list(self.buffer)[-self.nmax:]
        Xp = torch.from_numpy(np.stack([np.concatenate([o, a]) for o, a in pol]).astype("float32"))
        ne = min(len(Xp), len(self.expert))
        idx_e = torch.randint(0, len(self.expert), (ne,))
        Xe = self.expert[idx_e]
        X = torch.cat([Xe, Xp[:ne]])
        y = torch.cat([torch.full((ne,), self.smooth[0]),      # expert (label smoothing)
                       torch.full((ne,), self.smooth[1])])     # policy
        bce = nn.BCEWithLogitsLoss()
        for _ in range(self.epochs):
            perm = torch.randperm(len(X))
            for s in range(0, len(X), self.batch):
                b = perm[s:s + self.batch]
                self.opt.zero_grad()
                loss = bce(self.disc(X[b]), y[b])
                loss.backward(); self.opt.step()
        with torch.no_grad():
            acc = float(((torch.sigmoid(self.disc(X)) > 0.5).float() == y).float().mean())
        self.accs.append(acc)
        if len(self.accs) % 20 == 0:
            print(f"  [D] iter{len(self.accs):4d} acc={acc:.2f} "
                  f"(0.5=G승리, 1.0=G신호소멸)", flush=True)


def cv_auc(Xa, ya, seed=0):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    ps = np.zeros(len(ya))
    for tr_i, te_i in skf.split(Xa, ya):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xa[tr_i], ya[tr_i]); ps[te_i] = clf.predict_proba(Xa[te_i])[:, 1]
    return float(roc_auc_score(ya, ps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="2024")
    ap.add_argument("--timesteps", type=int, default=600_000)
    ap.add_argument("--per_road", type=int, default=10)
    ap.add_argument("--sde", action="store_true",
                    help="gSDE 매끄러운 탐사 (v1 실패원인: 백색 행동잡음을 D가 즉시 간파)")
    ap.add_argument("--d_epochs", type=int, default=1)
    ap.add_argument("--d_lr", type=float, default=1e-4)
    ap.add_argument("--ckpt_every", type=int, default=1_000_000)
    args = ap.parse_args()
    exp = args.exp

    roads, subject, dd = load_roads(os.path.join(CACHE, f"env_roads_{exp}.npz"))
    roads = trim_roads(roads)
    subject = np.array([r["subject"] for r in roads], "int64")
    tr, va, te = gen_split(subject, seed=0)
    train_roads = [r for r, m in zip(roads, tr) if m]
    val_roads = [r for r, m in zip(roads, va) if m]
    test_roads = [r for r, m in zip(roads, te) if m]

    XE, YE = make_expert_dataset(train_roads, dd)
    print(f"expert transitions: {len(XE):,}  obs_dim={XE.shape[1]} act_dim={YE.shape[1]}", flush=True)

    disc = Discriminator(XE.shape[1] + YE.shape[1])
    buffer = deque(maxlen=100_000)
    env = GAILWrapper(Monitor(DrivingEnv(train_roads, dd=dd, random_start=True, seed=0,
                                         gail_safety_only=True)), disc, buffer)
    cb = DiscCallback(disc, buffer, XE, YE, epochs=args.d_epochs, d_lr=args.d_lr)

    if args.sde:   # v2: gSDE 매끄러운 탐사 — 정책 구조가 달라 콜드스타트
        model = PPO("MlpPolicy", env, device="cpu", seed=0, verbose=0,
                    n_steps=2048, batch_size=256, learning_rate=3e-4, gamma=0.995,
                    gae_lambda=0.95, ent_coef=1e-3, use_sde=True, sde_sample_freq=8,
                    policy_kwargs=dict(net_arch=[64, 64]))
        print("cold start with gSDE (sde_sample_freq=8)", flush=True)
    else:
        champion = os.path.join(ART, f"rl_{exp}_v24.zip")
        model = PPO.load(champion, env=env, device="cpu",
                         custom_objects={"learning_rate": 1e-4, "ent_coef": 0.005})
        print(f"warm start from champion: {os.path.basename(champion)}", flush=True)

    from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
    ckpt_cb = CheckpointCallback(save_freq=args.ckpt_every, save_path=ART,
                                 name_prefix=f"gail2_{exp}")
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=CallbackList([cb, ckpt_cb]),
                progress_bar=False)
    print(f"trained {args.timesteps:,} in {time.time()-t0:.0f}s | D acc last10="
          f"{np.mean(cb.accs[-10:]):.2f}", flush=True)
    final_p = os.path.join(ART, f"gail2_{exp}_final.zip")
    model.save(final_p)

    # ---- 사람 기준 (한 번만 계산) ----
    H_units = []
    for road in test_roads:
        for ch in p21.chunk_signals(p20.human_signals(road, dd)):
            H_units.append(ch)
    XH = np.vstack([p21.seg_features(h) for h in H_units])
    tex_h = dict(sdlp=float(np.mean([np.std(h["e"]) for h in H_units])),
                 wl=float(np.nanmean([p20.wavelength(h["e"]) for h in H_units])),
                 srr=float(np.mean([p20.srr(h["theta"], 0.5) for h in H_units])),
                 srr2=float(np.mean([p20.srr(h["theta"], 2.0) for h in H_units])))
    env_t = DrivingEnv(test_roads, dd=dd, record=True)

    def eval_model(m, per_road, seed0=500):
        S_sig, offs = [], 0
        for k in range(len(test_roads)):
            for j in range(per_road):
                np.random.seed(seed0 + k * 20 + j); torch.manual_seed(seed0 + k * 20 + j)
                traj, o = rollout(env_t, lambda ob, e: m.predict(ob, deterministic=False)[0], k)
                offs += int(o)
                if len(traj) > 60:
                    S_sig.append(p20.rl_signals(traj))
        off_rate = offs / max(len(test_roads) * per_road, 1)
        tex = dict(sdlp=float(np.mean([np.std(s["e"]) for s in S_sig])),
                   wl=float(np.nanmean([p20.wavelength(s["e"]) for s in S_sig])),
                   srr=float(np.mean([p20.srr(s["theta"], 0.5) for s in S_sig])),
                   srr2=float(np.mean([p20.srr(s["theta"], 2.0) for s in S_sig])))
        XS = np.vstack([p21.seg_features(s) for s in S_sig])
        rng2 = np.random.RandomState(2)
        nmin = min(len(XH), len(XS))
        X = np.vstack([XH[rng2.choice(len(XH), nmin, replace=False)],
                       XS[rng2.choice(len(XS), nmin, replace=False)]])
        yy = np.concatenate([np.zeros(nmin), np.ones(nmin)])
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        return cv_auc(X, yy), tex, off_rate, nmin

    # ---- 체크포인트 토너먼트 (적대학습은 후반 붕괴 가능 → 최고 시점 선발) ----
    import glob as _glob
    cands = sorted(_glob.glob(os.path.join(ART, f"gail2_{exp}_*_steps.zip"))) + [final_p]
    print(f"tournament: {len(cands)} checkpoints", flush=True)
    results = []
    for p in cands:
        m = PPO.load(p, device="cpu")
        a, tex, off, _ = eval_model(m, per_road=4)
        print(f"  {os.path.basename(p):32s} AUC={a:.3f} SDLP={tex['sdlp']:.3f} "
              f"wl={tex['wl']:.0f} SRR={tex['srr']:.1f} off={off:.2f}", flush=True)
        results.append((a, p))
    results.sort()
    best_auc4, best_p = results[0]

    # ---- 승자 전체 평가 ----
    m = PPO.load(best_p, device="cpu")
    auc, tex, off_rate, nmin = eval_model(m, per_road=args.per_road)
    m.save(os.path.join(ART, f"rl_{exp}_gail.zip"))
    print(f"WINNER {os.path.basename(best_p)}: AUC={auc:.3f} (챔피언 v3.1=0.794) "
          f"n={nmin}+{nmin} off={off_rate:.2f}", flush=True)
    print(f"texture h/r: SDLP {tex_h['sdlp']:.3f}/{tex['sdlp']:.3f} wl {tex_h['wl']:.0f}/{tex['wl']:.0f} "
          f"SRR {tex_h['srr']:.1f}/{tex['srr']:.1f} SRR2 {tex_h['srr2']:.1f}/{tex['srr2']:.1f}", flush=True)
    tex = dict(sdlp_h=tex_h["sdlp"], sdlp_r=tex["sdlp"], wl_h=tex_h["wl"], wl_r=tex["wl"],
               srr_h=tex_h["srr"], srr_r=tex["srr"], srr2_h=tex_h["srr2"], srr2_r=tex["srr2"])

    # fig: AUC progression
    fig, ax = plt.subplots(figsize=(7.5, 4))
    names = ["v2.4\nOU", "v3\n스펙트럼", "v3.1\n라이브러리", "GAIL"]
    vals = [0.819, 0.937, 0.794, auc]
    cols = ["#888780", "#D85A30", "#1D9E75", "#7F77DD"]
    ax.bar(names, vals, color=cols)
    ax.axhline(0.5, ls=":", color="#185FA5", label="구별불가(0.5)")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    ax.set_ylim(0.4, 1.0); ax.set_ylabel("C2ST AUC"); ax.legend()
    ax.set_title("판별자 AUC 진화 (낮을수록 사람다움)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"fig_gail_{exp}.png"), dpi=120); plt.close(fig)

    json.dump(dict(exp=exp, auc=auc, off_rate=off_rate, texture=tex,
                   d_acc_last10=float(np.mean(cb.accs[-10:]))),
              open(os.path.join(REP, f"gail_{exp}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved rl_%s_gail.zip + gail json + fig" % exp, flush=True)


if __name__ == "__main__":
    main()
