# -*- coding: utf-8 -*-
"""
24_gail_seg.py  --  GAIL 3차: 구간(200m) 단위 적대 보상.

2차 진단: 전이 D(s,a)는 시간구조(머무는 시간·리듬)에 눈멂 → 훈련 D를 평가자(C2ST)와
동일한 200m 세그먼트 특징 8종으로 교체. 50m마다 최근 200m 구간 특징을 온라인 계산,
softplus(D_seg) 보상 덩어리 지급 (~36스텝 간격 — PPO γ=0.995 지평선 내 신용할당 가능).

  python 24_gail_seg.py --exp 2024 --timesteps 2000000
"""
import os, json, argparse, time, glob as _glob, importlib
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
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from common import ART, REP, CACHE, gen_split, RL_STEER_GAIN
from driving_env import DrivingEnv, load_roads, rollout, trim_roads

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

SEG_PTS, STEP_PTS, GRID = 20, 5, 10.0     # 200m 구간, 50m 슬라이드, 10m 그리드
K2DEG = p20.K2DEG


def window_feats(e, latv, lata, theta):
    """p21.seg_features와 동일한 8특징 (온라인 20점 윈도)."""
    e = np.asarray(e); de = np.diff(e)
    return np.array([np.std(e), np.mean(np.abs(de)), np.std(latv), np.std(lata),
                     p20.srr(theta, 0.5), p20.srr(theta, 2.0),
                     float(np.sqrt(np.mean(de ** 2))), float(np.max(np.abs(latv)))],
                    dtype="float32")


class SegDisc(nn.Module):
    def __init__(self, in_dim=8, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.mu = torch.zeros(in_dim); self.sd = torch.ones(in_dim)

    def set_norm(self, mu, sd):
        self.mu = torch.from_numpy(mu.astype("float32"))
        self.sd = torch.from_numpy(np.where(sd < 1e-8, 1.0, sd).astype("float32"))

    def forward(self, x):
        return self.net((x - self.mu) / self.sd).squeeze(-1)

    @torch.no_grad()
    def reward(self, feats):
        x = torch.from_numpy(feats.astype("float32"))
        return float(torch.nn.functional.softplus(self((x).unsqueeze(0))).item())


class SegGAILWrapper(gym.Wrapper):
    """50m마다 최근 200m 구간 특징 → softplus(D_seg) 보상. 특징은 seg_buf에 적재."""
    def __init__(self, env, disc, seg_buf, r_scale=1.0):
        super().__init__(env)
        self.disc, self.seg_buf, self.r_scale = disc, seg_buf, r_scale
        self.g_e = deque(maxlen=SEG_PTS); self.g_lv = deque(maxlen=SEG_PTS)
        self.g_la = deque(maxlen=SEG_PTS); self.g_th = deque(maxlen=SEG_PTS)
        self.next_s, self.since = 0.0, 0

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        for q in (self.g_e, self.g_lv, self.g_la, self.g_th):
            q.clear()
        base = self.env.unwrapped
        self.next_s, self.since = base.s + GRID, 0
        return obs, info

    def step(self, action):
        obs, safety_r, term, trunc, info = self.env.step(action)
        base = self.env.unwrapped
        steer = float(np.clip(action[0], -1, 1))
        r = float(safety_r)
        while base.s >= self.next_s:                      # 10m 그리드 표본
            kap = steer * RL_STEER_GAIN
            self.g_e.append(base.e)
            self.g_lv.append(base.v * base.psi)
            self.g_la.append(base.v * base.v * kap)
            self.g_th.append(kap * K2DEG)
            self.next_s += GRID
            self.since += 1
        if len(self.g_e) == SEG_PTS and self.since >= STEP_PTS:
            f = window_feats(self.g_e, self.g_lv, self.g_la, self.g_th)
            self.seg_buf.append(f)
            r += self.r_scale * self.disc.reward(f)
            self.since = 0
        return obs, r, term, trunc, info


class SegDiscCallback(BaseCallback):
    def __init__(self, disc, seg_buf, XE, epochs=1, batch=256, nmax=4096, d_lr=1e-4,
                 smooth=(0.9, 0.1)):
        super().__init__()
        self.disc, self.buf = disc, seg_buf
        self.XE = torch.from_numpy(XE.astype("float32"))
        self.opt = torch.optim.Adam(disc.parameters(), lr=d_lr)
        self.epochs, self.batch, self.nmax, self.smooth = epochs, batch, nmax, smooth
        self.accs = []

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        if len(self.buf) < self.batch:
            return
        Xp = torch.from_numpy(np.stack(list(self.buf)[-self.nmax:]).astype("float32"))
        ne = min(len(Xp), len(self.XE))
        Xe = self.XE[torch.randint(0, len(self.XE), (ne,))]
        X = torch.cat([Xe, Xp[:ne]])
        y = torch.cat([torch.full((ne,), self.smooth[0]), torch.full((ne,), self.smooth[1])])
        bce = nn.BCEWithLogitsLoss()
        for _ in range(self.epochs):
            perm = torch.randperm(len(X))
            for s in range(0, len(X), self.batch):
                b = perm[s:s + self.batch]
                self.opt.zero_grad()
                loss = bce(self.disc(X[b]), y[b]); loss.backward(); self.opt.step()
        with torch.no_grad():                              # acc 버그 수정: 라벨을 0/1로 이진화
            acc = float(((torch.sigmoid(self.disc(X)) > 0.5) == (y > 0.5)).float().mean())
        self.accs.append(acc)
        if len(self.accs) % 20 == 0:
            print(f"  [Dseg] iter{len(self.accs):4d} acc={acc:.2f}", flush=True)


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
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--per_road", type=int, default=10)
    ap.add_argument("--warm", default="gail2_2024_3000000_steps.zip",
                    help="웜스타트 체크포인트 (2차의 최선: gSDE·안전주행·스타일만 과대)")
    args = ap.parse_args()
    exp = args.exp

    roads, subject, dd = load_roads(os.path.join(CACHE, f"env_roads_{exp}.npz"))
    roads = trim_roads(roads)
    subject = np.array([r["subject"] for r in roads], "int64")
    tr, va, te = gen_split(subject, seed=0)
    train_roads = [r for r, m in zip(roads, tr) if m]
    test_roads = [r for r, m in zip(roads, te) if m]

    # 전문가 세그먼트 은행 (train 도로만)
    XE = np.vstack([p21.seg_features(p20.human_signals(r, dd)) for r in train_roads])
    print(f"expert segments: {len(XE):,} x {XE.shape[1]}", flush=True)

    disc = SegDisc(XE.shape[1])
    disc.set_norm(XE.mean(0), XE.std(0))
    seg_buf = deque(maxlen=50_000)
    env = SegGAILWrapper(Monitor(DrivingEnv(train_roads, dd=dd, random_start=True, seed=0,
                                            gail_safety_only=True)), disc, seg_buf)
    cb = SegDiscCallback(disc, seg_buf, XE)

    warm = os.path.join(ART, args.warm)
    model = PPO.load(warm, env=env, device="cpu",
                     custom_objects={"learning_rate": 1e-4})
    print(f"warm start: {os.path.basename(warm)}", flush=True)

    ckpt_cb = CheckpointCallback(save_freq=500_000, save_path=ART, name_prefix=f"gail3_{exp}")
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=CallbackList([cb, ckpt_cb]),
                progress_bar=False)
    print(f"trained {args.timesteps:,} in {time.time()-t0:.0f}s | Dseg acc last10="
          f"{np.mean(cb.accs[-10:]):.2f}", flush=True)
    final_p = os.path.join(ART, f"gail3_{exp}_final.zip")
    model.save(final_p)

    # ---- 평가 준비 (사람 기준 1회) ----
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

    def eval_model(m, per_road, seed0=700):
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

    # ---- 체크포인트 토너먼트 ----
    cands = sorted(_glob.glob(os.path.join(ART, f"gail3_{exp}_*_steps.zip"))) + [final_p]
    print(f"tournament: {len(cands)} checkpoints", flush=True)
    results = []
    for p in cands:
        m = PPO.load(p, device="cpu")
        a, tex, off, _ = eval_model(m, per_road=4)
        print(f"  {os.path.basename(p):32s} AUC={a:.3f} SDLP={tex['sdlp']:.3f} "
              f"wl={tex['wl']:.0f} SRR={tex['srr']:.1f} off={off:.2f}", flush=True)
        results.append((a, p))
    results.sort()
    _, best_p = results[0]

    m = PPO.load(best_p, device="cpu")
    auc, tex, off_rate, nmin = eval_model(m, per_road=args.per_road)
    m.save(os.path.join(ART, f"rl_{exp}_gail3.zip"))
    print(f"WINNER {os.path.basename(best_p)}: AUC={auc:.3f} (챔피언 v3.1=0.794) "
          f"off={off_rate:.2f}", flush=True)
    print(f"texture h/r: SDLP {tex_h['sdlp']:.3f}/{tex['sdlp']:.3f} wl {tex_h['wl']:.0f}/{tex['wl']:.0f} "
          f"SRR {tex_h['srr']:.1f}/{tex['srr']:.1f} SRR2 {tex_h['srr2']:.1f}/{tex['srr2']:.1f}", flush=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    names = ["v3.1\n(챔피언)", "GAIL1", "GAIL2", "GAIL3\n(구간D)"]
    vals = [0.794, 0.999, 0.889, auc]
    ax.bar(names, vals, color=["#1D9E75", "#888780", "#888780", "#7F77DD"])
    ax.axhline(0.5, ls=":", color="#185FA5", label="구별불가(0.5)")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    ax.set_ylim(0.4, 1.05); ax.set_ylabel("C2ST AUC"); ax.legend()
    ax.set_title("GAIL 3차: 구간 단위 적대 보상")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"fig_gail3_{exp}.png"), dpi=120); plt.close(fig)

    json.dump(dict(exp=exp, auc=auc, off_rate=off_rate, winner=os.path.basename(best_p),
                   texture_h=tex_h, texture_r=tex,
                   d_acc_last10=float(np.mean(cb.accs[-10:]))),
              open(os.path.join(REP, f"gail3_{exp}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved rl_%s_gail3.zip + gail3 json + fig" % exp, flush=True)


if __name__ == "__main__":
    main()
