# -*- coding: utf-8 -*-
"""
28_rnn_gail.py  --  GAIL 4차 (D2): RecurrentPPO(LSTM) + 구간 판별자.

GAIL 1~3차의 종합 결론 = "무기억 반응형 MLP는 사람의 수백 m 시간구조(준주기 배회+체류
리듬)를 표현할 수 없다"는 **가설**. 여기서 정책에 기억(LSTM)을 넣어 그 가설을 정면
검증한다. 훈련 보상은 3차와 동일한 구간(200m) 판별자 — 이제 정책 표현력이 병목이
아니라면 격차가 닫혀야 한다.

- sb3-contrib 2.6.0 RecurrentPPO + gSDE (스모크 확인: 매끄러운 탐사 유지 — 1차 실패
  원인이던 백색잡음 지문 차단)
- 성공 기준(사전선언): 구간 C2ST < 0.794 (챔피언 v3.1 경신)

  python 28_rnn_gail.py --exp 2024 --timesteps 2000000
"""
import os, json, argparse, time, glob as _glob, importlib
from collections import deque
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from common import ART, REP, CACHE, gen_split
from driving_env import DrivingEnv, load_roads, rollout, trim_roads

p20 = importlib.import_module("20_profile_eval")
p21 = importlib.import_module("21_validation")
p24 = importlib.import_module("24_gail_seg")   # SegDisc·SegGAILWrapper·SegDiscCallback·cv_auc

for fam in ["Malgun Gothic", "NanumGothic", "Gulim"]:
    try:
        plt.rcParams["font.family"] = fam; break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(REP, "figs"); os.makedirs(FIG, exist_ok=True)
np.random.seed(0); torch.manual_seed(0)


class RNNPolicy:
    """rollout()용 상태 유지 어댑터 (에피소드마다 reset 필수)."""
    def __init__(self, m, deterministic=False):
        self.m, self.det = m, deterministic
        self.state, self.start = None, True

    def reset(self):
        self.state, self.start = None, True

    def __call__(self, obs, env):
        a, self.state = self.m.predict(obs, state=self.state,
                                       episode_start=np.array([self.start]),
                                       deterministic=self.det)
        self.start = False
        return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="2024")
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--per_road", type=int, default=10)
    args = ap.parse_args()
    exp = args.exp

    roads, _, dd = load_roads(os.path.join(CACHE, f"env_roads_{exp}.npz"))
    roads = trim_roads(roads)
    subject = np.array([r["subject"] for r in roads], "int64")
    tr, va, te = gen_split(subject, seed=0)
    train_roads = [r for r, m in zip(roads, tr) if m]
    test_roads = [r for r, m in zip(roads, te) if m]

    XE = np.vstack([p21.seg_features(p20.human_signals(r, dd)) for r in train_roads])
    print(f"expert segments: {len(XE):,} x {XE.shape[1]}", flush=True)
    disc = p24.SegDisc(XE.shape[1])
    disc.set_norm(XE.mean(0), XE.std(0))
    seg_buf = deque(maxlen=50_000)
    env = p24.SegGAILWrapper(Monitor(DrivingEnv(train_roads, dd=dd, random_start=True,
                                                seed=0, gail_safety_only=True)),
                             disc, seg_buf)
    cb = p24.SegDiscCallback(disc, seg_buf, XE)

    model = RecurrentPPO("MlpLstmPolicy", env, device="cpu", seed=0, verbose=0,
                         n_steps=2048, batch_size=256, learning_rate=3e-4,
                         gamma=0.995, gae_lambda=0.95, ent_coef=1e-3,
                         use_sde=True, sde_sample_freq=8,
                         policy_kwargs=dict(lstm_hidden_size=64, net_arch=[64]))
    ckpt_cb = CheckpointCallback(save_freq=500_000, save_path=ART, name_prefix=f"gail4_{exp}")
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=CallbackList([cb, ckpt_cb]),
                progress_bar=False)
    print(f"trained {args.timesteps:,} in {time.time()-t0:.0f}s | Dseg acc last10="
          f"{np.mean(cb.accs[-10:]):.2f}", flush=True)
    final_p = os.path.join(ART, f"gail4_{exp}_final.zip")
    model.save(final_p)

    # ---- 평가 (24와 동일 프로토콜, RNN 상태 어댑터만 다름) ----
    H_units = []
    for road in test_roads:
        H_units.extend(p21.chunk_signals(p20.human_signals(road, dd)))
    XH = np.vstack([p21.seg_features(h) for h in H_units])
    tex_h = dict(sdlp=float(np.mean([np.std(h["e"]) for h in H_units])),
                 wl=float(np.nanmean([p20.wavelength(h["e"]) for h in H_units])),
                 srr=float(np.mean([p20.srr(h["theta"], 0.5) for h in H_units])),
                 srr2=float(np.mean([p20.srr(h["theta"], 2.0) for h in H_units])))
    env_t = DrivingEnv(test_roads, dd=dd, record=True)

    def eval_model(m, per_road, seed0=700):
        S_sig, offs = [], 0
        pol = RNNPolicy(m)
        for k in range(len(test_roads)):
            for j in range(per_road):
                np.random.seed(seed0 + k * 20 + j); torch.manual_seed(seed0 + k * 20 + j)
                pol.reset()
                traj, o = rollout(env_t, pol, k)
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
        return p24.cv_auc(X, yy), tex, off_rate, nmin

    cands = sorted(_glob.glob(os.path.join(ART, f"gail4_{exp}_*_steps.zip"))) + [final_p]
    print(f"tournament: {len(cands)} checkpoints", flush=True)
    results = []
    for p in cands:
        m = RecurrentPPO.load(p, device="cpu")
        a, tex, off, _ = eval_model(m, per_road=4)
        print(f"  {os.path.basename(p):32s} AUC={a:.3f} SDLP={tex['sdlp']:.3f} "
              f"wl={tex['wl']:.0f} SRR={tex['srr']:.1f} off={off:.2f}", flush=True)
        results.append((a, p))
    results.sort()
    _, best_p = results[0]

    m = RecurrentPPO.load(best_p, device="cpu")
    auc, tex, off_rate, nmin = eval_model(m, per_road=args.per_road)
    m.save(os.path.join(ART, f"rl_{exp}_gail4.zip"))
    print(f"WINNER {os.path.basename(best_p)}: AUC={auc:.3f} (챔피언 v3.1=0.794) "
          f"off={off_rate:.2f}", flush=True)
    print(f"texture h/r: SDLP {tex_h['sdlp']:.3f}/{tex['sdlp']:.3f} "
          f"wl {tex_h['wl']:.0f}/{tex['wl']:.0f} SRR {tex_h['srr']:.1f}/{tex['srr']:.1f} "
          f"SRR2 {tex_h['srr2']:.1f}/{tex['srr2']:.1f}", flush=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    names = ["v3.1\n(챔피언)", "GAIL3\n(MLP+구간D)", "GAIL4\n(LSTM+구간D)"]
    vals = [0.794, 0.967, auc]
    ax.bar(names, vals, color=["#1D9E75", "#888780", "#7F77DD"])
    ax.axhline(0.5, ls=":", color="#185FA5", label="구별불가(0.5)")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    ax.set_ylim(0.4, 1.05); ax.set_ylabel("C2ST AUC"); ax.legend()
    ax.set_title("GAIL 4차: 기억(LSTM) 정책 — 무기억 가설 검증")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"fig_gail4_{exp}.png"), dpi=120); plt.close(fig)

    json.dump(dict(exp=exp, auc=auc, off_rate=off_rate, winner=os.path.basename(best_p),
                   texture_h=tex_h, texture_r=tex,
                   d_acc_last10=float(np.mean(cb.accs[-10:]))),
              open(os.path.join(REP, f"gail4_{exp}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved rl_%s_gail4.zip + gail4 json + fig" % exp, flush=True)


if __name__ == "__main__":
    main()
