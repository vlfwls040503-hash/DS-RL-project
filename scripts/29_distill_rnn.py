# -*- coding: utf-8 -*-
"""
29_distill_rnn.py  --  D3: 챔피언 v3.2(하이브리드)를 단독 LSTM 정책으로 증류.

GAIL 4라운드의 결론: 적대 기울기는 사람의 시간구조로 인도하지 못한다. 대신 이미
사람다움을 검증받은 챔피언(광권한 RL 조향 + 사람 청크 목표 주입 + PD 속도)을
**교사**로 삼아, 그 닫힌루프 행동을 LSTM 학생에 지도학습으로 이식한다.
- 학생 입력 = 환경 관측(11) + 특성 3 (sigma_scale, b_bias, v_scale) → 군집 프로토콜 유지
- 사람 청크 주입이 만들던 배회는 학생의 은닉상태가 내재화해야 함 (관측 e 이력에 노출)
- 가우시안 헤드 + NLL (교훈: logstd_min=-3, grad clip 5.0, lr 5e-4)
- DAgger 1라운드: 학생이 몰고 교사가 라벨 (복합오차 보정)
- 평가 샘플링: AR(1) 상관 노이즈 (rho val 선택) — 백색잡음 지문 방지

  python 29_distill_rnn.py
"""
import os, json, argparse, importlib
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from common import ART, REP, CACHE, gen_split
from driving_env import DrivingEnv, load_roads, rollout, trim_roads, build_obs

p20 = importlib.import_module("20_profile_eval")
p21 = importlib.import_module("21_validation")
p22 = importlib.import_module("22_v3_spectral")
p24 = importlib.import_module("24_gail_seg")

for fam in ["Malgun Gothic", "NanumGothic", "Gulim"]:
    try:
        plt.rcParams["font.family"] = fam; break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(REP, "figs"); os.makedirs(FIG, exist_ok=True)
np.random.seed(0); torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
GAIN = 0.012                     # 챔피언 v3.2의 조향권한
SEQ, BATCH = 256, 64


class StudentLSTM(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True)
        self.mu = nn.Linear(hidden, 2)
        self.ls = nn.Linear(hidden, 2)

    def forward(self, x, hc=None):
        h, hc = self.lstm(x, hc)
        return self.mu(h), torch.clamp(self.ls(h), -3.0, 0.0), hc


class StudentPolicy:
    """rollout()용: 은닉상태 유지 + AR(1) 상관 샘플링. reset() 필수."""
    def __init__(self, net, traits, rho=0.9, seed=0):
        self.net, self.rho = net, rho
        self.tr = np.asarray(traits, np.float32)
        self.rng = np.random.RandomState(seed)
        self.hc, self.eps = None, np.zeros(2, np.float32)

    def reset(self):
        self.hc = None
        self.eps = np.zeros(2, np.float32)

    def __call__(self, obs, env):
        x = torch.from_numpy(np.concatenate([obs, self.tr])[None, None].astype("float32")).to(DEV)
        with torch.no_grad():
            mu, ls, self.hc = self.net(x, self.hc)
        mu = mu[0, 0].cpu().numpy(); sd = np.exp(ls[0, 0].cpu().numpy())
        z = self.rng.randn(2).astype("float32")
        self.eps = self.rho * self.eps + np.sqrt(1 - self.rho ** 2) * z
        return np.clip(mu + sd * self.eps, -1, 1).astype(np.float32)


def nll_train(net, seqs, epochs=12, lr=5e-4):
    """seqs: (X[T,in], Y[T,2]) 목록 → SEQ 청크 TBPTT 학습."""
    Xc, Yc = [], []
    for X, Y in seqs:
        for a in range(0, len(X) - SEQ + 1, SEQ):
            Xc.append(X[a:a + SEQ]); Yc.append(Y[a:a + SEQ])
    Xc = torch.from_numpy(np.stack(Xc)).float()
    Yc = torch.from_numpy(np.stack(Yc)).float()
    print(f"  train chunks: {len(Xc)} x {SEQ}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for ep in range(epochs):
        perm = torch.randperm(len(Xc))
        tot = 0.0
        for s in range(0, len(Xc), BATCH):
            b = perm[s:s + BATCH]
            xb, yb = Xc[b].to(DEV), Yc[b].to(DEV)
            mu, ls, _ = net(xb)
            loss = (0.5 * ((yb - mu) / ls.exp()) ** 2 + ls).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            tot += float(loss) * len(b)
        if ep % 3 == 0 or ep == epochs - 1:
            print(f"  ep{ep:02d} NLL={tot/len(Xc):.4f}", flush=True)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_road_collect", type=int, default=3)
    ap.add_argument("--per_road", type=int, default=10)
    args = ap.parse_args()

    roads, _, dd = load_roads(os.path.join(CACHE, "env_roads_2024.npz"))
    roads = trim_roads(roads)
    subj = np.array([r["subject"] for r in roads], "int64")
    tr, va, te = gen_split(subj, seed=0)
    fit_roads = [r for r, m in zip(roads, tr | va) if m]
    train_roads = [r for r, m in zip(roads, tr) if m]
    val_roads = [r for r, m in zip(roads, va) if m]
    test_roads = [r for r, m in zip(roads, te) if m]

    # ---- 교사 = 챔피언 v3.2 ----
    fit_chunks = []
    for r in fit_roads:
        for ch in p21.chunk_signals(p20.human_signals(r, dd)):
            fit_chunks.append(ch["e"])
    fr, A = p22.target_spectrum(fit_chunks)
    lib = []
    for e in fit_chunks:
        x = np.asarray(e, np.float64); x = x - x.mean()
        if len(x) >= 400 and x.std() > 1e-3:
            lib.append((x / x.std()).astype(np.float64))
    teacher_rl = PPO.load(os.path.join(ART, "rl_2024_wide.zip"), device="cpu")
    cal = json.load(open(os.path.join(REP, "v3_library_2024_wide.json"), encoding="utf-8"))
    sigma = float(cal["sigma"])
    pool = [dict(sdlp=float(np.std(r["e_ref"])), lpm=float(np.mean(r["e_ref"])),
                 v=float(np.mean(r["v_ref"]))) for r in fit_roads]
    sdlp_pool = float(np.mean([t["sdlp"] for t in pool]))
    v_pool = float(np.mean([t["v"] for t in pool]))

    def make_teacher(t, seed):
        s = float(np.clip(sigma * t["sdlp"] / sdlp_pool, 0.03, 1.2))
        pol = p22.SpectralPolicy(teacher_rl, fr, A, sigma=s, b_bias=t["lpm"],
                                 v_scale=t["v"] / v_pool, lib=lib, seed=seed)
        pol.reset()
        traits = np.array([s, t["lpm"], t["v"] / v_pool], np.float32)
        return pol, traits

    # ---- 수집 0회차: 교사 단독 주행 ----
    env = DrivingEnv(train_roads, dd=dd, record=True, steer_gain=GAIN)
    rng = np.random.RandomState(0)
    seqs = []
    print("collect round 0 (teacher drives)...", flush=True)
    for k in range(len(train_roads)):
        for j in range(args.per_road_collect):
            t = pool[rng.randint(len(pool))]
            pol, traits = make_teacher(t, seed=100 + k * 10 + j)
            obs, _ = env.reset(options={"road_idx": k})
            X, Y, done = [], [], False
            while not done:
                a = pol(obs, env)
                X.append(np.concatenate([obs, traits]))
                Y.append(a)
                obs, _, term, trunc, _ = env.step(a)
                done = term or trunc
            if len(X) > SEQ:
                seqs.append((np.asarray(X, np.float32), np.asarray(Y, np.float32)))
        if (k + 1) % 10 == 0:
            print(f"  road {k+1}/{len(train_roads)} ({len(seqs)} seqs)", flush=True)

    in_dim = seqs[0][0].shape[1]
    net = StudentLSTM(in_dim).to(DEV)
    print(f"train student (in_dim={in_dim}, {DEV})...", flush=True)
    nll_train(net, seqs)

    # ---- DAgger 1회차: 학생이 몰고 교사가 라벨 ----
    print("collect round 1 (DAgger: student drives, teacher labels)...", flush=True)
    dseqs = []
    for k in range(len(train_roads)):
        for j in range(2):
            t = pool[rng.randint(len(pool))]
            tpol, traits = make_teacher(t, seed=500 + k * 10 + j)
            spol = StudentPolicy(net, traits, rho=0.9, seed=700 + k * 10 + j)
            spol.reset()
            obs, _ = env.reset(options={"road_idx": k})
            X, Y, done = [], [], False
            while not done:
                a_t = tpol(obs, env)                     # 교사 라벨 (같은 상태에서)
                a_s = spol(obs, env)                     # 학생 행동으로 전진
                X.append(np.concatenate([obs, traits]))
                Y.append(a_t)
                obs, _, term, trunc, _ = env.step(a_s)
                done = term or trunc
            if len(X) > SEQ:
                dseqs.append((np.asarray(X, np.float32), np.asarray(Y, np.float32)))
        if (k + 1) % 15 == 0:
            print(f"  road {k+1}/{len(train_roads)}", flush=True)
    nll_train(net, seqs + dseqs, epochs=8)
    torch.save(net.state_dict(), os.path.join(ART, "student_lstm_2024.pt"))

    # ---- rho 선택 (val: SRR 근접) ----
    hv = [p20.human_signals(r, dd) for r in val_roads]
    srr_h = float(np.mean([p20.srr(h["theta"], 0.5) for h in hv]))
    env_v = DrivingEnv(val_roads, dd=dd, record=True, steer_gain=GAIN)
    best_rho, best_gap = 0.9, 1e9
    for rho in [0.0, 0.9, 0.98]:
        srrs, offs = [], 0
        for k in range(len(val_roads)):
            t = pool[np.random.RandomState(k).randint(len(pool))]
            _, traits = make_teacher(t, seed=0)
            spol = StudentPolicy(net, traits, rho=rho, seed=40 + k)
            spol.reset()
            traj, o = rollout(env_v, spol, k)
            offs += int(o)
            if len(traj) > 60:
                srrs.append(p20.srr(p20.rl_signals(traj, gain=GAIN)["theta"], 0.5))
        gap = abs(np.mean(srrs) - srr_h) + 100.0 * (offs > len(val_roads) * 0.2)
        print(f"  rho={rho}: SRR={np.mean(srrs):.1f} (사람 {srr_h:.1f}) off={offs}", flush=True)
        if gap < best_gap:
            best_gap, best_rho = gap, rho
    print(f"rho*={best_rho}", flush=True)

    # ---- 최종 평가 (군집 프로토콜, 챔피언과 동일) ----
    H_units = []
    for road in test_roads:
        H_units.extend(p21.chunk_signals(p20.human_signals(road, dd)))
    XH = np.vstack([p21.seg_features(h) for h in H_units])
    env_t = DrivingEnv(test_roads, dd=dd, record=True, steer_gain=GAIN)
    rng2 = np.random.RandomState(3)
    S_sig, offs, n = [], 0, 0
    for k in range(len(test_roads)):
        for j in range(args.per_road):
            t = pool[rng2.randint(len(pool))]
            _, traits = make_teacher(t, seed=0)
            spol = StudentPolicy(net, traits, rho=best_rho, seed=3000 + k * 20 + j)
            spol.reset()
            traj, o = rollout(env_t, spol, k)
            offs += int(o); n += 1
            if len(traj) > 60:
                S_sig.append(p20.rl_signals(traj, gain=GAIN))
    off_rate = offs / max(n, 1)
    tex_h = dict(sdlp=float(np.mean([np.std(h["e"]) for h in H_units])),
                 wl=float(np.nanmean([p20.wavelength(h["e"]) for h in H_units])),
                 srr=float(np.mean([p20.srr(h["theta"], 0.5) for h in H_units])),
                 srr2=float(np.mean([p20.srr(h["theta"], 2.0) for h in H_units])))
    tex = dict(sdlp=float(np.mean([np.std(s["e"]) for s in S_sig])),
               wl=float(np.nanmean([p20.wavelength(s["e"]) for s in S_sig])),
               srr=float(np.mean([p20.srr(s["theta"], 0.5) for s in S_sig])),
               srr2=float(np.mean([p20.srr(s["theta"], 2.0) for s in S_sig])))
    XS = np.vstack([p21.seg_features(s) for s in S_sig])
    rng3 = np.random.RandomState(2)
    nmin = min(len(XH), len(XS))
    X = np.vstack([XH[rng3.choice(len(XH), nmin, replace=False)],
                   XS[rng3.choice(len(XS), nmin, replace=False)]])
    yy = np.concatenate([np.zeros(nmin), np.ones(nmin)])
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    auc = p24.cv_auc(X, yy)
    print(f"[student standalone] AUC={auc:.3f} (교사 v3.2=0.671, v3.1=0.794) off={off_rate:.2f}",
          flush=True)
    print(f"texture h/s: SDLP {tex_h['sdlp']:.3f}/{tex['sdlp']:.3f} "
          f"wl {tex_h['wl']:.0f}/{tex['wl']:.0f} SRR {tex_h['srr']:.1f}/{tex['srr']:.1f} "
          f"SRR2 {tex_h['srr2']:.1f}/{tex['srr2']:.1f}", flush=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    names = ["v3.1", "v3.2 교사\n(하이브리드)", "학생 LSTM\n(단독)", "GAIL 최선\n(참고)"]
    vals = [0.794, 0.671, auc, 0.889]
    ax.bar(names, vals, color=["#888780", "#1D9E75", "#7F77DD", "#888780"])
    ax.axhline(0.5, ls=":", color="#185FA5", label="구별불가(0.5)")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    ax.set_ylim(0.4, 1.05); ax.set_ylabel("C2ST AUC"); ax.legend()
    ax.set_title("D3 증류: 하이브리드 챔피언 → 단독 LSTM")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_distill_2024.png"), dpi=120)
    plt.close(fig)
    json.dump(dict(exp="2024", auc=auc, off_rate=off_rate, rho=best_rho, gain=GAIN,
                   teacher_auc=0.671, texture_h=tex_h, texture_s=tex),
              open(os.path.join(REP, "distill_2024.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved student_lstm_2024.pt + distill_2024.json + fig", flush=True)


if __name__ == "__main__":
    main()
