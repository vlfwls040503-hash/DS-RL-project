# -*- coding: utf-8 -*-
"""
50_diffusion_policy.py  --  디퓨전 정책 (액션 청크 DDPM) — 캠페인 최종 계열.

근거: 3층 시험이 드러낸 잔여 지문(원시 0.978, GBM의 횡속피크·횡가분산)은 반응형
정책의 폐루프 응답 잔재 — 액션 '시퀀스'의 자기상관(사람의 펄스 구조)을 원생적으로
학습·생성하는 계열만 남음. 액션 청크(H=32, 1.6s)를 DDPM으로 생성해 환경에서 실행
(리시딩 호라이즌) — 인루프 요건 충족.

  python 50_diffusion_policy.py train      # 데이터셋 + 학습
  python 50_diffusion_policy.py eval       # 게이트 + 3층 시험(남산)
"""
import os, sys, json, importlib, time
import numpy as np
import torch
import torch.nn as nn

from common import ART, REP, CACHE, gen_split
from driving_env import (DrivingEnv, load_roads, rollout, trim_roads,
                         make_expert_dataset, build_obs, OBS_DIM, _smooth)

p20 = importlib.import_module("20_profile_eval")
p21 = importlib.import_module("21_validation")
p39 = importlib.import_module("39_discriminator_audit")
p43 = importlib.import_module("43_gbm_anatomy")

np.random.seed(0); torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
GAIN = 0.012
H, EXEC = 32, 8             # 청크 길이(1.6s), 실행 0.4s(사람 펄스 1개)
T_DIFF = 50                 # 학습 확산 스텝
DDIM_STEPS = 10             # 샘플링 스텝
CKPT = os.path.join(ART, "diffusion50.pt")


def make_sequences(roads, dd):
    """도로별 전문가 (obs, action) '순차' 시퀀스 → (cond, chunk) 창."""
    C, A = [], []
    for r in roads:
        e = _smooth(np.asarray(r["e_ref"], np.float64))
        v = _smooth(np.asarray(r["v_ref"], np.float64))
        psi = np.gradient(e, dd)
        kap = np.asarray(r["curv"], np.float64) + np.gradient(psi, dd)
        steer = np.clip(kap / GAIN, -1, 1)
        acc = np.clip(v * np.gradient(v, dd) / 3.0, -1, 1)
        # 시간축 정합(1차 게이트 전멸의 원인 수정): 실행은 0.05s 시간스텝이므로
        # 거리그리드 액션을 v 기반 시간 걸음(Δidx = v·dt/dd)으로 리샘플해 청크화.
        n = len(e)
        dtidx = np.maximum(v * 0.05 / dd, 0.2)
        for a0 in range(0, n - 1, 6):
            step = dtidx[min(a0, n - 1)]
            idx = a0 + np.arange(H) * step
            hidx = a0 - np.arange(HIST, 0, -1) * step
            if idx[-1] >= n - 1 or hidx[0] < 0:
                continue
            ii = idx.astype(int); hh = hidx.astype(int)
            obs = build_obs(r, a0, v[a0], e[a0], psi[a0])
            hist = steer[hh][:, None].reshape(-1)
            g_ff = np.clip(np.asarray(r["curv"], np.float64)[ii] / GAIN, -1, 1)
            C.append(np.concatenate([obs, hist, g_ff]).astype(np.float32))
            A.append(steer[ii][:, None])
    return np.asarray(C, np.float32), np.asarray(A, np.float32)


HIST = 8                     # 히스토리 조건: 직전 실행 액션 8스텝 (표준 레시피)
COND_DIM = OBS_DIM + HIST * 1 + H


class Denoiser(nn.Module):
    def __init__(self, obs_dim=COND_DIM, h=H):
        super().__init__()
        self.temb = nn.Embedding(T_DIFF, 32)
        self.net = nn.Sequential(
            nn.Linear(h * 1 + obs_dim + 32, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, h * 1))

    def forward(self, x, t, cond):        # x:(B,H,2)
        B = x.shape[0]
        inp = torch.cat([x.reshape(B, -1), cond, self.temb(t)], dim=1)
        return self.net(inp).reshape(B, H, 1)


def cosine_schedule(T):
    s = 0.008
    t = np.arange(T + 1) / T
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    ab = f / f[0]
    return torch.from_numpy(ab[1:].astype("float32"))     # alpha_bar[t]


def train():
    roads, _, dd = load_roads(os.path.join(CACHE, "env_roads_multi8.npz"))
    roads = trim_roads(roads)
    subj = np.array([r["subject"] for r in roads], "int64")
    tr, va, te = gen_split(subj, seed=0)
    train_roads = [r for r, m in zip(roads, tr) if m]
    C, A = make_sequences(train_roads, dd)
    print(f"sequences: {len(C):,} (H={H})", flush=True)
    mu_c, sd_c = C.mean(0), C.std(0) + 1e-6
    Cn = (C - mu_c) / sd_c
    net = Denoiser().to(DEV)
    ab = cosine_schedule(T_DIFF).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-5)
    Ct = torch.from_numpy(Cn); At = torch.from_numpy(A)
    t0 = time.time()
    for ep in range(25):
        perm = torch.randperm(len(Ct))
        tot = 0.0
        for s in range(0, len(Ct), 512):
            b = perm[s:s + 512]
            x0 = At[b].to(DEV); cond = Ct[b].to(DEV)
            t = torch.randint(0, T_DIFF, (len(b),), device=DEV)
            eps = torch.randn_like(x0)
            a = ab[t].view(-1, 1, 1)
            xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
            opt.zero_grad()
            loss = ((net(xt, t, cond) - eps) ** 2).mean()
            loss.backward(); opt.step()
            tot += float(loss) * len(b)
        if ep % 5 == 0 or ep == 24:
            print(f"  ep{ep:02d} loss={tot/len(Ct):.4f}", flush=True)
    torch.save(dict(state=net.state_dict(), mu_c=mu_c, sd_c=sd_c), CKPT)
    print(f"trained in {time.time()-t0:.0f}s -> diffusion50.pt", flush=True)


class DiffusionPolicy:
    """리시딩 호라이즌: EXEC 스텝마다 DDIM으로 청크 재샘플. reset() 필수."""
    def __init__(self, seed=0):
        d = torch.load(CKPT, map_location=DEV, weights_only=False)
        self.net = Denoiser().to(DEV); self.net.load_state_dict(d["state"]); self.net.eval()
        self.mu, self.sd = d["mu_c"], d["sd_c"]
        self.ab = cosine_schedule(T_DIFF).to(DEV)
        self.g = torch.Generator(device=DEV).manual_seed(seed)
        self.buf, self.ptr = None, 0

    def reset(self):
        self.buf, self.ptr = None, 0
        self.hist = np.zeros((HIST, 1), np.float32)

    @torch.no_grad()
    def _sample(self, cond, guide=None, lam=0.35):
        x = torch.randn(1, H, 1, generator=self.g, device=DEV)
        ts = np.linspace(T_DIFF - 1, 0, DDIM_STEPS).astype(int)
        gt = None if guide is None else torch.from_numpy(
            guide.astype("float32")).view(1, H, 1).to(DEV)
        for i, t in enumerate(ts):
            tt = torch.full((1,), int(t), device=DEV, dtype=torch.long)
            eps = self.net(x, tt, cond)
            a_t = self.ab[t]
            x0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
            x0 = x0.clamp(-1, 1)
            if gt is not None:                 # 안정성 가이던스: 도로추종 참조로 혼합
                x0 = (1 - lam) * x0 + lam * gt
            if i < len(ts) - 1:
                a_p = self.ab[ts[i + 1]]
                x = a_p.sqrt() * x0 + (1 - a_p).sqrt() * eps
            else:
                x = x0
        return x[0].cpu().numpy()

    def _guide(self, env):
        """참조 청크: 현재 PD 조향(초항) + 전방 곡률 피드포워드."""
        from driving_env import pd_action
        r = env.road
        M = len(r["curv"])
        st0 = float(pd_action(env)[0])
        g = np.empty(H, np.float32)
        g[0] = st0
        for k in range(1, H):
            j = min(int((env.s + env.v * 0.05 * k) / env.dd), M - 1)
            g[k] = np.clip(r["curv"][j] / GAIN, -1, 1)
        return g

    def __call__(self, obs, env):
        if self.buf is None or self.ptr >= EXEC:
            c = np.concatenate([obs, self.hist.reshape(-1)]).astype("float32")
            g = self._guide(env)
            c2 = np.concatenate([c, g]).astype("float32")
            cond = torch.from_numpy(((c2 - self.mu) / self.sd)[None]).to(DEV)
            new = self._sample(cond, guide=g, lam=0.1)
            if self.buf is not None:              # 경계 크로스페이드(SRR 아티팩트 완화)
                for kk in range(4):
                    w = (kk + 1) / 5.0
                    new[kk] = (1 - w) * self.buf[min(self.ptr + kk, H - 1)] + w * new[kk]
            self.buf = new
            self.ptr = 0
        st = float(np.clip(self.buf[self.ptr][0], -1, 1))
        self.ptr += 1
        self.hist = np.vstack([self.hist[1:], [[st]]])
        i = min(int(env.s / env.dd), len(env.road["v_ref"]) - 1)
        acc = float(np.clip((env.road["v_ref"][i] - env.v) / 3.0, -1, 1))
        return np.array([st, acc], np.float32)


def evaluate():
    # 인도메인(2024 val) 게이트 먼저 — 개념 성립 확인 후 남산
    r24, _, dd24 = load_roads(os.path.join(CACHE, "env_roads_2024.npz"))
    r24 = trim_roads(r24)
    s24 = np.array([r["subject"] for r in r24], "int64")
    _, va24, _ = gen_split(s24, seed=0)
    val24 = [r for r, m in zip(r24, va24) if m]
    env24 = DrivingEnv([dict(r, e_ref=np.zeros_like(r["v_ref"])) for r in val24],
                       dd=dd24, record=True, steer_gain=GAIN)
    o24 = 0
    for k in range(len(val24)):
        pol = DiffusionPolicy(seed=k)
        pol.reset()
        _, off = rollout(env24, pol, k)
        o24 += int(off)
    print(f"[게이트-인도메인 2024] 이탈 {o24}/{len(val24)}", flush=True)
    if o24 / len(val24) > 0.3:
        print("GATE FAIL (in-domain)", flush=True)
        json.dump(dict(gate="FAIL_indomain", off=o24 / len(val24)),
                  open(os.path.join(REP, "diffusion.json"), "w", encoding="utf-8"))
        return
    # ---- 홈(2024 test) 공정시험 — 질감 가치의 1차 판정 (현 기록 0.770) ----
    _, _, te24 = gen_split(s24, seed=0)
    test24 = [r for r, m in zip(r24, te24) if m]
    envH = DrivingEnv([dict(r, e_ref=np.zeros_like(r["v_ref"])) for r in test24],
                      dd=dd24, record=True, steer_gain=GAIN)
    TH, offH = [], 0
    for k in range(len(test24)):
        for j in range(4):
            pol = DiffusionPolicy(seed=7000 + k * 20 + j)
            pol.reset()
            traj, off = rollout(envH, pol, k)
            offH += int(off)
            if len(traj) > 60:
                TH.append((traj, test24[k]))
    SH = [p21.seg_features(p39.symmetric_signals(t, r, dd24, GAIN)) for t, r in TH]
    HH = []
    for r in test24:
        for c in p21.chunk_signals(p20.human_signals(r, dd24)):
            HH.append(p21.seg_features(c))
    rngb = np.random.RandomState(1)
    nu = min(len(HH), len(SH))
    aucH, impH = p43.fair_exam([HH[i] for i in rngb.choice(len(HH), nu, replace=False)],
                               [SH[i] for i in rngb.choice(len(SH), nu, replace=False)],
                               ret_imp=True)
    topH = np.argsort(-impH)[:3]
    print(f"[2024-home] 디퓨전 GBM AUC={aucH:.3f} off={offH/max(len(test24)*4,1):.2f} | 상위: "
          + ", ".join(f"{p43.FEATS[i]}={impH[i]:.3f}" for i in topH), flush=True)
    json.dump(dict(home_auc=float(aucH), home_off=offH / max(len(test24) * 4, 1)),
              open(os.path.join(REP, "diffusion.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    roads, _, dd = load_roads(os.path.join(CACHE, "env_roads_namsan.npz"))
    roads = trim_roads(roads)
    sN = np.array([r["subject"] for r in roads], "int64")
    _, vaN, _ = gen_split(sN, seed=0)
    valN = [r for r, m in zip(roads, vaN) if m]
    env_v = DrivingEnv([dict(r, e_ref=np.zeros_like(r["v_ref"])) for r in valN],
                       dd=dd, record=True, steer_gain=GAIN)
    offs, sds = 0, []
    for k in range(len(valN)):
        pol = DiffusionPolicy(seed=k)
        pol.reset()
        traj, off = rollout(env_v, pol, k)
        offs += int(off)
        if len(traj) > 60:
            sds.append(float(np.std(p20.rl_signals(traj, gain=GAIN)["e"])))
    print(f"[게이트] 남산 val: 이탈 {offs}/{len(valN)} SDLP {np.mean(sds):.3f}", flush=True)
    if offs / len(valN) > 0.3:
        print("GATE FAIL", flush=True)
        json.dump(dict(gate="FAIL", off=offs / len(valN)),
                  open(os.path.join(REP, "diffusion.json"), "w", encoding="utf-8"))
        return
    env = DrivingEnv([dict(r, e_ref=np.zeros_like(r["v_ref"])) for r in roads],
                     dd=dd, record=True, steer_gain=GAIN)
    T, offc = [], 0
    for k in range(len(roads)):
        for j in range(2):
            pol = DiffusionPolicy(seed=7000 + k * 20 + j)
            pol.reset()
            traj, off = rollout(env, pol, k)
            offc += int(off)
            if len(traj) > 60:
                T.append((traj, roads[k]))
    S = [p21.seg_features(p39.symmetric_signals(t, r, dd, GAIN)) for t, r in T]
    Hu = [p21.seg_features(p20.human_signals(r, dd)) for r in roads]
    rngb = np.random.RandomState(1)
    nu = min(len(Hu), len(S))
    Hb = [Hu[i] for i in rngb.choice(len(Hu), nu, replace=False)]
    Sb = [S[i] for i in rngb.choice(len(S), nu, replace=False)]
    auc, imp = p43.fair_exam(Hb, Sb, ret_imp=True)
    top = np.argsort(-imp)[:3]
    print(f"[namsan] 디퓨전 GBM AUC={auc:.3f} off={offc/max(len(roads)*2,1):.2f} | 상위: "
          + ", ".join(f"{p43.FEATS[i]}={imp[i]:.3f}" for i in top), flush=True)
    json.dump(dict(auc=float(auc), off=offc / max(len(roads) * 2, 1)),
              open(os.path.join(REP, "diffusion.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved diffusion.json", flush=True)


if __name__ == "__main__":
    (train if "train" in sys.argv else evaluate)()
