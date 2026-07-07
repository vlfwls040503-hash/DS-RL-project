# -*- coding: utf-8 -*-
"""
51_trackA_diffusion.py  --  트랙 A 본격 스택: 잔차 디퓨전 (가이드+τ·잔차).

경량 6판의 실패 지도가 정한 설계:
  - **잔차 정식화**: 디퓨전은 사람 질감 잔차만 생성(도로추종은 가이드 담당)
    → 안정성 구조 보장 + 남산 OOD 해소 + 챔피언 분업구조의 액션판
  - 대형화(512×4)·EMA(0.999)·전체 train+val 데이터·80에폭
  - 게이트에 진행량 기준(정지표류 차단 교훈), τ는 val SDLP 1노브 보정
  - EXEC=8(0.4s 개루프 상한 교훈), 경계 크로스페이드 유지

  python 51_trackA_diffusion.py train
  python 51_trackA_diffusion.py eval
"""
import os, sys, json, importlib, time, copy
import numpy as np
import torch
import torch.nn as nn

from common import ART, REP, CACHE, gen_split
from driving_env import (DrivingEnv, load_roads, rollout, trim_roads,
                         build_obs, OBS_DIM, _smooth, pd_action)

p20 = importlib.import_module("20_profile_eval")
p21 = importlib.import_module("21_validation")
p39 = importlib.import_module("39_discriminator_audit")
p43 = importlib.import_module("43_gbm_anatomy")

np.random.seed(0); torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
GAIN = 0.012
H, EXEC, HIST = 256, 8, 32
T_DIFF, DDIM_STEPS = 50, 12
CKPT = os.path.join(ART, "trackA51.pt")


def guide_from_road(r, ii):
    return np.clip(np.asarray(r["curv"], np.float64)[ii] / GAIN, -1, 1)


def make_sequences(roads, dd):
    C, A = [], []
    for r in roads:
        # w=3 저평활: 사람 질감을 잔차 라벨에 보존. BC에선 폐루프 불안정으로 기각됐지만
        # 3대역에선 잔차가 개루프 가산이라 안정성은 가이드가 보장 — 구조적 이점.
        e = _smooth(np.asarray(r["e_ref"], np.float64), w=3)
        v = _smooth(np.asarray(r["v_ref"], np.float64))
        psi = np.gradient(e, dd)
        kap = np.asarray(r["curv"], np.float64) + np.gradient(psi, dd)
        steer = np.clip(kap / GAIN, -1, 1)
        n = len(e)
        dtidx = np.maximum(v * 0.05 / dd, 0.2)
        for a0 in range(0, n - 1, 24):
            step = dtidx[min(a0, n - 1)]
            idx = a0 + np.arange(H) * step
            hidx = a0 - np.arange(HIST, 0, -1) * step
            if idx[-1] >= n - 1 or hidx[0] < 0:
                continue
            ii = idx.astype(int); hh = hidx.astype(int)
            g = guide_from_road(r, ii)
            resid = steer[ii] - g                        # ★ 잔차 = 사람 질감만
            hist_resid = steer[hh] - guide_from_road(r, hh)
            obs = build_obs(r, a0, v[a0], e[a0], psi[a0])
            C.append(np.concatenate([obs, hist_resid, g]).astype(np.float32))
            A.append(resid[:, None].astype(np.float32))
    return np.asarray(C, np.float32), np.asarray(A, np.float32)


COND_DIM = OBS_DIM + HIST + H


class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.temb = nn.Embedding(T_DIFF, 64)
        self.net = nn.Sequential(
            nn.Linear(H + COND_DIM + 64, 512), nn.SiLU(),
            nn.Linear(512, 512), nn.SiLU(),
            nn.Linear(512, 512), nn.SiLU(),
            nn.Linear(512, 512), nn.SiLU(),
            nn.Linear(512, H))

    def forward(self, x, t, cond):
        B = x.shape[0]
        return self.net(torch.cat([x.reshape(B, -1), cond, self.temb(t)], 1)).reshape(B, H, 1)


def cosine_schedule(T):
    s = 0.008
    t = np.arange(T + 1) / T
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    return torch.from_numpy((f / f[0])[1:].astype("float32"))


def train():
    roads, _, dd = load_roads(os.path.join(CACHE, "env_roads_multi8.npz"))
    roads = trim_roads(roads)
    subj = np.array([r["subject"] for r in roads], "int64")
    tr, va, te = gen_split(subj, seed=0)
    train_roads = [r for r, m in zip(roads, tr | va) if m]   # 전체 train+val
    C, A = make_sequences(train_roads, dd)
    print(f"sequences: {len(C):,} | resid std={A.std():.3f}", flush=True)
    mu_c, sd_c = C.mean(0), C.std(0) + 1e-6
    Cn = (C - mu_c) / sd_c
    r_sd = float(A.std()) + 1e-6
    An = A / r_sd                                            # 잔차 정규화
    net = Denoiser().to(DEV)
    ema = copy.deepcopy(net)
    for p in ema.parameters():
        p.requires_grad_(False)
    ab = cosine_schedule(T_DIFF).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)
    Ct = torch.from_numpy(Cn); At = torch.from_numpy(An)
    t0 = time.time()
    for ep in range(80):
        perm = torch.randperm(len(Ct))
        tot = 0.0
        for s in range(0, len(Ct), 1024):
            b = perm[s:s + 1024]
            x0 = At[b].to(DEV); cond = Ct[b].to(DEV)
            t = torch.randint(0, T_DIFF, (len(b),), device=DEV)
            eps = torch.randn_like(x0)
            a = ab[t].view(-1, 1, 1)
            xt = a.sqrt() * x0 + (1 - a).sqrt() * eps
            opt.zero_grad()
            loss = ((net(xt, t, cond) - eps) ** 2).mean()
            loss.backward(); opt.step()
            with torch.no_grad():                            # EMA
                for pe, pn in zip(ema.parameters(), net.parameters()):
                    pe.mul_(0.999).add_(pn, alpha=0.001)
            tot += float(loss) * len(b)
        sch.step()
        if ep % 10 == 0 or ep == 79:
            print(f"  ep{ep:02d} loss={tot/len(Ct):.4f}", flush=True)
    torch.save(dict(state=ema.state_dict(), mu_c=mu_c, sd_c=sd_c, r_sd=r_sd), CKPT)
    print(f"trained {time.time()-t0:.0f}s -> trackA51.pt (EMA)", flush=True)


class ResidualDiffusionPolicy:
    """3대역: 조향 = 가이드(도로추종, b-목표 PD) + τ·디퓨전 잔차(질감).
    배회(저주파 의도)는 청크 주입(sigma·w)을 b로 — 챔피언 부품 결합."""
    def __init__(self, tau=1.0, sigma=0.0, lib=None, seed=0):
        d = torch.load(CKPT, map_location=DEV, weights_only=False)
        self.net = Denoiser().to(DEV); self.net.load_state_dict(d["state"]); self.net.eval()
        self.mu, self.sd, self.r_sd = d["mu_c"], d["sd_c"], float(d["r_sd"])
        self.ab = cosine_schedule(T_DIFF).to(DEV)
        self.tau, self.sigma, self.lib = float(tau), float(sigma), lib
        self.g = torch.Generator(device=DEV).manual_seed(seed)
        self.rng = np.random.RandomState(seed)

    def reset(self):
        self.buf, self.gbuf, self.ptr, self.rptr = None, None, 0, 0
        self.hist = np.zeros(HIST, np.float32)
        self.w = None
        if self.lib is not None and self.sigma > 0:      # 배회 목표 (10m 그리드)
            need = 1200
            w = None
            while w is None or len(w) < need:
                c = self.lib[self.rng.randint(len(self.lib))].astype(np.float64)
                if self.rng.rand() < 0.5:
                    c = -c[::-1]
                w = c.copy() if w is None else np.concatenate(
                    [w[:-20], (w[-20:] * np.linspace(1, 0, 20) + c[:20]
                               * np.linspace(0, 1, 20)), c[20:]])
            self.w = self.sigma * w

    @torch.no_grad()
    def _sample(self, cond):
        x = torch.randn(1, H, 1, generator=self.g, device=DEV)
        ts = np.linspace(T_DIFF - 1, 0, DDIM_STEPS).astype(int)
        for i, t in enumerate(ts):
            tt = torch.full((1,), int(t), device=DEV, dtype=torch.long)
            eps = self.net(x, tt, cond)
            a_t = self.ab[t]
            x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-3, 3)
            if i < len(ts) - 1:
                a_p = self.ab[ts[i + 1]]
                x = a_p.sqrt() * x0 + (1 - a_p).sqrt() * eps
            else:
                x = x0
        return x[0, :, 0].cpu().numpy() * self.r_sd

    def _btarget(self, s):
        if self.w is None:
            return 0.0
        gi = min(int(s / 10.0), len(self.w) - 2)
        f = s / 10.0 - int(s / 10.0)
        return float(np.clip((1 - f) * self.w[gi] + f * self.w[gi + 1], -1.0, 1.0))

    def _guide(self, env):
        r = env.road
        M = len(r["curv"])
        g = np.empty(H, np.float32)
        # b-목표 PD 초항: 배회 목표를 향한 수렴 (pd_action 게인과 동일)
        b = self._btarget(env.s)
        i = min(int(env.s / env.dd), M - 1)
        psi_des = float(np.clip(0.10 * (b - env.e), -0.12, 0.12))
        kcmd = float(r["curv"][i]) + 0.6 * (psi_des - env.psi)
        st_pd = float(np.clip(kcmd / GAIN, -1.0, 1.0))
        for k in range(H):
            j = min(int((env.s + env.v * 0.05 * k) / env.dd), M - 1)
            ff = float(np.clip(r["curv"][j] / GAIN, -1, 1))
            wgt = max(0.0, 1.0 - k / 8.0)                   # PD 초항 8스텝 감쇠
            g[k] = wgt * st_pd + (1 - wgt) * ff
        return g

    def __call__(self, obs, env):
        # 잔차 스트림(질감, 장지평 연속) — 소진 임박 시 크로스페이드 재샘플
        if self.buf is None or self.rptr >= H - 8:
            gd_full = self._guide(env)
            c = np.concatenate([obs, self.hist, gd_full]).astype("float32")
            cond = torch.from_numpy(((c - self.mu) / self.sd)[None]).to(DEV)
            new = self.tau * self._sample(cond)
            if self.buf is not None:
                for kk in range(8):
                    w = (kk + 1) / 9.0
                    new[kk] = (1 - w) * self.buf[min(self.rptr + kk, H - 1)] + w * new[kk]
            self.buf, self.rptr = new, 0
        # 가이드(되먹임)는 EXEC=8 스텝마다 갱신
        if self.gbuf is None or self.ptr >= EXEC:
            self.gbuf, self.ptr = self._guide(env)[:EXEC + 8], 0
        st = float(np.clip(self.gbuf[self.ptr] + self.buf[self.rptr], -1, 1))
        self.hist = np.concatenate([self.hist[1:], [self.buf[self.rptr]]]).astype(np.float32)
        self.ptr += 1
        self.rptr += 1
        i = min(int(env.s / env.dd), len(env.road["v_ref"]) - 1)
        acc = float(np.clip((env.road["v_ref"][i] - env.v) / 3.0, -1, 1))
        return np.array([st, acc], np.float32)


def gate(env, roads_sub, tau, n=None, seed0=0):
    """이탈율 + 진행량(도로 80% 주파) 이중 게이트."""
    n = n or len(roads_sub)
    offs, prog = 0, []
    for k in range(n):
        pol = ResidualDiffusionPolicy(tau=tau, seed=seed0 + k)
        pol.reset()
        traj, off = rollout(env, pol, k % len(roads_sub))
        offs += int(off)
        r_ = roads_sub[k % len(roads_sub)]
        L = len(r_["curv"]) * env.dd
        reach = min(L, float(np.mean(r_["v_ref"])) * 4000 * 0.05)   # 스텝상한 내 도달가능거리
        prog.append(traj[-1, 0] / max(reach, 1) if len(traj) else 0.0)
    return offs / n, float(np.mean(prog))


def evaluate():
    r24, _, dd24 = load_roads(os.path.join(CACHE, "env_roads_2024.npz"))
    r24 = trim_roads(r24)
    s24 = np.array([r["subject"] for r in r24], "int64")
    _, va24, te24 = gen_split(s24, seed=0)
    val24 = [dict(r, e_ref=np.zeros_like(r["v_ref"])) for r, m in zip(r24, va24) if m]
    env24 = DrivingEnv(val24, dd=dd24, record=True, steer_gain=GAIN)
    off, prog = gate(env24, val24, tau=1.0)
    print(f"[게이트 인도메인] off={off:.2f} 진행={prog:.2f}", flush=True)
    if off > 0.3 or prog < 0.8:
        print("GATE FAIL (in-domain)", flush=True)
        return

    # ---- 청크 라이브러리 (2024 fit — 챔피언 부품) ----
    fitlib = []
    fit24 = [r for r, m in zip(r24, ~te24) if m]
    for r in fit24:
        for ch_ in p21.chunk_signals(p20.human_signals(r, dd24)):
            x = np.asarray(ch_["e"], np.float64); x -= x.mean()
            if len(x) >= 400 and x.std() > 1e-3:
                fitlib.append((x / x.std()).astype(np.float64))
    print(f"chunk lib {len(fitlib)}", flush=True)

    def calib(env_v, val_roads_, dd_):
        """(τ, σ) 2노브 결합 보정: 목표 = 사람 (mean|Δe|, SDLP)."""
        hv_ = [p20.human_signals(r, dd_) for r in val_roads_]
        tgt_sd = float(np.mean([np.std(h["e"]) for h in hv_]))
        tgt_de = float(np.mean([np.mean(np.abs(np.diff(h["e"]))) for h in hv_]))
        best = None
        for tau in [0.3, 0.6, 1.0]:
            s0 = 0.2
            for it in range(2):                      # σ-solve 2회 반복
                sds, des = [], []
                for k in range(len(val_roads_)):
                    pol = ResidualDiffusionPolicy(tau=tau, sigma=s0, lib=fitlib,
                                                  seed=90 + k)
                    pol.reset()
                    traj, _ = rollout(env_v, pol, k)
                    if len(traj) > 60:
                        sg = p39.symmetric_signals(traj, val_roads_[k], dd_, GAIN)
                        sds.append(float(np.std(sg["e"])))
                        des.append(float(np.mean(np.abs(np.diff(sg["e"])))))
                s0 = float(np.clip(s0 * tgt_sd / max(np.mean(sds), 1e-6), 0.03, 1.0))
            gap = abs(np.mean(des) - tgt_de) / tgt_de \
                + 0.5 * abs(np.mean(sds) - tgt_sd) / tgt_sd
            print(f"    tau={tau} σ={s0:.3f}: |de|={np.mean(des):.4f}({tgt_de:.4f}) "
                  f"SDLP={np.mean(sds):.3f}({tgt_sd:.3f}) gap={gap:.3f}", flush=True)
            if best is None or gap < best[2]:
                best = (tau, s0, gap)
        return best[0], best[1]

    # ---- 남산 val: (τ, σ) 보정 + 게이트 ----
    rN, _, ddN = load_roads(os.path.join(CACHE, "env_roads_namsan.npz"))
    rN = trim_roads(rN)
    sN = np.array([r["subject"] for r in rN], "int64")
    _, vaN, _ = gen_split(sN, seed=0)
    valN_r = [r for r, m in zip(rN, vaN) if m]
    valN = [dict(r, e_ref=np.zeros_like(r["v_ref"])) for r in valN_r]
    envN = DrivingEnv(valN, dd=ddN, record=True, steer_gain=GAIN)
    print("[남산 보정]", flush=True)
    tauN, sigN = calib(envN, valN_r, ddN)
    offv, progv = gate(envN, valN, tau=tauN)
    print(f"[남산 게이트] tau={tauN} σ={sigN:.3f} off={offv:.2f} 진행={progv:.2f}", flush=True)
    print("[2024 보정]", flush=True)
    tau24, sig24 = calib(env24, [r for r, m in zip(r24, va24) if m], dd24)

    # ---- 시험: 홈 + 남산 GBM ----
    out = {"tau_namsan": tauN, "tau_2024": tau24}
    for exp, roads_all, ddx, per_road in [("2024", [r for r, m in zip(r24, te24) if m], dd24, 4),
                                          ("namsan", rN, ddN, 2)]:
        blind = [dict(r, e_ref=np.zeros_like(r["v_ref"])) for r in roads_all]
        env = DrivingEnv(blind, dd=ddx, record=True, steer_gain=GAIN)
        sig_d, tau_d = (sig24, tau24) if exp == "2024" else (sigN, tauN)
        T, offc = [], 0
        for k in range(len(blind)):
            for j in range(per_road):
                pol = ResidualDiffusionPolicy(tau=tau_d, sigma=sig_d, lib=fitlib,
                                              seed=7000 + k * 20 + j)
                pol.reset()
                traj, off = rollout(env, pol, k)
                offc += int(off)
                if len(traj) > 60:
                    T.append((traj, roads_all[k]))
        S = [p21.seg_features(p39.symmetric_signals(t, r, ddx, GAIN)) for t, r in T]
        if exp == "2024":
            Hh = []
            for r in roads_all:
                for c in p21.chunk_signals(p20.human_signals(r, ddx)):
                    Hh.append(p21.seg_features(c))
        else:
            Hh = [p21.seg_features(p20.human_signals(r, ddx)) for r in roads_all]
        auc, imp = p43.fair_exam(Hh, S, ret_imp=True)
        top = np.argsort(-imp)[:3]
        out[exp] = dict(auc=float(auc), off=offc / max(len(blind) * per_road, 1))
        print(f"[{exp}] 트랙A GBM AUC={auc:.3f} off={out[exp]['off']:.2f} | 상위: "
              + ", ".join(f"{p43.FEATS[i]}={imp[i]:.3f}" for i in top), flush=True)
    json.dump(out, open(os.path.join(REP, "trackA.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved trackA.json", flush=True)


if __name__ == "__main__":
    (train if "train" in sys.argv else evaluate)()
