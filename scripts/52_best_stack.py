# -*- coding: utf-8 -*-
"""
52_best_stack.py  --  최강 부품 총결합: DAgger-BC 응답 + 청크 배회 + 디퓨전 잔차.

현직(46d, 0.770/0.817)의 응답함수를 운반자로, 트랙A의 살아난 잔차를 소량(τ) 가산.
τ∈{0, 0.15, 0.3} × 도메인별 σ-solve — τ=0이 곧 현직이므로 하락만 채택(단조 안전).

  python 52_best_stack.py
"""
import os, json, importlib
import numpy as np
import torch
import torch.nn as nn

from common import ART, REP, CACHE, gen_split
from driving_env import DrivingEnv, load_roads, rollout, trim_roads, OBS_DIM

p20 = importlib.import_module("20_profile_eval")
p21 = importlib.import_module("21_validation")
p39 = importlib.import_module("39_discriminator_audit")
p43 = importlib.import_module("43_gbm_anatomy")
p44 = importlib.import_module("44_bc_native")
p51 = importlib.import_module("51_trackA_diffusion")

np.random.seed(0); torch.manual_seed(0)
GAIN = 0.012
IDX_E = p20.IDX_E


class BestStackPolicy(p51.ResidualDiffusionPolicy):
    """운반자=DAgger-BC(주입 obs), 배회=청크(부모), 질감=디퓨전 잔차 스트림(부모)."""
    def __init__(self, bc, beta=1.0, **kw):
        super().__init__(**kw)
        self.bc = bc
        self.beta = float(beta)

    def reset(self):
        super().reset()
        if self.w is not None and self.beta != 1.0:
            import importlib as _il
            _ma = _il.import_module("41_midband").ma
            w_low = _ma(self.w, 31)
            self.w = w_low + self.beta * _ma(self.w - w_low, 5)

    def __call__(self, obs, env):
        # 잔차 스트림 갱신 (부모 로직 재사용을 위해 가이드 조건만 구성)
        if self.buf is None or self.rptr >= p51.H - 8:
            gd_full = self._guide(env)
            c = np.concatenate([obs, self.hist, gd_full]).astype("float32")
            cond = torch.from_numpy(((c - self.mu) / self.sd)[None]).to(p51.DEV)
            new = self.tau * self._sample(cond)
            if self.buf is not None:
                for kk in range(8):
                    w = (kk + 1) / 9.0
                    new[kk] = (1 - w) * self.buf[min(self.rptr + kk, p51.H - 1)] + w * new[kk]
            self.buf, self.rptr = new, 0
        # 운반자: DAgger-BC (배회 목표를 e-채널 주입)
        b = self._btarget(env.s)
        o = obs.copy()
        o[IDX_E] = o[IDX_E] - float(np.clip(b, -1.0, 1.0))
        a = self.bc.predict(o)[0]
        st = float(np.clip(a[0] + self.buf[self.rptr], -1, 1))
        self.hist = np.concatenate([self.hist[1:], [self.buf[self.rptr]]]).astype(np.float32)
        self.rptr += 1
        i = min(int(env.s / env.dd), len(env.road["v_ref"]) - 1)
        acc = float(np.clip((env.road["v_ref"][i] - env.v) / 3.0, -1, 1))
        return np.array([st, acc], np.float32)


def main():
    net = nn.Sequential(nn.Linear(OBS_DIM, 64), nn.ReLU(),
                        nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2), nn.Tanh())
    net.load_state_dict(torch.load(os.path.join(ART, "bc_dagger46_inj.pt")))
    net.eval()
    bc = p44.BCAdapter(net)

    r24, _, dd24 = load_roads(os.path.join(CACHE, "env_roads_2024.npz"))
    r24 = trim_roads(r24)
    s24 = np.array([r["subject"] for r in r24], "int64")
    _, va24, te24 = gen_split(s24, seed=0)
    fitlib = []
    for r in [r for r, m in zip(r24, ~te24) if m]:
        for ch_ in p21.chunk_signals(p20.human_signals(r, dd24)):
            x = np.asarray(ch_["e"], np.float64); x -= x.mean()
            if len(x) >= 400 and x.std() > 1e-3:
                fitlib.append((x / x.std()).astype(np.float64))

    out = {}
    for exp, per_road in [("namsan", 2), ("2024", 4)]:
        roads, _, dd = load_roads(os.path.join(CACHE, f"env_roads_{exp}.npz"))
        roads = trim_roads(roads)
        sx = np.array([r["subject"] for r in roads], "int64")
        trx, vax, tex = gen_split(sx, seed=0)
        val_r = [r for r, m in zip(roads, vax) if m]
        val_b = [dict(r, e_ref=np.zeros_like(r["v_ref"])) for r in val_r]
        env_v = DrivingEnv(val_b, dd=dd, record=True, steer_gain=GAIN)
        hv = [p20.human_signals(r, dd) for r in val_r]
        tgt_sd = float(np.mean([np.std(h["e"]) for h in hv]))
        tgt_de = float(np.mean([np.mean(np.abs(np.diff(h["e"]))) for h in hv]))
        best = None
        betas = [1.0, 2.0] if exp == "namsan" else [1.0]
        for beta in betas:
          for tau in [0.0, 0.15, 0.3]:
            s0 = 0.2
            for it in range(2):
                sds, des = [], []
                for k in range(len(val_r)):
                    pol = BestStackPolicy(bc, beta=beta, tau=tau, sigma=s0, lib=fitlib,
                                          seed=90 + k)
                    pol.reset()
                    traj, _ = rollout(env_v, pol, k)
                    if len(traj) > 60:
                        sg = p39.symmetric_signals(traj, val_r[k], dd, GAIN)
                        sds.append(float(np.std(sg["e"])))
                        des.append(float(np.mean(np.abs(np.diff(sg["e"])))))
                s0 = float(np.clip(s0 * tgt_sd / max(np.mean(sds), 1e-6), 0.03, 1.0))
            gap = abs(np.mean(des) - tgt_de) / tgt_de \
                + 0.5 * abs(np.mean(sds) - tgt_sd) / tgt_sd
            print(f"  [{exp}] tau={tau} σ={s0:.3f}: |de|={np.mean(des):.4f}({tgt_de:.4f}) "
                  f"SDLP={np.mean(sds):.3f}({tgt_sd:.3f}) gap={gap:.3f}", flush=True)
            if best is None or gap < best[3]:
                best = (tau, s0, beta, gap)
        tau_d, sig_d, beta_d, _ = best
        exam_roads = roads if exp == "namsan" else [r for r, m in zip(roads, tex) if m]
        blind = [dict(r, e_ref=np.zeros_like(r["v_ref"])) for r in exam_roads]
        env = DrivingEnv(blind, dd=dd, record=True, steer_gain=GAIN)
        T, offc = [], 0
        for k in range(len(blind)):
            for j in range(per_road):
                pol = BestStackPolicy(bc, beta=beta_d, tau=tau_d, sigma=sig_d,
                                      lib=fitlib, seed=7000 + k * 20 + j)
                pol.reset()
                traj, off = rollout(env, pol, k)
                offc += int(off)
                if len(traj) > 60:
                    T.append((traj, exam_roads[k]))
        S = [p21.seg_features(p39.symmetric_signals(t, r, dd, GAIN)) for t, r in T]
        if exp == "2024":
            Hh = []
            for r in exam_roads:
                for c in p21.chunk_signals(p20.human_signals(r, dd)):
                    Hh.append(p21.seg_features(c))
        else:
            Hh = [p21.seg_features(p20.human_signals(r, dd)) for r in exam_roads]
        auc, imp = p43.fair_exam(Hh, S, ret_imp=True)
        top = np.argsort(-imp)[:3]
        out[exp] = dict(auc=float(auc), tau=tau_d, sigma=sig_d, beta=beta_d,
                        off=offc / max(len(blind) * per_road, 1))
        print(f"[{exp}] 총결합 GBM AUC={auc:.3f} (현직 {'0.817' if exp=='namsan' else '0.770'}) "
              f"off={out[exp]['off']:.2f} | 상위: "
              + ", ".join(f"{p43.FEATS[i]}={imp[i]:.3f}" for i in top), flush=True)
    json.dump(out, open(os.path.join(REP, "best_stack.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved best_stack.json", flush=True)


if __name__ == "__main__":
    main()
