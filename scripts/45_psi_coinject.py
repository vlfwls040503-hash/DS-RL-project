# -*- coding: utf-8 -*-
"""
45_psi_coinject.py  --  수술 4호: ψ 동시주입 (물리 일관 참조).

현행 주입은 e-채널만 이동 → 정책은 "헤딩 없이 위치오차 발생"이라는 모순 참조에
급반응(→ 날카로운 횡속 피크 = GBM 주무기). 목표의 기울기 ψ_t = σ·dw/ds를 ψ-채널
(obs[3], ×5 스케일)에도 일관 주입해 예측적·완만한 추종을 유도.

  python 45_psi_coinject.py
"""
import os, json, importlib
import numpy as np

from common import REP, CACHE, gen_split
from driving_env import DrivingEnv, load_roads, rollout, trim_roads

p20 = importlib.import_module("20_profile_eval")
p21 = importlib.import_module("21_validation")
p22 = importlib.import_module("22_v3_spectral")
p34 = importlib.import_module("34_virtual_cohort")
p39 = importlib.import_module("39_discriminator_audit")
p43 = importlib.import_module("43_gbm_anatomy")

np.random.seed(0)
GAIN, BASE = 0.012, "rl_multi.zip"
GRID = p20.GRID
IDX_PSI, PSI_SCALE = 3, 5.0


class PsiPolicy(p22.SpectralPolicy):
    def reset(self):
        super().reset()
        self.dw = np.gradient(np.asarray(self.w, np.float64), GRID)

    def __call__(self, obs, env):
        gi = env.s / GRID
        i0 = min(int(gi), len(self.w) - 2)
        f = gi - int(gi)
        b = self.sigma * ((1 - f) * self.w[i0] + f * self.w[i0 + 1])
        psi_t = self.sigma * self.dw[i0]
        o = obs.copy()
        o[p20.IDX_E] = o[p20.IDX_E] - float(np.clip(b + self.b_bias, -1.0, 1.0))
        o[IDX_PSI] = o[IDX_PSI] - float(np.clip(psi_t, -0.2, 0.2)) * PSI_SCALE
        steer = float(self.m.predict(o, deterministic=True)[0][0])
        i = min(int(env.s / env.dd), len(env.road["v_ref"]) - 1)
        from common import RL_A_MAX
        acc = float(np.clip(self.k_v * (self.v_scale * env.road["v_ref"][i] - env.v)
                            / RL_A_MAX, -1, 1))
        return np.array([np.clip(steer, -1, 1), acc], np.float32)


def main():
    ch = p34.load_champion(base=BASE)
    cal = json.load(open(os.path.join(REP, "v3_library_2024_multi.json"), encoding="utf-8"))
    sigma = float(cal["sigma"])

    roadsN, _, ddN = load_roads(os.path.join(CACHE, "env_roads_namsan.npz"))
    roadsN = trim_roads(roadsN)
    sN = np.array([r["subject"] for r in roadsN], "int64")
    trN, vaN, teN = gen_split(sN, seed=0)
    valN = [r for r, m in zip(roadsN, vaN) if m]
    env_v = DrivingEnv([dict(r, e_ref=np.zeros_like(r["v_ref"])) for r in valN],
                       dd=ddN, record=True, steer_gain=GAIN)
    hv = [p20.human_signals(r, ddN) for r in valN]
    tgt_sd = float(np.mean([np.std(h["e"]) for h in hv]))

    def probe(sig):
        sds, offs = [], 0
        for k in range(len(valN)):
            pol = PsiPolicy(ch["model"], ch["fr"], ch["A"], sig, lib=ch["lib"], seed=k)
            pol.reset()
            traj, off = rollout(env_v, pol, k)
            offs += int(off)
            if len(traj) > 60:
                sds.append(float(np.std(p20.rl_signals(traj, gain=GAIN)["e"])))
        return float(np.mean(sds)), offs / len(valN)

    sd1, off1 = probe(sigma)
    sigma2 = float(np.clip(sigma * (tgt_sd / max(sd1, 1e-6)), 0.02, 1.2))
    print(f"[게이트] val: SDLP {sd1:.3f} off {off1:.2f} | σ {sigma:.3f}->{sigma2:.3f}",
          flush=True)
    if off1 > 0.3:
        print("GATE FAIL", flush=True)
        return

    out = {}
    for exp, per_road in [("2024", 4), ("namsan", 2)]:
        roads, _, dd = load_roads(os.path.join(CACHE, f"env_roads_{exp}.npz"))
        roads = trim_roads(roads)
        if exp == "2024":
            s2 = np.array([r["subject"] for r in roads], "int64")
            _, _, te2 = gen_split(s2, seed=0)
            roads = [r for r, m in zip(roads, te2) if m]
        env = DrivingEnv([dict(r, e_ref=np.zeros_like(r["v_ref"])) for r in roads],
                         dd=dd, record=True, steer_gain=GAIN)
        T, offc = [], 0
        for k in range(len(roads)):
            for j in range(per_road):
                pol = PsiPolicy(ch["model"], ch["fr"], ch["A"], sigma2, lib=ch["lib"],
                                seed=7000 + k * 20 + j)
                pol.reset()
                traj, off = rollout(env, pol, k)
                offc += int(off)
                if len(traj) > 60:
                    T.append((traj, roads[k]))
        S = [p21.seg_features(p39.symmetric_signals(t, r, dd, GAIN)) for t, r in T]
        if exp == "2024":
            H = []
            for r in roads:
                for c in p21.chunk_signals(p20.human_signals(r, dd)):
                    H.append(p21.seg_features(c))
        else:
            H = [p21.seg_features(p20.human_signals(r, dd)) for r in roads]
        rngb = np.random.RandomState(1)
        nu = min(len(H), len(S))
        Hb = [H[i] for i in rngb.choice(len(H), nu, replace=False)]
        Sb = [S[i] for i in rngb.choice(len(S), nu, replace=False)]
        auc, imp = p43.fair_exam(Hb, Sb, ret_imp=True)
        top = np.argsort(-imp)[:3]
        out[exp] = dict(auc=float(auc), off=offc / max(len(roads) * per_road, 1))
        print(f"[{exp}] ψ동시주입 GBM AUC={auc:.3f} off={out[exp]['off']:.2f} | 상위: "
              + ", ".join(f"{p43.FEATS[i]}={imp[i]:.3f}" for i in top), flush=True)
    out["sigma"] = sigma2
    json.dump(out, open(os.path.join(REP, "psi_coinject.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved psi_coinject.json", flush=True)


if __name__ == "__main__":
    main()
