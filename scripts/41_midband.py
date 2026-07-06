# -*- coding: utf-8 -*-
"""
41_midband.py  --  캠페인 1단계 수정판: 중대역(50~300m) 피드포워드 주입.

펄스(40) 무효의 원인: 진단 지표 mean|Δe|는 90m 평활을 살아남는 50~300m 중대역 질감
— 10m 펄스는 과녁 밖. 진짜 결핍 = 청크 목표의 중대역이 폐루프 추종에서 감쇠됨.
처방: 청크를 장파(>300m, 기존 e-목표 주입)와 중대역으로 분해, 중대역은 필요 곡률
(목표 2계미분)을 조향에 직접 가산하는 **피드포워드**로 공급(추종 감쇠 우회, 인루프).
사람의 예측적 개루프 조향과 동일 기전.

  python 41_midband.py
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

np.random.seed(0)
GAIN, BASE = 0.012, "rl_multi.zip"
GRID = p20.GRID


def ma(x, w):
    k = np.ones(w) / w
    return np.convolve(np.pad(x, (w // 2, w // 2), mode="edge"), k, mode="valid")[:len(x)]


class MidbandPolicy(p22.SpectralPolicy):
    """장파는 e-목표 주입(기존), 중대역(50~300m)은 곡률 피드포워드로 조향에 직접."""
    def __init__(self, *a, boost=1.0, **kw):
        super().__init__(*a, **kw)
        self.boost = float(boost)

    def reset(self):
        super().reset()
        w = np.asarray(self.w, np.float64)
        w_low = ma(w, 31)                      # >~300m
        w_mid = ma(w - w_low, 5)               # 50~300m (10m 그리드 5점 평활로 <50m 제거)
        self.w = w_low                         # 부모 주입 경로는 장파만
        tgt = self.sigma * w_mid
        d2 = np.gradient(np.gradient(tgt, GRID), GRID)   # 필요 곡률
        self.ff = np.clip(d2 / max(1e-9, GAIN) * self.boost, -0.5, 0.5)

    def __call__(self, obs, env):
        a = super().__call__(obs, env)
        gi = min(int(env.s / GRID), len(self.ff) - 1)
        a[0] = float(np.clip(a[0] + self.ff[gi], -1, 1))
        return a


def collect(ch, roads_sub, dd, sigma, per_road, seed0, boost):
    env = DrivingEnv([dict(r, e_ref=np.zeros_like(r["v_ref"])) for r in roads_sub],
                     dd=dd, record=True, steer_gain=GAIN)
    T, offs = [], 0
    for k in range(len(roads_sub)):
        for j in range(per_road):
            pol = MidbandPolicy(ch["model"], ch["fr"], ch["A"], sigma, lib=ch["lib"],
                                boost=boost, seed=seed0 + k * 20 + j)
            pol.reset()
            traj, off = rollout(env, pol, k)
            offs += int(off)
            if len(traj) > 60:
                T.append((traj, roads_sub[k]))
    return T, offs / max(len(roads_sub) * per_road, 1)


def stats(sigs):
    de = [float(np.mean(np.abs(np.diff(s["e"])))) for s in sigs]
    srr = [float(p20.srr(s["theta"], 0.5)) for s in sigs]
    sd = [float(np.std(s["e"])) for s in sigs]
    return float(np.mean(de)), float(np.mean(srr)), float(np.mean(sd))


def main():
    roads, _, dd = load_roads(os.path.join(CACHE, "env_roads_namsan.npz"))
    roads = trim_roads(roads)
    subj = np.array([r["subject"] for r in roads], "int64")
    tr, va, te = gen_split(subj, seed=0)
    val_roads = [r for r, m in zip(roads, va) if m]
    ch = p34.load_champion(base=BASE)
    cal = json.load(open(os.path.join(REP, "v3_library_2024_multi.json"), encoding="utf-8"))
    sigma = float(cal["sigma"])

    hv = [p20.human_signals(r, dd) for r in val_roads]
    tgt_de, tgt_srr, tgt_sd = stats(hv)
    print(f"목표(val 사람): mean|de|={tgt_de:.4f} SRR={tgt_srr:.1f} SDLP={tgt_sd:.3f}",
          flush=True)

    best = None
    for boost in [0.0, 1.0, 2.0, 4.0]:
        T, off = collect(ch, val_roads, dd, sigma, 2, 61, boost)
        sigs = [p39.symmetric_signals(t, r, dd, GAIN) for t, r in T]
        de, srr, sd = stats(sigs)
        pen = 100.0 * (off > 0.1) + 2.0 * max(0.0, srr / tgt_srr - 1.6)
        gap = abs(de - tgt_de) / tgt_de + 0.5 * abs(sd - tgt_sd) / tgt_sd + pen
        print(f"  boost={boost}: mean|de|={de:.4f} SRR={srr:.1f} SDLP={sd:.3f} "
              f"off={off:.2f} gap={gap:.3f}", flush=True)
        if best is None or gap < best[1]:
            best = (boost, gap)
    b_b = best[0]
    print(f"선택: boost={b_b}", flush=True)

    # σ 재보정(장파만 주입하므로 SDLP 재점검) 1스텝
    T, _ = collect(ch, val_roads, dd, sigma, 2, 71, b_b)
    sigs = [p39.symmetric_signals(t, r, dd, GAIN) for t, r in T]
    _, _, sd_now = stats(sigs)
    sigma2 = float(np.clip(sigma * (tgt_sd / max(sd_now, 1e-6)), 0.02, 1.2))
    print(f"σ 재보정: {sigma:.3f} -> {sigma2:.3f} (SDLP {sd_now:.3f} -> 목표 {tgt_sd:.3f})",
          flush=True)

    # ---- 판정: 전체 도로, 공정시험 ----
    T1, off1 = collect(ch, roads, dd, sigma2, 2, 7000, b_b)
    S1 = [p21.seg_features(p39.symmetric_signals(t, r, dd, GAIN)) for t, r in T1]
    H_units = [p21.seg_features(p20.human_signals(r, dd)) for r in roads]
    rngb = np.random.RandomState(1)
    nu = min(len(H_units), len(S1))
    H = [H_units[i] for i in rngb.choice(len(H_units), nu, replace=False)]
    res1 = p39.auc_variants(H, [S1[i] for i in rngb.choice(len(S1), nu, replace=False)])
    sig_all = [p39.symmetric_signals(t, r, dd, GAIN) for t, r in T1]
    de1, srr1, sd1 = stats(sig_all)
    print(f"[중대역 FF] " + " | ".join(f"{k}={v:.3f}" for k, v in res1.items()), flush=True)
    print(f"  질감: mean|de|={de1:.4f}(사람 {tgt_de:.4f}) SRR={srr1:.1f}({tgt_srr:.1f}) "
          f"SDLP={sd1:.3f}({tgt_sd:.3f}) off={off1:.2f}", flush=True)
    json.dump(dict(boost=b_b, sigma=sigma2, off=float(off1), auc=res1,
                   mean_de=de1, srr=srr1, sdlp=sd1,
                   baseline_gbm_unitcv=0.869),
              open(os.path.join(REP, "midband.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved midband.json", flush=True)


if __name__ == "__main__":
    main()
