# -*- coding: utf-8 -*-
"""
42_preemphasis.py  --  캠페인 1단계 3판: 목표 사전강조 (중대역 β 증폭).

38/40/41 무효의 공통 원인 규명: RL 정책은 강한 외란 제거기 — 정책을 우회하는 조향
가산(펄스·FF)은 전부 상쇄됨. 유일한 유효 경로 = 목표 주입(정책이 스스로 추종).
중대역(50~300m) 결핍은 폐루프 추종 이득 <1 때문 → **주입 목표를 장파+β·중대역으로
사전강조**, β를 val에서 mean|Δe| 목표로 보정(폐루프 감쇠 역보상).

  python 42_preemphasis.py
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
p41 = importlib.import_module("41_midband")

np.random.seed(0)
GAIN, BASE = 0.012, "rl_multi.zip"


class PreemphPolicy(p22.SpectralPolicy):
    """주입 목표 w를 장파 + beta×중대역으로 재합성 (경로는 부모의 e-채널 그대로)."""
    def __init__(self, *a, beta=1.0, **kw):
        super().__init__(*a, **kw)
        self.beta = float(beta)

    def reset(self):
        super().reset()
        w = np.asarray(self.w, np.float64)
        w_low = p41.ma(w, 31)
        w_mid = p41.ma(w - w_low, 5)
        self.w = w_low + self.beta * w_mid


def collect(ch, roads_sub, dd, sigma, per_road, seed0, beta):
    env = DrivingEnv([dict(r, e_ref=np.zeros_like(r["v_ref"])) for r in roads_sub],
                     dd=dd, record=True, steer_gain=GAIN)
    T, offs = [], 0
    for k in range(len(roads_sub)):
        for j in range(per_road):
            pol = PreemphPolicy(ch["model"], ch["fr"], ch["A"], sigma, lib=ch["lib"],
                                beta=beta, seed=seed0 + k * 20 + j)
            pol.reset()
            traj, off = rollout(env, pol, k)
            offs += int(off)
            if len(traj) > 60:
                T.append((traj, roads_sub[k]))
    return T, offs / max(len(roads_sub) * per_road, 1)


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
    tgt_de, tgt_srr, tgt_sd = p41.stats(hv)
    print(f"목표(val 사람): mean|de|={tgt_de:.4f} SRR={tgt_srr:.1f} SDLP={tgt_sd:.3f}",
          flush=True)

    best = None
    for beta in [1.0, 2.0, 4.0, 7.0]:
        T, off = collect(ch, val_roads, dd, sigma, 2, 61, beta)
        sigs = [p39.symmetric_signals(t, r, dd, GAIN) for t, r in T]
        de, srr, sd = p41.stats(sigs)
        pen = 100.0 * (off > 0.15) + 2.0 * max(0.0, srr / tgt_srr - 1.6)
        gap = abs(de - tgt_de) / tgt_de + 0.5 * abs(sd - tgt_sd) / tgt_sd + pen
        print(f"  beta={beta}: mean|de|={de:.4f} SRR={srr:.1f} SDLP={sd:.3f} "
              f"off={off:.2f} gap={gap:.3f}", flush=True)
        if best is None or gap < best[1]:
            best = (beta, gap)
    b_b = best[0]

    # σ 재보정 1스텝 (β가 SDLP를 키우므로)
    T, _ = collect(ch, val_roads, dd, sigma, 2, 71, b_b)
    sigs = [p39.symmetric_signals(t, r, dd, GAIN) for t, r in T]
    _, _, sd_now = p41.stats(sigs)
    sigma2 = float(np.clip(sigma * (tgt_sd / max(sd_now, 1e-6)), 0.02, 1.2))
    T, _ = collect(ch, val_roads, dd, sigma2, 2, 73, b_b)
    sigs = [p39.symmetric_signals(t, r, dd, GAIN) for t, r in T]
    de2, srr2, sd2 = p41.stats(sigs)
    print(f"선택 beta={b_b}, σ {sigma:.3f}->{sigma2:.3f} | val 확인: mean|de|={de2:.4f} "
          f"SRR={srr2:.1f} SDLP={sd2:.3f}", flush=True)

    # ---- 판정: 전체, 공정시험 ----
    T1, off1 = collect(ch, roads, dd, sigma2, 2, 7000, b_b)
    S1 = [p21.seg_features(p39.symmetric_signals(t, r, dd, GAIN)) for t, r in T1]
    H_units = [p21.seg_features(p20.human_signals(r, dd)) for r in roads]
    rngb = np.random.RandomState(1)
    nu = min(len(H_units), len(S1))
    H = [H_units[i] for i in rngb.choice(len(H_units), nu, replace=False)]
    res1 = p39.auc_variants(H, [S1[i] for i in rngb.choice(len(S1), nu, replace=False)])
    sig_all = [p39.symmetric_signals(t, r, dd, GAIN) for t, r in T1]
    de1, srr1, sd1 = p41.stats(sig_all)
    print("[사전강조] " + " | ".join(f"{k}={v:.3f}" for k, v in res1.items()), flush=True)
    print(f"  질감: mean|de|={de1:.4f}({tgt_de:.4f}) SRR={srr1:.1f}({tgt_srr:.1f}) "
          f"SDLP={sd1:.3f}({tgt_sd:.3f}) off={off1:.2f}", flush=True)
    json.dump(dict(beta=b_b, sigma=sigma2, off=float(off1), auc=res1,
                   mean_de=de1, srr=srr1, sdlp=sd1, baseline_gbm_unitcv=0.869),
              open(os.path.join(REP, "preemphasis.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved preemphasis.json", flush=True)


if __name__ == "__main__":
    main()
