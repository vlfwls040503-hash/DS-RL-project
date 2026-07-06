# -*- coding: utf-8 -*-
"""
40_pulse_layer.py  --  캠페인 1단계: 간헐 교정 펄스 레이어 (문헌 기반 미세질감).

근거(§27): 사람 조향 = ~0.4s 종형 각속도 펄스의 간헐 열(Benderius & Markkula 2014;
Markkula 간헐제어; PCM). 가상의 지문 1호(mean|Δe| 과매끈)를 사람과 같은 '기전'으로
치료한다 — 백색잡음이 아니라 이산 사인로브 펄스.

PulsePolicy: 챔피언 조향 위에 사인로브 조향 펄스 중첩.
  - 트리거: 스텝당 확률 p_rate, 인지오차(|주입목표-실제e|)가 클수록 상승(간헐 결정)
  - 진폭: 오차비례 + 신호의존 노이즈, 방향 = 오차 축소 쪽
  - 지속: 0.4~0.8s 균등 (사인 반 주기 = 각도 상승 후 복귀 → 순수 횡위치 넛지)
보정: (p_rate, amp) 2노브를 val에서 mean|Δe| 목표로, SRR 악화 가드 포함.
판정: 감사(39)가 정한 공정시험에서 AUC. 목표 <0.55.

  python 40_pulse_layer.py
"""
import os, json, importlib
import numpy as np

from common import REP, CACHE, gen_split
from driving_env import DrivingEnv, load_roads, rollout, trim_roads, RL_DT

p20 = importlib.import_module("20_profile_eval")
p21 = importlib.import_module("21_validation")
p22 = importlib.import_module("22_v3_spectral")
p34 = importlib.import_module("34_virtual_cohort")
p39 = importlib.import_module("39_discriminator_audit")

np.random.seed(0)
GAIN, BASE = 0.012, "rl_multi.zip"
GRID = p20.GRID
IDX_E = p20.IDX_E


class PulsePolicy(p22.SpectralPolicy):
    """챔피언 + 간헐 사인로브 교정 펄스 (Benderius & Markkula 기전)."""
    def __init__(self, *a, p_rate=0.02, amp=0.08, dur_s=(0.4, 0.8), **kw):
        super().__init__(*a, **kw)
        self.p_rate, self.amp = float(p_rate), float(amp)
        self.dur_s = dur_s
        self.pulses = []              # [ (남은스텝, 총스텝, 부호크기) ]

    def reset(self):
        super().reset()
        self.pulses = []

    def __call__(self, obs, env):
        a = super().__call__(obs, env)
        # 인지 오차: 주입 목표 대비 실제 e (부모가 obs를 보정하므로 근사로 원시 e 사용)
        gi = env.s / GRID
        i0 = min(int(gi), len(self.w) - 2)
        target = self.sigma * self.w[i0] + self.b_bias
        err = float(env.e - np.clip(target, -1.0, 1.0))
        # 간헐 트리거: 기본율 + 오차 가중
        p = self.p_rate * (1.0 + 4.0 * min(abs(err), 0.5))
        if self.rng.rand() < p:
            T = int(self.rng.uniform(*self.dur_s) / RL_DT)
            mag = -np.sign(err) * self.amp * (0.5 + abs(err)) \
                * (1.0 + 0.3 * self.rng.randn())
            self.pulses.append([T, T, mag])
        # 활성 펄스 합산 (사인로브: 0→peak→0)
        add = 0.0
        keep = []
        for pl in self.pulses:
            rem, T, mag = pl
            phase = 1.0 - rem / T
            add += mag * np.sin(np.pi * phase)
            pl[0] -= 1
            if pl[0] > 0:
                keep.append(pl)
        self.pulses = keep
        a[0] = float(np.clip(a[0] + add, -1, 1))
        return a


def collect(ch, roads_sub, dd, sigma, per_road, seed0, p_rate, amp):
    env = DrivingEnv([dict(r, e_ref=np.zeros_like(r["v_ref"])) for r in roads_sub],
                     dd=dd, record=True, steer_gain=GAIN)
    T, offs = [], 0
    for k in range(len(roads_sub)):
        for j in range(per_road):
            pol = PulsePolicy(ch["model"], ch["fr"], ch["A"], sigma, lib=ch["lib"],
                              p_rate=p_rate, amp=amp, seed=seed0 + k * 20 + j)
            pol.reset()
            traj, off = rollout(env, pol, k)
            offs += int(off)
            if len(traj) > 60:
                T.append((traj, roads_sub[k]))
    return T, offs / max(len(roads_sub) * per_road, 1)


def micro_stats(sig):
    de = np.diff(sig["e"])
    return float(np.mean(np.abs(de))), float(p20.srr(sig["theta"], 0.5))


def main():
    roads, _, dd = load_roads(os.path.join(CACHE, "env_roads_namsan.npz"))
    roads = trim_roads(roads)
    subj = np.array([r["subject"] for r in roads], "int64")
    tr, va, te = gen_split(subj, seed=0)
    val_roads = [r for r, m in zip(roads, va) if m]
    ch = p34.load_champion(base=BASE)
    cal = json.load(open(os.path.join(REP, "v3_library_2024_multi.json"), encoding="utf-8"))
    sigma = float(cal["sigma"])

    # 사람 목표 (val): 대칭 산출 기준의 mean|Δe| 와 SRR
    hv = [p20.human_signals(r, dd) for r in val_roads]
    tgt_de = float(np.mean([np.mean(np.abs(np.diff(h["e"]))) for h in hv]))
    tgt_srr = float(np.mean([p20.srr(h["theta"], 0.5) for h in hv]))
    print(f"목표(val, 사람): mean|de|={tgt_de:.4f} SRR={tgt_srr:.1f}", flush=True)

    # ---- (p_rate, amp) 그리드 val 보정 ----
    best = None
    for p_rate in [0.0, 0.015, 0.03]:
        for amp in ([0.0] if p_rate == 0.0 else [0.05, 0.10]):
            T, off = collect(ch, val_roads, dd, sigma, 2, 61, p_rate, amp)
            sigs = [p39.symmetric_signals(t, r, dd, GAIN) for t, r in T]
            de = np.mean([micro_stats(s)[0] for s in sigs])
            srr = np.mean([micro_stats(s)[1] for s in sigs])
            pen = 100.0 * (off > 0.1) + 2.0 * max(0.0, srr / tgt_srr - 1.5)
            gap = abs(de - tgt_de) / tgt_de + pen
            print(f"  p={p_rate} A={amp}: mean|de|={de:.4f} SRR={srr:.1f} off={off:.2f} "
                  f"gap={gap:.3f}", flush=True)
            if best is None or gap < best[2]:
                best = (p_rate, amp, gap)
    p_b, a_b, _ = best
    print(f"선택: p_rate={p_b} amp={a_b}", flush=True)

    # ---- 판정: 전체 도로, 공정시험(대칭+GBM+unitCV — 39의 auc_variants) ----
    T, off = collect(ch, roads, dd, sigma, 2, 7000, p_b, a_b)
    S_units = [p21.seg_features(p39.symmetric_signals(t, r, dd, GAIN)) for t, r in T]
    T0, _ = collect(ch, roads, dd, sigma, 2, 7000, 0.0, 0.0)
    S0_units = [p21.seg_features(p39.symmetric_signals(t, r, dd, GAIN)) for t, r in T0]
    H_units = [p21.seg_features(p20.human_signals(r, dd)) for r in roads]
    rngb = np.random.RandomState(1)
    nu = min(len(H_units), len(S_units), len(S0_units))
    H = [H_units[i] for i in rngb.choice(len(H_units), nu, replace=False)]
    idx = rngb.choice(min(len(S_units), len(S0_units)), nu, replace=False)
    res0 = p39.auc_variants(H, [S0_units[i] for i in idx])
    res1 = p39.auc_variants(H, [S_units[i] for i in idx])
    print("[펄스 전] " + " | ".join(f"{k}={v:.3f}" for k, v in res0.items()), flush=True)
    print("[펄스 후] " + " | ".join(f"{k}={v:.3f}" for k, v in res1.items()), flush=True)
    json.dump(dict(p_rate=p_b, amp=a_b, off=off, before=res0, after=res1,
                   tgt_de=tgt_de, tgt_srr=tgt_srr),
              open(os.path.join(REP, "pulse_layer.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved pulse_layer.json", flush=True)


if __name__ == "__main__":
    main()
