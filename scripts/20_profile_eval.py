# -*- coding: utf-8 -*-
"""
20_profile_eval.py  --  frame/segment-level MULTI-SIGNAL profile evaluation (v2.1).

Driver v2.1 (v2 + 잔 교정 개선):
  - lateral : RL deterministic steering + LOW-FREQUENCY OU noise (tau_ou, colored)
              + FIRST-ORDER STEERING LAG tau_steer (신경근/조향계 지연 모사)
              -> tau_steer는 val에서 사람 SRR_0.5에 맞춰 보정 (잔 교정 과다 억제).
  - longitudinal: PD speed controller on v_ref (속도는 학습·평가 축에서 제외).

Metrics (SDLP 단독지표 금지 — 사용자 지시):
  frame(10m)/segment(200m): 횡위치 e, 상대 yaw ψ, 횡속도, 횡가속 — W1/corr/RMSE
  texture: 횡위치 주파장, 횡속도 반전율, ★SRR_0.5·SRR_2 (조향휠 각도, /km)
  (조향휠 환산: wheel_deg = kappa * WHEELBASE(2.75m) * STEER_RATIO(15, 가정) * 57.3)

  python 20_profile_eval.py --exp 2024
"""
import os, json, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from common import ART, REP, CACHE, gen_split, wasserstein1d, RL_A_MAX, RL_STEER_GAIN, RL_DT
from driving_env import DrivingEnv, load_roads, rollout, trim_roads, _smooth

for fam in ["Malgun Gothic", "NanumGothic", "Gulim"]:
    try:
        plt.rcParams["font.family"] = fam; break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(REP, "figs"); os.makedirs(FIG, exist_ok=True)
np.random.seed(0); torch.manual_seed(0)

GRID = 10.0
SEG = 200.0
WL_BAND = (40.0, 3000.0)
WHEELBASE, STEER_RATIO = 2.75, 15.0            # ratio: assumption, documented
K2DEG = WHEELBASE * STEER_RATIO * 57.29578     # kappa(1/m) -> steering wheel deg


IDX_E = 2   # observation layout: [v, v_ref, e, psi, halfwidth, slope, curv...]


class HumanlikePolicy:
    """v2.3 — TARGET-WANDER injection: the OU process shifts the *intended lane position*
    (obs e-channel), not the steering command. The policy then steers smoothly toward a
    slowly wandering target — lateral spread appears WITHOUT steering fights.
    (History: steering-noise injection needed large sigma to beat the stabilizing policy
     -> SRR 2.5-11x human. First-order lag destabilized the loop. Deadband helped ~16%.
     Root cause was the injection POINT: human variability originates at intent level.)
    Longitudinal: PD speed controller on v_ref."""
    def __init__(self, model, e_tau=300.0, e_sigma=0.25, e_lpf=50.0, b_max=0.6,
                 b_bias=0.0, v_scale=1.0,
                 steer_sigma=0.0, steer_db=0.0, k_v=1.0, seed=0):
        self.m = model
        self.e_tau, self.e_sigma, self.b_max = float(e_tau), float(e_sigma), float(b_max)
        self.e_lpf = float(e_lpf)      # 2nd stage: smooths the OU (OU is Brownian-rough;
        self.b_bias = float(b_bias)    # driver trait: preferred lane offset (m)
        self.v_scale = float(v_scale)  # driver trait: speed-preference ratio
        self.steer_sigma, self.steer_db, self.k_v = float(steer_sigma), float(steer_db), k_v
        self.rng = np.random.RandomState(seed)   # tracking a rough target forces steering
        self.b, self.bs, self.x, self.u = 0.0, 0.0, 0.0, None   # reversals -> SRR blow-up)

    def reset(self):
        self.b = float(np.clip(self.rng.randn() * self.e_sigma, -self.b_max, self.b_max))
        self.bs = self.b
        self.x = self.rng.randn() * self.steer_sigma
        self.u = None

    def __call__(self, obs, env):
        dm = max(env.v, 0.1) * RL_DT
        # intent wander (meters, OU with correlation length e_tau) + low-pass (smooth intent)
        self.b += -self.b * dm / self.e_tau + self.e_sigma * np.sqrt(2 * dm / self.e_tau) * self.rng.randn()
        if self.e_lpf > 1e-6:
            self.bs += (self.b - self.bs) * min(dm / self.e_lpf, 1.0)
        else:
            self.bs = self.b
        o = obs.copy()
        o[IDX_E] = o[IDX_E] - float(np.clip(self.bs + self.b_bias, -1.0, 1.0))
        tgt = float(self.m.predict(o, deterministic=True)[0][0])
        if self.steer_sigma > 1e-9:                # optional residual steering noise
            self.x += -self.x * dm / self.e_tau + self.steer_sigma * np.sqrt(2 * dm / self.e_tau) * self.rng.randn()
            tgt += self.x
        if self.steer_db > 1e-9:                   # optional satisficing deadband
            if self.u is None or abs(tgt - self.u) >= self.steer_db:
                self.u = tgt
            steer = self.u
        else:
            steer = tgt
        i = min(int(env.s / env.dd), len(env.road["v_ref"]) - 1)
        acc = float(np.clip(self.k_v * (self.v_scale * env.road["v_ref"][i] - env.v) / RL_A_MAX, -1, 1))
        return np.array([np.clip(steer, -1, 1), acc], np.float32)


# ---------------- signals ----------------
def human_signals(road, dd_road):
    e = _smooth(np.asarray(road["e_ref"], np.float64))
    v = _smooth(np.asarray(road["v_ref"], np.float64))
    s = np.arange(len(e)) * dd_road
    psi = np.gradient(e, dd_road)
    kap = np.asarray(road["curv"], np.float64) + np.gradient(psi, dd_road)
    g = np.arange(0, s[-1], GRID)
    def rs(x): return np.interp(g, s, x)
    e, v, psi, kap = rs(e), rs(v), rs(psi), rs(kap)
    return dict(s=g, e=e, psi=psi, latv=v * psi, kappa=kap, lata=v * v * kap,
                theta=kap * K2DEG)


def rl_signals(traj):
    s0 = traj[:, 0]
    g = np.arange(s0[0], s0[-1], GRID)
    def rs(col): return np.interp(g, s0, traj[:, col])
    e, v, psi, steer = rs(1), rs(2), rs(4), rs(5)
    kap = steer * RL_STEER_GAIN
    return dict(s=g, e=e, psi=psi, latv=v * psi, kappa=kap, lata=v * v * kap,
                theta=kap * K2DEG)


def wavelength(e, grid=GRID):
    x = np.asarray(e, np.float64); x = x - x.mean()
    if len(x) < 32:
        return float("nan")
    F = np.fft.rfft(x * np.hanning(len(x)))
    fr = np.fft.rfftfreq(len(x), d=grid)
    P = np.abs(F) ** 2
    m = (fr > 1.0 / WL_BAND[1]) & (fr < 1.0 / WL_BAND[0])
    if not m.any() or P[m].sum() <= 0:
        return float("nan")
    return float(np.sum((1.0 / fr[m]) * P[m]) / np.sum(P[m]))


def reversal_rate(latv, grid=GRID, thr=0.02):
    q = np.where(latv > thr, 1, np.where(latv < -thr, -1, 0))
    q = q[q != 0]
    n = int(np.sum(q[1:] != q[:-1])) if len(q) > 1 else 0
    km = len(latv) * grid / 1000.0
    return n / km if km > 0 else float("nan")


def srr(theta_deg, thr, grid=GRID):
    """Steering reversal rate (/km): direction changes with amplitude >= thr deg (hysteresis)."""
    th = np.asarray(theta_deg, np.float64)
    if len(th) < 3:
        return float("nan")
    cnt, ref, direction = 0, th[0], 0
    for x in th[1:]:
        d = x - ref
        if direction >= 0 and d <= -thr:
            cnt += 1; direction = -1; ref = x
        elif direction <= 0 and d >= thr:
            cnt += 1; direction = 1; ref = x
        else:
            ref = max(ref, x) if direction >= 0 else min(ref, x)
    km = len(th) * grid / 1000.0
    return cnt / km


def mean_psd(sig_list, grid=GRID, n=512):
    fr = np.fft.rfftfreq(n, d=grid)
    acc = []
    for x in sig_list:
        x = np.asarray(x, np.float64)
        if len(x) < n:
            continue
        seg = x[:n] - x[:n].mean()
        acc.append(np.abs(np.fft.rfft(seg * np.hanning(n))) ** 2)
    return fr, (np.mean(acc, axis=0) if acc else np.full(len(fr), np.nan))


SIGS = [("e", "횡위치 e(m)"), ("psi", "상대 yaw ψ(rad)"), ("latv", "횡속도(m/s)"),
        ("lata", "횡가속(m/s²)")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="2024")
    args = ap.parse_args()
    exp = args.exp

    roads, subject, dd = load_roads(os.path.join(CACHE, f"env_roads_{exp}.npz"))
    roads = trim_roads(roads)
    subject = np.array([r["subject"] for r in roads], "int64")
    tr, va, te = gen_split(subject, seed=0)
    val_roads = [r for r, m in zip(roads, va) if m]
    test_roads = [r for r, m in zip(roads, te) if m]
    model = PPO.load(os.path.join(ART, f"rl_{exp}.zip"), device="cpu")
    print(f"[{exp}] val={len(val_roads)} test={len(test_roads)}", flush=True)

    # ---- human targets on val ----
    hv = [human_signals(r, dd) for r in val_roads]
    t_std = float(np.mean([h["e"].std() for h in hv]))
    t_wl = float(np.nanmean([wavelength(h["e"]) for h in hv]))
    t_srr = float(np.mean([srr(h["theta"], 0.5) for h in hv]))
    print(f"val human targets: e-std={t_std:.3f} wl={t_wl:.0f}m SRR0.5={t_srr:.1f}/km", flush=True)

    env_v = DrivingEnv(val_roads, dd=dd, record=True)

    def probe(etau, esig, lpf):
        pol = HumanlikePolicy(model, e_tau=etau, e_sigma=esig, e_lpf=lpf, seed=0)
        stds, wls, srrs, off = [], [], [], 0
        for k in range(len(val_roads)):
            pol.reset()
            traj, o = rollout(env_v, pol, k)
            off += int(o)
            if len(traj) > 50:
                sg = rl_signals(traj)
                stds.append(sg["e"].std()); wls.append(wavelength(sg["e"]))
                srrs.append(srr(sg["theta"], 0.5))
        return float(np.mean(stds)), float(np.nanmean(wls)), float(np.mean(srrs)), off

    # JOINT 2D calibration over (e_tau, e_lpf) — 파장(wl)과 SRR을 동시에 맞춤.
    # (v2.4 단일축 보정의 트레이드오프: lpf가 SRR을 내리면서 파장을 밀어올림 → 2D 탐색으로 균형)
    # std는 sigma에 ~선형이므로 콤보당 1회 프로브 후 해석적 재스케일로 점수화(시간 절약).
    best = None
    for tau in [75.0, 150.0, 300.0]:
        for lpf in [50.0, 100.0, 150.0]:
            sd, wl, s5, off = probe(tau, t_std, lpf)
            score = (abs(wl - t_wl) / t_wl + abs(s5 - t_srr) / t_srr
                     + (100.0 if off else 0.0))
            print(f"  tau={tau:4.0f} lpf={lpf:4.0f} -> wl={wl:.0f} SRR={s5:.1f} "
                  f"std={sd:.3f} off={off} score={score:.3f}", flush=True)
            if best is None or score < best[0]:
                best = (score, tau, lpf, sd)
    _, tau_s, lpf_s, sd_l = best

    # sigma rescale to e-std + guarded verify
    sig_s = float(np.clip(t_std * t_std / max(sd_l, 1e-6), 0.05, 1.2))
    sd_f, wl_f, s5_f, off_f = probe(tau_s, sig_s, lpf_s)
    if not (sd_f < 1.5 * t_std and off_f == 0):
        sig_s, (sd_f, wl_f, s5_f, off_f) = t_std, probe(tau_s, t_std, lpf_s)
    db_s = 0.0
    print(f"calibrated: e_tau={tau_s:.0f}m e_sigma={sig_s:.3f}m e_lpf={lpf_s:.0f}m -> "
          f"std={sd_f:.3f} wl={wl_f:.0f} SRR0.5={s5_f:.1f} (사람 wl={t_wl:.0f} SRR={t_srr:.1f})", flush=True)

    # ---- test rollouts ----
    env = DrivingEnv(test_roads, dd=dd, record=True)
    pol = HumanlikePolicy(model, e_tau=tau_s, e_sigma=sig_s, e_lpf=lpf_s, seed=1)
    H, R, offs = [], [], 0
    for k in range(len(test_roads)):
        H.append(human_signals(test_roads[k], dd))
        pol.reset()
        traj, off = rollout(env, pol, k)
        offs += int(off)
        R.append(rl_signals(traj))
    print(f"test rollouts: offroad {offs}/{len(test_roads)}", flush=True)

    # ---- frame metrics ----
    res = {}
    for key, _ in SIGS:
        h_all = np.concatenate([h[key] for h in H])
        r_all = np.concatenate([r[key] for r in R])
        cors, rmses = [], []
        for h, r in zip(H, R):
            n = min(len(h[key]), len(r[key]))
            if n > 50:
                a, b = h[key][:n], r[key][:n]
                cors.append(float(np.corrcoef(a, b)[0, 1]))
                rmses.append(float(np.sqrt(np.mean((a - b) ** 2))))
        res[key] = dict(w1=float(wasserstein1d(r_all, h_all)),
                        h_mean=float(h_all.mean()), h_std=float(h_all.std()),
                        r_mean=float(r_all.mean()), r_std=float(r_all.std()),
                        profile_corr=float(np.mean(cors)), profile_rmse=float(np.mean(rmses)))
    tex = dict(wl_h=float(np.nanmean([wavelength(h["e"]) for h in H])),
               wl_r=float(np.nanmean([wavelength(r["e"]) for r in R])),
               rev_h=float(np.mean([reversal_rate(h["latv"]) for h in H])),
               rev_r=float(np.mean([reversal_rate(r["latv"]) for r in R])),
               srr05_h=float(np.mean([srr(h["theta"], 0.5) for h in H])),
               srr05_r=float(np.mean([srr(r["theta"], 0.5) for r in R])),
               srr2_h=float(np.mean([srr(h["theta"], 2.0) for h in H])),
               srr2_r=float(np.mean([srr(r["theta"], 2.0) for r in R])))

    # ---- segment metrics ----
    seg_pts = int(SEG / GRID)
    seg_h, seg_r = {k: [] for k, _ in SIGS}, {k: [] for k, _ in SIGS}
    for h, r in zip(H, R):
        n = min(len(h["e"]), len(r["e"]))
        for a in range(0, n - seg_pts, seg_pts):
            for key, _ in SIGS:
                seg_h[key].append(float(np.std(h[key][a:a + seg_pts])))
                seg_r[key].append(float(np.std(r[key][a:a + seg_pts])))
    seg_stats = {k: dict(corr=float(np.corrcoef(seg_h[k], seg_r[k])[0, 1]),
                         w1=float(wasserstein1d(np.array(seg_r[k]), np.array(seg_h[k]))))
                 for k, _ in SIGS}

    # ---- figures ----
    h0, r0 = H[0], R[0]
    n0 = min(len(h0["s"]), len(r0["s"]), 400)
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for ax, (key, name) in zip(axes, SIGS):
        ax.plot(h0["s"][:n0], h0[key][:n0], color="#185FA5", lw=1.2, label="사람")
        ax.plot(r0["s"][:n0], r0[key][:n0], color="#7F77DD", lw=1.1, alpha=.9, label="RL v2.4")
        ax.set_ylabel(name); ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("거리(m)")
    fig.suptitle(f"프레임 프로파일 v2.4 (의도-방황 OU τ={tau_s:.0f}m·σ={sig_s:.2f}m, 속도=컨트롤러)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"fig_profile_signals_{exp}.png"), dpi=120); plt.close(fig)

    fr, Ph = mean_psd([h["e"] for h in H]); _, Pr = mean_psd([r["e"] for r in R])
    fig, ax = plt.subplots(figsize=(7, 4))
    m = fr > 1.0 / WL_BAND[1]
    ax.loglog(1.0 / fr[m], Ph[m], color="#185FA5", lw=1.4, label="사람")
    ax.loglog(1.0 / fr[m], Pr[m], color="#7F77DD", lw=1.4, label="RL v2.4")
    ax.set_xlabel("파장 (m)"); ax.set_ylabel("횡위치 스펙트럼 파워")
    ax.set_title(f"주파장 사람 {tex['wl_h']:.0f}m vs RL {tex['wl_r']:.0f}m")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"fig_profile_spectrum_{exp}.png"), dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    labels = ["SRR 0.5°", "SRR 2°", "횡속도 반전"]
    hvals = [tex["srr05_h"], tex["srr2_h"], tex["rev_h"]]
    rvals = [tex["srr05_r"], tex["srr2_r"], tex["rev_r"]]
    x = np.arange(3); w = 0.36
    ax.bar(x - w / 2, hvals, w, label="사람", color="#185FA5")
    ax.bar(x + w / 2, rvals, w, label="RL v2.4", color="#7F77DD")
    for i, (a, b) in enumerate(zip(hvals, rvals)):
        ax.text(i - w / 2, a, f"{a:.1f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + w / 2, b, f"{b:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("/km")
    ax.set_title("조향/교정 활동 지표 (잔 교정 점검)"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"fig_profile_srr_{exp}.png"), dpi=120); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, nm in [(axes[0], "e", "구간(200m) 횡변동 std"), (axes[1], "lata", "구간 횡가속 std")]:
        ax.scatter(seg_h[key], seg_r[key], s=14, alpha=.5, color="#7F77DD")
        hi = max(max(seg_h[key]), max(seg_r[key])) * 1.05
        ax.plot([0, hi], [0, hi], "r--", lw=1)
        ax.set_xlabel("사람"); ax.set_ylabel("RL v2.4")
        ax.set_title(f"{nm}  r={seg_stats[key]['corr']:.2f}")
    fig.suptitle(f"구간 단위 비교 ({exp})"); fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"fig_profile_segments_{exp}.png"), dpi=120); plt.close(fig)

    # ---- save + report ----
    out = dict(exp=exp, e_tau=tau_s, e_sigma=sig_s, e_lpf=lpf_s, offroad=offs,
               n_test=len(test_roads), frame=res, texture=tex, segment=seg_stats)
    json.dump(out, open(os.path.join(REP, f"profile_eval_{exp}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    L = ["\n\n---\n\n## 7. v2.4 — 조향지표(SRR) 추가 · 평활 의도-방황(smooth intent wander)\n",
         f"드라이버 v2.4 = **OU 의도-방황(τ={tau_s:.0f}m, σ={sig_s:.3f}m)을 저역필터"
         f"(τ₂={lpf_s:.0f}m, SRR_0.5로 보정)로 평활한 뒤 목표 차선위치(관측 e채널)에 주입** "
         f"+ RL 조향 + 속도 컨트롤러. test 이탈 {offs}/{len(test_roads)}.\n",
         "※ 잔 교정 원인 규명 과정(전부 수치로 기각/확정): ①조향 노이즈 주입 → 안정화 정책과 씨름, "
         "SRR 2.5~11배 ②1차 조향지연 → 피드백 지연으로 폐루프 불안정 ③데드밴드 → 16% 개선에 그침 "
         "④의도-방황(생 OU) → SRR 그대로(~32) ⑤진단: **정책 단독 SRR 6.4/km(사람 11.9보다 낮음, "
         "정책 무죄)** → 범인은 OU의 브라운 거칠기(미분불가 경로 추종 = 강제 조향반전). "
         "→ 의도 경로를 평활하면 해소. 보정엔 안정성 가드 적용.\n",
         f"조향휠 환산: wheel_deg = κ·휠베이스({WHEELBASE}m)·조향비({STEER_RATIO:.0f}, 가정)·57.3 — "
         "SRR은 히스테리시스 방식 반전 카운트(/km).\n",
         "### 조향/교정 활동 (잔 교정 점검)\n",
         "| 지표 | 사람 | RL v2.4 |", "|---|---|---|",
         f"| SRR_0.5 (/km) | {tex['srr05_h']:.1f} | **{tex['srr05_r']:.1f}** |",
         f"| SRR_2 (/km) | {tex['srr2_h']:.1f} | {tex['srr2_r']:.1f} |",
         f"| 횡속도 반전율 (/km) | {tex['rev_h']:.1f} | {tex['rev_r']:.1f} |",
         f"| 횡위치 주파장 (m) | {tex['wl_h']:.0f} | {tex['wl_r']:.0f} |",
         "\n### 프레임 단위 (10m)\n",
         "| 신호 | 사람 mean±std | RL mean±std | W1 | corr | RMSE |", "|---|---|---|---|---|---|"]
    for key, name in SIGS:
        d = res[key]
        L.append(f"| {name} | {d['h_mean']:.3g}±{d['h_std']:.3g} | {d['r_mean']:.3g}±{d['r_std']:.3g} "
                 f"| {d['w1']:.3g} | {d['profile_corr']:.2f} | {d['profile_rmse']:.3g} |")
    L += ["\n### 구간 단위 (200m)\n", "| 신호 | 구간 std 상관 | 구간 std W1 |", "|---|---|---|"]
    for key, name in SIGS:
        L.append(f"| {name} | {seg_stats[key]['corr']:.2f} | {seg_stats[key]['w1']:.3g} |")
    L += [f"\n![조향지표](figs/fig_profile_srr_{exp}.png)\n",
          f"![프로파일](figs/fig_profile_signals_{exp}.png)\n",
          f"![스펙트럼](figs/fig_profile_spectrum_{exp}.png)\n",
          f"![구간비교](figs/fig_profile_segments_{exp}.png)\n",
          "**주석**: 조향비 15는 가정값(장비 제원 확인 시 교체). pitch 미모델·속도 컨트롤러 담당은 §6과 동일.\n"]
    open(os.path.join(REP, "report_rl.md"), "a", encoding="utf-8").write("\n".join(L))
    print("SRR0.5 h=%.1f r=%.1f | SRR2 h=%.1f r=%.1f | rev h=%.1f r=%.1f | wl h=%.0f r=%.0f" %
          (tex["srr05_h"], tex["srr05_r"], tex["srr2_h"], tex["srr2_r"],
           tex["rev_h"], tex["rev_r"], tex["wl_h"], tex["wl_r"]), flush=True)
    print("frame W1: " + ", ".join(f"{k}={res[k]['w1']:.3g}" for k, _ in SIGS), flush=True)
    print("wrote profile_eval json + figs + report section 7", flush=True)


if __name__ == "__main__":
    main()
