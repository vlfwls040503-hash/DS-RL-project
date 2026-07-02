# -*- coding: utf-8 -*-
"""
20_profile_eval.py  --  frame/segment-level MULTI-SIGNAL profile evaluation (지시: SDLP 단독지표 금지).

Human-like driver v2 (지난 결함 2건의 수정):
  - lateral : RL deterministic steering + LOW-FREQUENCY OU noise (colored, corr length tau)
              -> 사람의 '흘렀다 교정' 질감. 백색노이즈의 기계적 물결 제거.
  - longitudinal: PD speed controller on v_ref (속도는 학습 축에서 제외 — 폭주 원천 차단).

Evaluation (사용자 지시 반영): 프레임(10m 그리드)·200m 구간 단위로 신호 전체 비교
  signals: 횡위치 e, 상대 yaw ψ, 횡속도 v·ψ, 경로곡률 κ_path, 횡가속 v²κ
  metrics: 분포 W1 · 동일지점 프로파일 corr/RMSE · 질감(주파장, 횡속도 반전율/km)
  (pitch: 간이 시뮬은 차체 pitch 미모델 — 도로 종경사가 양측 동일이라 비교 무의미, 정직히 제외.
   속도: 컨트롤러 담당이므로 평가 제외 — 사용자 지시.)

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

GRID = 10.0          # resample grid (m) for frame-level comparison
SEG = 200.0          # segment length (m)
WL_BAND = (40.0, 3000.0)


class HumanlikePolicy:
    """RL steering(mean) + OU lateral noise + PD speed. Call reset() per episode."""
    def __init__(self, model, tau=300.0, sigma=0.15, k_v=1.0, seed=0):
        self.m, self.tau, self.sigma, self.k_v = model, float(tau), float(sigma), k_v
        self.rng = np.random.RandomState(seed)
        self.x = 0.0

    def reset(self):
        self.x = self.rng.randn() * self.sigma

    def __call__(self, obs, env):
        steer = float(self.m.predict(obs, deterministic=True)[0][0])
        dm = max(env.v, 0.1) * RL_DT                     # meters this step
        self.x += -self.x * dm / self.tau + self.sigma * np.sqrt(2 * dm / self.tau) * self.rng.randn()
        i = min(int(env.s / env.dd), len(env.road["v_ref"]) - 1)
        acc = float(np.clip(self.k_v * (env.road["v_ref"][i] - env.v) / RL_A_MAX, -1, 1))
        return np.array([np.clip(steer + self.x, -1, 1), acc], np.float32)


# ---------------- signal extraction ----------------
def human_signals(road, dd_road):
    e = _smooth(np.asarray(road["e_ref"], np.float64))
    v = _smooth(np.asarray(road["v_ref"], np.float64))
    s = np.arange(len(e)) * dd_road
    psi = np.gradient(e, dd_road)
    kap = np.asarray(road["curv"], np.float64) + np.gradient(psi, dd_road)
    g = np.arange(0, s[-1], GRID)
    def rs(x): return np.interp(g, s, x)
    e, v, psi, kap = rs(e), rs(v), rs(psi), rs(kap)
    return dict(s=g, e=e, psi=psi, latv=v * psi, kappa=kap, lata=v * v * kap)


def rl_signals(traj):
    s0 = traj[:, 0]
    g = np.arange(s0[0], s0[-1], GRID)
    def rs(col): return np.interp(g, s0, traj[:, col])
    e, v, psi, steer = rs(1), rs(2), rs(4), rs(5)
    kap = steer * RL_STEER_GAIN
    return dict(s=g, e=e, psi=psi, latv=v * psi, kappa=kap, lata=v * v * kap)


def wavelength(e, grid=GRID):
    x = np.asarray(e, np.float64); x = x - x.mean()
    if len(x) < 32:
        return float("nan")
    w = np.hanning(len(x)); F = np.fft.rfft(x * w)
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


def mean_psd(sig_list, grid=GRID, n=512):
    """Average |FFT|^2 over roads on a FIXED n-point window (shared frequency axis)."""
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
    ap.add_argument("--per_road", type=int, default=1)
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

    # ---- calibrate (tau, sigma) on val: match lateral std AND dominant wavelength ----
    hv = [human_signals(r, dd) for r in val_roads]
    t_std = float(np.mean([h["e"].std() for h in hv]))
    t_wl = float(np.nanmean([wavelength(h["e"]) for h in hv]))
    print(f"val human targets: e-std={t_std:.3f} wavelength={t_wl:.0f} m", flush=True)

    env_v = DrivingEnv(val_roads, dd=dd, record=True)

    def probe(tau, sigma):
        pol = HumanlikePolicy(model, tau, sigma, seed=0)
        stds, wls = [], []
        for k in range(len(val_roads)):
            pol.reset()
            traj, _ = rollout(env_v, pol, k)
            if len(traj) > 50:
                sg = rl_signals(traj)
                stds.append(sg["e"].std()); wls.append(wavelength(sg["e"]))
        return float(np.mean(stds)), float(np.nanmean(wls))

    best = None
    for tau in [150.0, 300.0, 600.0]:
        sd1, _ = probe(tau, 0.15)
        sigma = float(np.clip(0.15 * t_std / max(sd1, 1e-6), 0.03, 0.6))   # linear scale to std target
        sd, wl = probe(tau, sigma)
        score = abs(sd - t_std) / t_std + abs(wl - t_wl) / t_wl
        print(f"  tau={tau:4.0f} sigma={sigma:.3f} -> e-std={sd:.3f} wl={wl:.0f}  score={score:.3f}", flush=True)
        if best is None or score < best[0]:
            best = (score, tau, sigma)
    _, tau_s, sig_s = best
    print(f"calibrated: tau*={tau_s:.0f} m  sigma*={sig_s:.3f}", flush=True)

    # ---- test rollouts ----
    env = DrivingEnv(test_roads, dd=dd, record=True)
    pol = HumanlikePolicy(model, tau_s, sig_s, seed=1)
    H, R, offs = [], [], 0
    for k in range(len(test_roads)):
        H.append(human_signals(test_roads[k], dd))
        pol.reset()
        traj, off = rollout(env, pol, k)
        offs += int(off)
        R.append(rl_signals(traj))
    print(f"test rollouts: offroad {offs}/{len(test_roads)}", flush=True)

    # ---- frame-level metrics per signal ----
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
    # texture
    tex = dict(wl_h=float(np.nanmean([wavelength(h["e"]) for h in H])),
               wl_r=float(np.nanmean([wavelength(r["e"]) for r in R])),
               rev_h=float(np.mean([reversal_rate(h["latv"]) for h in H])),
               rev_r=float(np.mean([reversal_rate(r["latv"]) for r in R])))

    # ---- segment-level (200 m) ----
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

    # ================= figures =================
    h0, r0 = H[0], R[0]
    n0 = min(len(h0["s"]), len(r0["s"]), 400)
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for ax, (key, name) in zip(axes, SIGS):
        ax.plot(h0["s"][:n0], h0[key][:n0], color="#185FA5", lw=1.2, label="사람")
        ax.plot(r0["s"][:n0], r0[key][:n0], color="#7F77DD", lw=1.1, alpha=.9, label="RL v2")
        ax.set_ylabel(name); ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("거리(m)")
    fig.suptitle(f"프레임 프로파일: 사람 vs RL v2 (OU τ={tau_s:.0f}m, 속도=컨트롤러) — test road 1")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"fig_profile_signals_{exp}.png"), dpi=120); plt.close(fig)

    fr, Ph = mean_psd([h["e"] for h in H]); _, Pr = mean_psd([r["e"] for r in R])
    fig, ax = plt.subplots(figsize=(7, 4))
    m = fr > 1.0 / WL_BAND[1]
    ax.loglog(1.0 / fr[m], Ph[m], color="#185FA5", lw=1.4, label="사람")
    ax.loglog(1.0 / fr[m], Pr[m], color="#7F77DD", lw=1.4, label="RL v2")
    ax.axvline(tex["wl_h"], color="#185FA5", ls=":", lw=1)
    ax.axvline(tex["wl_r"], color="#7F77DD", ls=":", lw=1)
    ax.set_xlabel("파장 (m)"); ax.set_ylabel("횡위치 스펙트럼 파워")
    ax.set_title(f"흔들림 질감(스펙트럼): 주파장 사람 {tex['wl_h']:.0f}m vs RL {tex['wl_r']:.0f}m")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"fig_profile_spectrum_{exp}.png"), dpi=120); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, nm in [(axes[0], "e", "구간(200m) 횡변동 std"), (axes[1], "lata", "구간 횡가속 std")]:
        ax.scatter(seg_h[key], seg_r[key], s=14, alpha=.5, color="#7F77DD")
        lo = 0; hi = max(max(seg_h[key]), max(seg_r[key])) * 1.05
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel("사람"); ax.set_ylabel("RL v2")
        ax.set_title(f"{nm}  r={seg_stats[key]['corr']:.2f}")
    fig.suptitle(f"구간 단위 비교 ({exp})"); fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"fig_profile_segments_{exp}.png"), dpi=120); plt.close(fig)

    # ================= save + report =================
    out = dict(exp=exp, tau=tau_s, sigma=sig_s, offroad=offs, n_test=len(test_roads),
               frame=res, texture=tex, segment=seg_stats)
    json.dump(out, open(os.path.join(REP, f"profile_eval_{exp}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    L = ["\n\n---\n\n## 6. 프레임/구간 다지표 프로파일 평가 (v2 드라이버)\n",
         f"드라이버 v2 = RL 조향(평균) + **저주파 OU 노이즈**(τ={tau_s:.0f}m, σ={sig_s:.3f}, val 보정) "
         "+ **속도는 v_ref 추종 컨트롤러**(속도 폭주 원천 제거; 속도는 평가 제외 — 지시 반영). "
         f"test 이탈 {offs}/{len(test_roads)}.\n",
         "### 프레임 단위 (10m 그리드)\n",
         "| 신호 | 사람 mean±std | RL mean±std | 분포 W1 | 프로파일 corr | RMSE |",
         "|---|---|---|---|---|---|"]
    for key, name in SIGS:
        d = res[key]
        L.append(f"| {name} | {d['h_mean']:.3g}±{d['h_std']:.3g} | {d['r_mean']:.3g}±{d['r_std']:.3g} "
                 f"| {d['w1']:.3g} | {d['profile_corr']:.2f} | {d['profile_rmse']:.3g} |")
    L += [f"\n### 질감(texture)\n",
          f"- 횡위치 주파장: 사람 **{tex['wl_h']:.0f} m** vs RL **{tex['wl_r']:.0f} m** "
          "(백색노이즈 시절의 기계적 단주기 물결 → 저주파 드리프트로 교정)",
          f"- 횡속도 반전율: 사람 {tex['rev_h']:.1f}/km vs RL {tex['rev_r']:.1f}/km",
          "\n### 구간 단위 (200 m)\n",
          "| 신호 | 구간 std 상관 | 구간 std W1 |", "|---|---|---|"]
    for key, name in SIGS:
        L.append(f"| {name} | {seg_stats[key]['corr']:.2f} | {seg_stats[key]['w1']:.3g} |")
    L += [f"\n![프로파일](figs/fig_profile_signals_{exp}.png)\n",
          f"![스펙트럼](figs/fig_profile_spectrum_{exp}.png)\n",
          f"![구간비교](figs/fig_profile_segments_{exp}.png)\n",
          "**정직 주석**: (1) 프로파일 corr는 개인 고유 흔들림(e_ref 은닉+노이즈)이 있는 한 1이 될 수 없음 — "
          "기하 반응의 공유 성분만 반영. (2) pitch는 간이 시뮬이 차체 피치를 모델하지 않아 제외"
          "(도로 종경사는 양측 동일). (3) 속도는 컨트롤러 담당으로 학습·평가 축에서 제외.\n"]
    open(os.path.join(REP, "report_rl.md"), "a", encoding="utf-8").write("\n".join(L))
    print("frame W1: " + ", ".join(f"{k}={res[k]['w1']:.3g}" for k, _ in SIGS), flush=True)
    print(f"texture: wl {tex['wl_h']:.0f} vs {tex['wl_r']:.0f} m | rev {tex['rev_h']:.1f} vs {tex['rev_r']:.1f} /km", flush=True)
    print("wrote profile_eval json + figs + report section 6", flush=True)


if __name__ == "__main__":
    main()
