# -*- coding: utf-8 -*-
"""
22_v3_spectral.py  --  v3: 스펙트럼 의도합성 (OU 계열의 파레토 한계 돌파 시도).

원리: v2.4의 잔차(주파장 1386 vs 847, SRR 16.7 vs 10.1)는 OU+LPF의 스펙트럼 '모양'이
사람과 달라서 생김. → train/val 사람 횡위치의 **평균 진폭 스펙트럼을 직접 측정**하고,
위상만 무작위화해 의도-방황 신호를 합성(스펙트럼 합성법). 정책이 이 목표를 추종하면
횡위치 스펙트럼이 구성상 사람과 일치.

성공 기준(사전선언): C2ST AUC가 v2.4의 0.82에서 0.5 방향으로 유의하게 하락.

  python 22_v3_spectral.py --exp 2024 --per_road 10
"""
import os, json, argparse, importlib
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from common import ART, REP, CACHE, gen_split, wasserstein1d, RL_A_MAX, RL_DT
from driving_env import DrivingEnv, load_roads, rollout, trim_roads

p20 = importlib.import_module("20_profile_eval")
p21 = importlib.import_module("21_validation")

for fam in ["Malgun Gothic", "NanumGothic", "Gulim"]:
    try:
        plt.rcParams["font.family"] = fam; break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(REP, "figs"); os.makedirs(FIG, exist_ok=True)
np.random.seed(0); torch.manual_seed(0)
GRID = p20.GRID
NFFT = 512
IDX_E = p20.IDX_E


# ---------------- spectral synthesis ----------------
def target_spectrum(chunks):
    """평균 진폭 스펙트럼 |E(f)| (train/val 사람 5.2km 청크, 512pt)."""
    amps = []
    for e in chunks:
        e = np.asarray(e, np.float64)
        if len(e) < NFFT:
            continue
        x = e[:NFFT] - e[:NFFT].mean()
        amps.append(np.abs(np.fft.rfft(x * np.hanning(NFFT))))
    A = np.mean(amps, axis=0)
    A[0] = 0.0
    return np.fft.rfftfreq(NFFT, d=GRID), A


def synth_wander(fr, A, n_out, rng):
    """사람 스펙트럼 모양 + 무작위 위상 → 단위표준편차 방황 신호 (10m 그리드)."""
    fr_out = np.fft.rfftfreq(n_out, d=GRID)
    A_out = np.interp(fr_out, fr, A)
    spec = A_out * np.exp(1j * rng.uniform(0, 2 * np.pi, len(A_out)))
    spec[0] = 0.0
    x = np.fft.irfft(spec, n_out)
    return x / (x.std() + 1e-12)


class SpectralPolicy:
    """RL 조향 + 의도-방황(목표 e채널 주입) + PD 속도. reset() 필수.
    lib=None  -> 스펙트럼 합성(사람 진폭스펙트럼 + 무작위 위상; 가우시안 → 위상구조 상실)
    lib=list  -> **사람 궤적 라이브러리 부트스트랩**: 실제 사람 잔차 청크를 이어붙여 목표로
                 사용 → 위상·반전·고차 구조까지 사람 것 그대로 (v3.1)."""
    def __init__(self, model, fr, A, sigma, b_bias=0.0, v_scale=1.0, k_v=1.0,
                 n_grid=1024, lib=None, seed=0):
        self.m, self.fr, self.A, self.lib = model, fr, A, lib
        self.sigma, self.b_bias, self.v_scale, self.k_v = float(sigma), float(b_bias), float(v_scale), k_v
        self.n_grid = n_grid
        self.rng = np.random.RandomState(seed)
        self.w = None

    def reset(self):
        if self.lib is not None:                       # v3.1: 사람 청크 이어붙이기(크로스페이드)
            need = self.n_grid + 40
            w = None
            while w is None or len(w) < need:
                c = self.lib[self.rng.randint(len(self.lib))].astype(np.float64)
                if self.rng.rand() < 0.5:
                    c = -c[::-1]                       # 좌우/진행 반전으로 다양성 확대
                if w is None:
                    w = c.copy()
                else:
                    ov = 20
                    blend = np.linspace(1.0, 0.0, ov)
                    w[-ov:] = w[-ov:] * blend + c[:ov] * (1.0 - blend)
                    w = np.concatenate([w, c[ov:]])
            self.w = w
        else:                                          # v3: 스펙트럼 합성 (~10.2 km, unit std)
            self.w = synth_wander(self.fr, self.A, self.n_grid, self.rng)

    def __call__(self, obs, env):
        gi = env.s / GRID
        i0 = min(int(gi), len(self.w) - 2); f = gi - int(gi)
        b = self.sigma * ((1 - f) * self.w[i0] + f * self.w[i0 + 1])
        o = obs.copy()
        o[IDX_E] = o[IDX_E] - float(np.clip(b + self.b_bias, -1.0, 1.0))
        steer = float(self.m.predict(o, deterministic=True)[0][0])
        i = min(int(env.s / env.dd), len(env.road["v_ref"]) - 1)
        acc = float(np.clip(self.k_v * (self.v_scale * env.road["v_ref"][i] - env.v) / RL_A_MAX, -1, 1))
        return np.array([np.clip(steer, -1, 1), acc], np.float32)


def cv_auc(Xa, ya, seed=0):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    ps = np.zeros(len(ya))
    for tr_i, te_i in skf.split(Xa, ya):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xa[tr_i], ya[tr_i])
        ps[te_i] = clf.predict_proba(Xa[te_i])[:, 1]
    return float(roc_auc_score(ya, ps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="2024")
    ap.add_argument("--per_road", type=int, default=10)
    ap.add_argument("--mode", choices=["spectral", "library"], default="spectral")
    ap.add_argument("--model", default="", help="정책 zip (기본 rl_{exp}.zip)")
    ap.add_argument("--gain", type=float, default=None, help="steer_gain (기본 RL_STEER_GAIN)")
    ap.add_argument("--tag", default="", help="출력 접미사; 지정시 report 추가 안함")
    args = ap.parse_args()
    exp = args.exp
    mode = args.mode
    from common import RL_STEER_GAIN
    gain = args.gain if args.gain is not None else RL_STEER_GAIN
    tag = args.tag

    roads, subject, dd = load_roads(os.path.join(CACHE, f"env_roads_{exp}.npz"))
    roads = trim_roads(roads)
    subject = np.array([r["subject"] for r in roads], "int64")
    tr, va, te = gen_split(subject, seed=0)
    fit_roads = [r for r, m in zip(roads, tr | va) if m]
    val_roads = [r for r, m in zip(roads, va) if m]
    test_roads = [r for r, m in zip(roads, te) if m]
    model = PPO.load(os.path.join(ART, args.model or f"rl_{exp}.zip"), device="cpu")
    print(f"model={args.model or f'rl_{exp}.zip'} gain={gain}", flush=True)

    # ---- 사람 스펙트럼 (train+val에서만 — test 오염 방지) ----
    fit_chunks = []
    for r in fit_roads:
        hs = p20.human_signals(r, dd)
        for ch in p21.chunk_signals(hs):
            fit_chunks.append(ch["e"])
    fr, A = target_spectrum(fit_chunks)
    lib = None
    if mode == "library":
        lib = []
        for e in fit_chunks:
            x = np.asarray(e, np.float64); x = x - x.mean()
            if len(x) >= 400 and x.std() > 1e-3:
                lib.append((x / x.std()).astype(np.float64))
        print(f"library mode: {len(lib)} human chunks (반전 포함 실효 {2*len(lib)})", flush=True)
    print(f"target spectrum from {len(fit_chunks)} fit-chunks | mode={mode}", flush=True)

    # ---- σ 보정 (val, e-std 1노브) + 검증지표 ----
    hv = [p20.human_signals(r, dd) for r in val_roads]
    t_std = float(np.mean([h["e"].std() for h in hv]))
    t_wl = float(np.nanmean([p20.wavelength(h["e"]) for h in hv]))
    t_srr = float(np.mean([p20.srr(h["theta"], 0.5) for h in hv]))
    env_v = DrivingEnv(val_roads, dd=dd, record=True, steer_gain=gain)

    def probe(sigma):
        pol = SpectralPolicy(model, fr, A, sigma, lib=lib, seed=0)
        stds, wls, srrs, off = [], [], [], 0
        for k in range(len(val_roads)):
            pol.reset()
            traj, o = rollout(env_v, pol, k)
            off += int(o)
            if len(traj) > 60:
                sg = p20.rl_signals(traj, gain=gain)
                stds.append(sg["e"].std()); wls.append(p20.wavelength(sg["e"]))
                srrs.append(p20.srr(sg["theta"], 0.5))
        return float(np.mean(stds)), float(np.nanmean(wls)), float(np.mean(srrs)), off

    sd1, wl1, s51, off1 = probe(t_std)
    sigma = float(np.clip(t_std * t_std / max(sd1, 1e-6), 0.05, 1.2))
    sd, wl, s5, off = probe(sigma)
    print(f"val: sigma={sigma:.3f} -> e-std={sd:.3f}(목표 {t_std:.3f}) wl={wl:.0f}(목표 {t_wl:.0f}) "
          f"SRR0.5={s5:.1f}(목표 {t_srr:.1f}) off={off}", flush=True)

    # ---- test: 군집 롤아웃 (특성 부트스트랩, 21과 동일 설계) ----
    pool = [dict(sdlp=float(np.std(r["e_ref"])), lpm=float(np.mean(r["e_ref"])),
                 v=float(np.mean(r["v_ref"])), cond=int(r.get("cond", 0))) for r in fit_roads]
    sdlp_pool = float(np.mean([t["sdlp"] for t in pool]))
    v_pool = float(np.mean([t["v"] for t in pool]))
    by_cond = {}
    for t in pool:
        by_cond.setdefault(t["cond"], []).append(t)
    rng = np.random.RandomState(0)

    env = DrivingEnv(test_roads, dd=dd, record=True, steer_gain=gain)
    H_units, h_cond, S_sig, s_meta = [], [], [], []
    for k, road in enumerate(test_roads):
        hs = p20.human_signals(road, dd)
        for ch in p21.chunk_signals(hs):
            H_units.append(ch); h_cond.append(int(road.get("cond", 0)))
        cand = by_cond.get(int(road.get("cond", 0)), pool)
        for j in range(args.per_road):
            t = cand[rng.randint(len(cand))]
            pol = SpectralPolicy(model, fr, A,
                                 sigma=float(np.clip(sigma * t["sdlp"] / sdlp_pool, 0.03, 1.2)),
                                 b_bias=t["lpm"], v_scale=t["v"] / v_pool, lib=lib,
                                 seed=3000 + k * 20 + j)
            pol.reset()
            traj, o = rollout(env, pol, k)
            if len(traj) > 60:
                S_sig.append(p20.rl_signals(traj, gain=gain))
                s_meta.append(dict(cond=int(road.get("cond", 0)), off=bool(o)))
        print(f"  road {k+1}/{len(test_roads)}", flush=True)
    off_rate = float(np.mean([m["off"] for m in s_meta]))

    # texture on test
    tex = dict(wl_h=float(np.nanmean([p20.wavelength(h["e"]) for h in H_units])),
               wl_r=float(np.nanmean([p20.wavelength(s["e"]) for s in S_sig])),
               srr_h=float(np.mean([p20.srr(h["theta"], 0.5) for h in H_units])),
               srr_r=float(np.mean([p20.srr(s["theta"], 0.5) for s in S_sig])),
               srr2_h=float(np.mean([p20.srr(h["theta"], 2.0) for h in H_units])),
               srr2_r=float(np.mean([p20.srr(s["theta"], 2.0) for s in S_sig])),
               sdlp_h=float(np.mean([np.std(h["e"]) for h in H_units])),
               sdlp_r=float(np.mean([np.std(s["e"]) for s in S_sig])))

    # ---- C2ST (21과 동일 특징·프로토콜) ----
    XH = np.vstack([p21.seg_features(h) for h in H_units])
    XS = np.vstack([p21.seg_features(s) for s in S_sig])
    rng2 = np.random.RandomState(2)
    nmin = min(len(XH), len(XS))
    X = np.vstack([XH[rng2.choice(len(XH), nmin, replace=False)],
                   XS[rng2.choice(len(XS), nmin, replace=False)]])
    y = np.concatenate([np.zeros(nmin), np.ones(nmin)])
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    auc = cv_auc(X, y)
    null_aucs = []
    for i in range(50):
        yp = y.copy(); np.random.RandomState(100 + i).shuffle(yp)
        null_aucs.append(cv_auc(X, yp, seed=i))
    p_auc = float((np.sum(np.array(null_aucs) >= auc) + 1) / (len(null_aucs) + 1))
    print(f"C2ST v3: AUC={auc:.3f} (v2.4은 0.819; null {np.mean(null_aucs):.3f}, p={p_auc:.3f}) "
          f"n={nmin}+{nmin}", flush=True)
    print(f"texture: SDLP {tex['sdlp_h']:.3f}/{tex['sdlp_r']:.3f} | wl {tex['wl_h']:.0f}/{tex['wl_r']:.0f} "
          f"| SRR0.5 {tex['srr_h']:.1f}/{tex['srr_r']:.1f} | SRR2 {tex['srr2_h']:.1f}/{tex['srr2_r']:.1f} "
          f"| off={off_rate:.2f}", flush=True)

    # ---- figures: spectrum overlay + AUC 비교 ----
    fr_m, Ph = p20.mean_psd([h["e"] for h in H_units])
    _, Pr = p20.mean_psd([s["e"] for s in S_sig])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    m = fr_m > 1.0 / 3000.0
    ax.loglog(1.0 / fr_m[m], Ph[m], color="#185FA5", lw=1.4, label="사람")
    ax.loglog(1.0 / fr_m[m], Pr[m], color="#7F77DD", lw=1.4, label="v3 스펙트럼합성")
    ax.set_xlabel("파장(m)"); ax.set_ylabel("파워"); ax.legend()
    ax.set_title(f"횡위치 스펙트럼 (주파장 {tex['wl_h']:.0f} vs {tex['wl_r']:.0f} m)")
    ax = axes[1]
    ax.bar(["v2.4", "v3"], [0.819, auc], color=["#888780", "#7F77DD"])
    ax.axhline(0.5, ls=":", color="#185FA5", label="구별불가(0.5)")
    for i, v in enumerate([0.819, auc]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    ax.set_ylim(0.4, 1.0); ax.set_ylabel("C2ST AUC"); ax.legend()
    ax.set_title("판별자 AUC: v2.4 → v3")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"fig_v3_{mode}_{exp}{tag}.png"), dpi=120); plt.close(fig)

    json.dump(dict(exp=exp, mode=mode, sigma=sigma, off_rate=off_rate,
                   model=args.model, gain=gain,
                   c2st=dict(auc=auc, p=p_auc, n_seg=int(nmin), auc_v24=0.819),
                   texture=tex),
              open(os.path.join(REP, f"v3_{mode}_{exp}{tag}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    if tag:                        # 변형 실험(예: wide gain 게이트)은 report 미첨부
        print("tagged run - report append skipped", flush=True)
        return
    if mode == "spectral":
        head = ["\n\n---\n\n## 10. v3 — 스펙트럼 의도합성 (파레토 돌파 시도)\n",
                "OU 계열의 한계는 스펙트럼 '모양' → **train/val 사람 횡위치의 평균 진폭 스펙트럼을 "
                "그대로, 위상만 무작위화해 의도-방황을 합성**(튜닝 노브는 σ 하나). 정책이 추종하면 "
                "횡위치 스펙트럼이 구성상 사람과 일치.\n"]
    else:
        head = ["\n\n---\n\n## 10.1 v3.1 — 사람 궤적 라이브러리 부트스트랩\n",
                "v3(스펙트럼 합성)의 실패가 증명한 것: **사람다움의 지문은 파워스펙트럼(2차 통계)이 "
                "아니라 위상 구조(고차 통계)** — 무작위 위상=가우시안 과정은 같은 스펙트럼에서 가장 "
                "무질서한 경로라 반전율이 폭발(SRR_2 14.5 vs 사람 2.1). → **실제 사람 잔차 청크"
                "(train/val, 반전 포함 ~650개)를 크로스페이드로 이어붙여 의도-목표로 재생**: "
                "위상·반전·고차 구조 전부 사람 것 그대로.\n"]
    L = head + [
         f"| 지표 | 사람(5.2km 단위) | v2.4 | **v3** |", "|---|---|---|---|",
         f"| C2ST AUC | 0.5=이상 | 0.819 | **{auc:.3f}** (p={p_auc:.3f}) |",
         f"| 주파장(m) | {tex['wl_h']:.0f} | 1386 | **{tex['wl_r']:.0f}** |",
         f"| SRR_0.5(/km) | {tex['srr_h']:.1f} | 16.7 | **{tex['srr_r']:.1f}** |",
         f"| SRR_2(/km) | {tex['srr2_h']:.1f} | 1.5 | {tex['srr2_r']:.1f} |",
         f"| SDLP(m) | {tex['sdlp_h']:.3f} | - | {tex['sdlp_r']:.3f} |",
         f"| 이탈율 | - | 0.00 | {off_rate:.2f} |",
         f"\n![v3](figs/fig_v3_{mode}_{exp}.png)\n",
         "- 스펙트럼은 train/val에서만 추정(test 오염 없음). 사람 e에는 기하반응 성분이 일부 포함되나 "
         "결정론 정책의 자체 e-변동(std 0.089)이 작아 이중계상은 소폭 — 명시적 한계.\n"]
    open(os.path.join(REP, "report_rl.md"), "a", encoding="utf-8").write("\n".join(L))
    print("wrote section 10 + fig + v3 json", flush=True)


if __name__ == "__main__":
    main()
