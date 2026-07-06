# -*- coding: utf-8 -*-
"""
36_population.py  --  모집단 모델: 다실험 성격 은행 + 퍼짐(분산) 보정.

조건축(§23)에서 확정된 결함 치료: 가상 군집의 개인간 SDLP 분산이 사람보다 작아
효과크기(d)가 과대. 원인 = ①특성 풀이 단일실험(2024) ②σ-역산이 군집 '평균'만 정합.

  1) 성격 은행: 2024+남산+왕숙 피험자별 (SDLP, LPM, v선호) → 실험 내 z-정규화 후 통합
     (상관구조 보존 재표집). 배포 시 타깃 스케일(파일럿 19명 모멘트)로 복원.
  2) 퍼짐 보정: σ→실현SDLP 프로브 선형맵 → 운전자별 목표 SDLP를 개별 명중.
  3) 검증: 남산 실현 SDLP 분포(평균·표준편차) vs 사람 + 조건축 d 과대 정상화 재시험.

  python 36_population.py --n 29
"""
import os, json, argparse, importlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import REP, CACHE, gen_split, wasserstein1d
from driving_env import DrivingEnv, load_roads, rollout, trim_roads

p20 = importlib.import_module("20_profile_eval")
p22 = importlib.import_module("22_v3_spectral")
p26 = importlib.import_module("26_newroad_pipeline")
p34 = importlib.import_module("34_virtual_cohort")

for fam in ["Malgun Gothic", "NanumGothic", "Gulim"]:
    try:
        plt.rcParams["font.family"] = fam; break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(REP, "figs"); os.makedirs(FIG, exist_ok=True)
np.random.seed(0)
GAIN, BASE = 0.012, "rl_multi.zip"          # v3.3


def subject_traits(exp):
    roads, _, dd = load_roads(os.path.join(CACHE, f"env_roads_{exp}.npz"))
    roads = trim_roads(roads)
    out = {}
    for r in roads:
        out.setdefault(int(r["subject"]), []).append(
            (float(np.std(r["e_ref"])), float(np.mean(r["e_ref"])), float(np.mean(r["v_ref"]))))
    return {s: np.mean(v, axis=0) for s, v in out.items()}, roads, dd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=29)
    args = ap.parse_args()

    # ---- 1) 다실험 성격 은행 (z-벡터) ----
    Z = []
    for exp in ["2024", "namsan", "wangsuk"]:
        tr_map, _, _ = subject_traits(exp)
        T = np.array(list(tr_map.values()))
        z = (T - T.mean(0)) / (T.std(0) + 1e-9)
        Z.append(z)
        print(f"  {exp}: {len(T)}명 | SDLP {T[:,0].mean():.3f}±{T[:,0].std():.3f}", flush=True)
    Z = np.vstack(Z)
    C = np.corrcoef(Z.T)
    print(f"성격 은행: {len(Z)}명 | corr(SDLP,LPM)={C[0,1]:+.2f} corr(SDLP,v)={C[0,2]:+.2f}",
          flush=True)

    # ---- 남산 배포: 파일럿(train 19명) 모멘트로 스케일 복원 ----
    tr_map, roadsN, ddN = subject_traits("namsan")
    subjN = np.array(sorted(tr_map.keys()))
    roads_by_subj = np.array([r["subject"] for r in roadsN])
    trm, vam, tem = gen_split(np.array([r["subject"] for r in roadsN], "int64"), seed=0)
    subj_tr = sorted(set(np.array([r["subject"] for r in roadsN])[trm].tolist()))
    subj_te = sorted(set(np.array([r["subject"] for r in roadsN])[vam | tem].tolist()))
    Ttr = np.array([tr_map[s] for s in subj_tr])
    mu_t, sd_t = Ttr.mean(0), Ttr.std(0)
    Tte = np.array([tr_map[s] for s in subj_te])
    print(f"파일럿 {len(subj_tr)}명 모멘트: SDLP {mu_t[0]:.3f}±{sd_t[0]:.3f} | "
          f"검증 {len(subj_te)}명: {Tte[:,0].mean():.3f}±{Tte[:,0].std():.3f}", flush=True)

    rng = np.random.RandomState(1)
    zdraw = Z[rng.randint(len(Z), size=args.n)]
    targets = mu_t + zdraw * sd_t                 # (n,3): 목표 [SDLP, LPM, v]
    targets[:, 0] = np.clip(targets[:, 0], 0.08, 0.6)

    # ---- 2) 퍼짐 보정: σ→실현 SDLP 프로브 선형맵 (남산 블라인드, v3.3) ----
    ch = p34.load_champion(base=BASE)
    packs = [p26.load_cvae(c) for c in ["wangsuk", "merge"]]
    test_geo = [r for r in roadsN if r["subject"] in subj_te][:20]
    rngv = np.random.RandomState(3)
    blind = []
    for r in test_geo:
        r2 = dict(r)
        vs = [p26.synth_vref(r, cvae, gs, bs, z_dim, rngv) for (cvae, gs, bs, z_dim) in packs]
        r2["v_ref"] = np.mean(vs, axis=0).astype("float32")
        r2["e_ref"] = np.zeros_like(r2["v_ref"])
        blind.append(r2)
    env = DrivingEnv(blind, dd=ddN, record=True, steer_gain=GAIN)

    def probe(sig, n=6, seed=90):
        out = []
        for j in range(n):
            pol = p22.SpectralPolicy(ch["model"], ch["fr"], ch["A"], sig,
                                     lib=ch["lib"], seed=seed + j)
            pol.reset()
            traj, _ = rollout(env, pol, j % len(blind))
            if len(traj) > 60:
                out.append(float(np.std(p20.rl_signals(traj, gain=GAIN)["e"])))
        return float(np.mean(out))

    sig_grid = np.array([0.05, 0.15, 0.30, 0.50])
    sd_grid = np.array([probe(s, seed=90 + i * 10) for i, s in enumerate(sig_grid)])
    b, a = np.polyfit(sig_grid, sd_grid, 1)       # realized = a + b·σ
    print("프로브 맵: σ", sig_grid.tolist(), "-> SDLP", np.round(sd_grid, 3).tolist(),
          f"| 선형 a={a:.3f} b={b:.3f}", flush=True)

    v_base = float(np.mean([np.mean(r["v_ref"]) for r in blind]))

    def run_cohort(tgts, seed0=500):
        vals = dict(sdlp=[], lpm=[], v=[])
        for j, (t_sdlp, t_lpm, t_v) in enumerate(tgts):
            sig_j = float(np.clip((t_sdlp - a) / max(b, 1e-6), 0.02, 1.2))
            pol = p22.SpectralPolicy(ch["model"], ch["fr"], ch["A"], sig_j,
                                     b_bias=t_lpm, v_scale=t_v / v_base,
                                     lib=ch["lib"], seed=seed0 + j)
            pol.reset()
            traj, _ = rollout(env, pol, j % len(blind))
            if len(traj) > 60:
                s = p20.rl_signals(traj, gain=GAIN)
                vals["sdlp"].append(float(np.std(s["e"])))
                vals["lpm"].append(float(np.mean(s["e"])))
                vals["v"].append(float(np.mean(traj[:, 2])))
        return vals

    V = run_cohort(targets)
    sd_v = np.array(V["sdlp"])
    res = dict(n=args.n, bank=len(Z),
               human_te=dict(mean=float(Tte[:, 0].mean()), std=float(Tte[:, 0].std())),
               virtual=dict(mean=float(sd_v.mean()), std=float(sd_v.std())),
               w1=float(wasserstein1d(Tte[:, 0], sd_v)),
               probe=dict(a=float(a), b=float(b)))
    print(f"[퍼짐 검증] SDLP 사람(te) {Tte[:,0].mean():.3f}±{Tte[:,0].std():.3f} "
          f"vs 가상 {sd_v.mean():.3f}±{sd_v.std():.3f} | W1={res['w1']:.3f}", flush=True)

    # ---- 3) 조건축 d 정상화 재시험 (cond0→3, §23 수정자 재사용) ----
    mods = json.load(open(os.path.join(REP, "condition_axis.json"), encoding="utf-8"))["mods"]
    m3 = mods["3"] if "3" in mods else mods[3]
    tg3 = targets.copy()
    tg3[:, 0] *= m3["r_sdlp"]; tg3[:, 1] += m3["d_lpm"]; tg3[:, 2] *= m3["r_v"]
    V3 = run_cohort(tg3, seed0=900)
    b3 = np.array(V3["sdlp"])
    sp = np.sqrt((sd_v.var(ddof=1) + b3.var(ddof=1)) / 2)
    d_v = float((b3.mean() - sd_v.mean()) / (sp + 1e-12))
    res["cond3_d"] = dict(virtual=d_v, human=-0.54, before=-1.28)
    print(f"[조건 d 재시험] cond0->3: 가상 d={d_v:+.2f} (사람 -0.54, 퍼짐보정 전 -1.28)",
          flush=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0.05, 0.55, 18)
    ax.hist(Tte[:, 0], bins=bins, alpha=0.6, label=f"사람 te (n={len(Tte)})", color="#185FA5")
    ax.hist(sd_v, bins=bins, alpha=0.6, label=f"가상 (n={len(sd_v)})", color="#7F77DD")
    ax.set_xlabel("SDLP (m)"); ax.set_ylabel("명"); ax.legend()
    ax.set_title(f"모집단 모델: SDLP 분포 (W1={res['w1']:.3f})")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_population.png"), dpi=120)
    plt.close(fig)
    json.dump(res, open(os.path.join(REP, "population.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved fig_population.png + population.json", flush=True)


if __name__ == "__main__":
    main()
