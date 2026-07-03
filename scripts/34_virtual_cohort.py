# -*- coding: utf-8 -*-
"""
34_virtual_cohort.py  --  E: 통합 진입점 "도로 기하 → 가상 피실험자 군집 → 평가 리포트".

입력 (둘 중 하나):
  --geometry road.csv     : 열 [dd_m 간격 그리드의] curv, slope, lane_w, cw  (+선택 v_ref)
  --cache env_roads_X.npz --idx 3 : 기존 캐시의 도로 재사용

속도: v_ref 열이 없으면 CVAE 앙상블(왕숙+merge)이 기하에서 생성 (§20 검증 구성).
드라이버: 챔피언 v3.2 (광권한 RL 조향 + 사람 청크 의도 + PD 속도), 특성 부트스트랩.

  python 34_virtual_cohort.py --cache env_roads_namsan.npz --idx 0 --n 30 --tag demo
출력: reports/cohort_{tag}/  (driver_XX.csv, summary.json, report.md, fig_cohort.png)
"""
import os, json, argparse, importlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from common import ART, REP, CACHE, gen_split, GEN_DD
from driving_env import DrivingEnv, load_roads, rollout, trim_roads

p20 = importlib.import_module("20_profile_eval")
p21 = importlib.import_module("21_validation")
p22 = importlib.import_module("22_v3_spectral")
p26 = importlib.import_module("26_newroad_pipeline")

for fam in ["Malgun Gothic", "NanumGothic", "Gulim"]:
    try:
        plt.rcParams["font.family"] = fam; break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False
GAIN = 0.012


def load_champion():
    """챔피언 v3.2 자산 일체 (2024 학습분)."""
    roads24, _, dd24 = load_roads(os.path.join(CACHE, "env_roads_2024.npz"))
    roads24 = trim_roads(roads24)
    subj = np.array([r["subject"] for r in roads24], "int64")
    tr, va, _ = gen_split(subj, seed=0)
    fit = [r for r, m in zip(roads24, tr | va) if m]
    chunks = []
    for r in fit:
        for ch in p21.chunk_signals(p20.human_signals(r, dd24)):
            chunks.append(ch["e"])
    fr, A = p22.target_spectrum(chunks)
    lib = []
    for e in chunks:
        x = np.asarray(e, np.float64); x = x - x.mean()
        if len(x) >= 400 and x.std() > 1e-3:
            lib.append((x / x.std()).astype(np.float64))
    model = PPO.load(os.path.join(ART, "rl_2024_wide.zip"), device="cpu")
    cal = json.load(open(os.path.join(REP, "v3_library_2024_wide.json"), encoding="utf-8"))
    pool = [dict(sdlp=float(np.std(r["e_ref"])), lpm=float(np.mean(r["e_ref"])),
                 v=float(np.mean(r["v_ref"]))) for r in fit]
    return dict(model=model, fr=fr, A=A, lib=lib, sigma=float(cal["sigma"]), pool=pool,
                sdlp_pool=float(np.mean([t["sdlp"] for t in pool])),
                v_pool=float(np.mean([t["v"] for t in pool])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="", help="기하 CSV (curv,slope,lane_w,cw[,v_ref])")
    ap.add_argument("--dd", type=float, default=GEN_DD)
    ap.add_argument("--cache", default="", help="기존 도로캐시 npz 이름")
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--tag", default="demo")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_dir = os.path.join(REP, f"cohort_{args.tag}")
    os.makedirs(out_dir, exist_ok=True)

    # ---- 도로 준비 ----
    if args.geometry:
        import pandas as pd
        g = pd.read_csv(args.geometry)
        road = dict(curv=g["curv"].to_numpy("float32"),
                    slope=g.get("slope", 0 * g["curv"]).to_numpy("float32"),
                    lane_w=g.get("lane_w", 0 * g["curv"] + 3.5).to_numpy("float32"),
                    cw=g.get("cw", 0 * g["curv"] + 7.0).to_numpy("float32"),
                    subject=0, cond=0)
        road["v_ref"] = g["v_ref"].to_numpy("float32") if "v_ref" in g else None
        road["e_ref"] = np.zeros(len(road["curv"]), "float32")
        dd = args.dd
    else:
        roads, _, dd = load_roads(os.path.join(CACHE, args.cache or "env_roads_namsan.npz"))
        roads = trim_roads(roads)
        road = dict(roads[args.idx])
        road["e_ref"] = np.zeros_like(road["v_ref"])       # 블라인드 원칙: 사람 흔적 미사용
        if not args.geometry:
            road["v_ref"] = None                            # 속도도 생성 (완전 블라인드)

    kmax = float(np.abs(road["curv"]).max())
    if kmax > GAIN / 2.4:
        print(f"경고: max|curv|={kmax:.4f} > 권한여유 기준 {GAIN/2.4:.4f} - "
              f"급커브 구간 이탈 가능(여유율 법칙)", flush=True)

    # ---- 속도 공급 (없으면 CVAE 앙상블) ----
    if road["v_ref"] is None:
        rng = np.random.RandomState(7)
        packs = [p26.load_cvae(c) for c in ["wangsuk", "merge"]]
        vs = [p26.synth_vref(road, cvae, gs, bs, z_dim, rng)
              for (cvae, gs, bs, z_dim) in packs]
        road["v_ref"] = np.mean(vs, axis=0).astype("float32")
        v_src = "CVAE ensemble(wangsuk+merge)"
    else:
        v_src = "입력 v_ref"
    print(f"도로 {len(road['curv'])*dd/1000:.1f} km | v_ref {v_src} "
          f"평균 {np.mean(road['v_ref'])*3.6:.0f} km/h", flush=True)

    # ---- 군집 주행 ----
    ch = load_champion()
    env = DrivingEnv([road], dd=dd, record=True, steer_gain=GAIN)
    rng = np.random.RandomState(args.seed)
    sigs, offs, rows = [], 0, []
    for j in range(args.n):
        t = ch["pool"][rng.randint(len(ch["pool"]))]
        pol = p22.SpectralPolicy(ch["model"], ch["fr"], ch["A"],
                                 sigma=float(np.clip(ch["sigma"] * t["sdlp"] / ch["sdlp_pool"],
                                                     0.03, 1.2)),
                                 b_bias=t["lpm"], v_scale=t["v"] / ch["v_pool"],
                                 lib=ch["lib"], seed=args.seed * 1000 + j)
        pol.reset()
        traj, off = rollout(env, pol, 0)
        offs += int(off)
        np.savetxt(os.path.join(out_dir, f"driver_{j:02d}.csv"), traj, delimiter=",",
                   header="s_m,e_m,v_mps,a_mps2,psi_rad,steer", comments="")
        if len(traj) > 60:
            s = p20.rl_signals(traj, gain=GAIN)
            sigs.append(s)
            rows.append(dict(driver=j, sdlp=float(np.std(s["e"])),
                             lpm=float(np.mean(s["e"])),
                             v=float(np.mean(traj[:, 2])),
                             srr=float(p20.srr(s["theta"], 0.5)),
                             wl=float(p20.wavelength(s["e"])), off=bool(off)))
        if (j + 1) % 10 == 0:
            print(f"  driver {j+1}/{args.n}", flush=True)

    sd = np.array([r["sdlp"] for r in rows]); vv = np.array([r["v"] for r in rows])
    summary = dict(tag=args.tag, n=args.n, km=float(len(road["curv"]) * dd / 1000),
                   v_source=v_src, off_rate=offs / args.n,
                   sdlp=dict(mean=float(sd.mean()), std=float(sd.std())),
                   lpm=float(np.mean([r["lpm"] for r in rows])),
                   v_kmh=dict(mean=float(vv.mean() * 3.6), std=float(vv.std() * 3.6)),
                   srr=float(np.mean([r["srr"] for r in rows])),
                   wl=float(np.nanmean([r["wl"] for r in rows])), drivers=rows)
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # ---- 그림 + 미니 리포트 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    ax = axes[0]
    for s in sigs[:15]:
        ax.plot(s["s"] / 1000, s["e"], lw=0.7, alpha=0.6)
    ax.set_xlabel("거리 (km)"); ax.set_ylabel("차로중심 offset (m)")
    ax.set_title(f"가상 피실험자 {args.n}명 궤적 (표시 15명)")
    ax = axes[1]
    ax.hist(sd, bins=12, color="#7F77DD", alpha=0.8)
    ax.set_xlabel("SDLP (m)"); ax.set_ylabel("명")
    ax.set_title(f"SDLP 분포 {sd.mean():.3f}±{sd.std():.3f} m")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fig_cohort.png"), dpi=120)
    plt.close(fig)

    L = [f"# 가상 피실험자 군집 리포트 - {args.tag}", "",
         f"- 도로: {summary['km']:.1f} km, max|curv| {kmax:.4f}",
         f"- 속도 공급: {v_src} (평균 {summary['v_kmh']['mean']:.0f} km/h)",
         f"- 드라이버: 챔피언 v3.2 (2024 AUC 0.671 / 남산 블라인드 0.786 검증)",
         f"- n={args.n}, 이탈율 {summary['off_rate']:.2f}", "",
         f"| 지표 | 값 |", "|---|---|",
         f"| SDLP | {sd.mean():.3f} ± {sd.std():.3f} m |",
         f"| LPM | {summary['lpm']:.3f} m |",
         f"| 속도 | {summary['v_kmh']['mean']:.0f} ± {summary['v_kmh']['std']:.0f} km/h |",
         f"| SRR_0.5 | {summary['srr']:.1f} /km |",
         f"| 주파장 | {summary['wl']:.0f} m |", "",
         "![cohort](fig_cohort.png)", ""]
    open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8").write("\n".join(L))
    print(f"saved -> {out_dir} (driver csv x{args.n} + summary.json + report.md + fig)",
          flush=True)


if __name__ == "__main__":
    main()
