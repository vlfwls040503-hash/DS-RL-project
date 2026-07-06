# -*- coding: utf-8 -*-
"""
37_preregister.py  --  전향적 검증 사전등록: 홍대 지하차도 조명 실험.

실험 '전에' 가상 군집의 예측을 생성·해시·커밋(타임스탬프 공증)해 두고, 실험 후
실측과 대조한다. 맞으면 "가상 피실험자 = 검증된 방법론"이 된다.

구성:
  1) 조명류 조건 수정자: 2024 지상/지하 짝(32명) 대비에서, 같은 도로를 달린
     v3.3 가상(기하만 앎)의 대비를 나눠 기하 성분을 제거 → 순수 환경(조명류) 효과
     + 부트스트랩 95% CI.
  2) 예측 생성: 홍대 기하 CSV → CVAE 앙상블 속도 → 모집단 모델(95명 은행,
     퍼짐 보정) → 조건별 SDLP/LPM/속도 평균±CI + 검정력 표.
  3) 잠금: preregister_{tag}.json + SHA256 → 공개 레포 커밋이 공증.

  python 37_preregister.py --dry_run              # 대역 기하로 전 과정 통주
  python 37_preregister.py --geometry hongdae.csv --n_planned 30 --tag hongdae
"""
import os, json, argparse, hashlib, importlib, datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import REP, CACHE, gen_split, GEN_DD
from driving_env import DrivingEnv, load_roads, rollout, trim_roads

p20 = importlib.import_module("20_profile_eval")
p22 = importlib.import_module("22_v3_spectral")
p26 = importlib.import_module("26_newroad_pipeline")
p34 = importlib.import_module("34_virtual_cohort")
p36 = importlib.import_module("36_population")

for fam in ["Malgun Gothic", "NanumGothic", "Gulim"]:
    try:
        plt.rcParams["font.family"] = fam; break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(REP, "figs"); os.makedirs(FIG, exist_ok=True)
np.random.seed(0)
GAIN, BASE = 0.012, "rl_multi.zip"          # v3.3


def lighting_modifier(ch, nb=200):
    """2024 지상/지하: 사람 짝 대비 ÷ 가상(기하만) 대비 = 순수 환경효과 (+CI)."""
    roads, _, dd = load_roads(os.path.join(CACHE, "env_roads_2024.npz"))
    roads = trim_roads(roads)
    by = {}
    for r in roads:
        by.setdefault(int(r["subject"]), {})[int(r["cond"])] = r
    pairs = [(d[0], d[1]) for s, d in sorted(by.items()) if 0 in d and 1 in d]
    h0 = np.array([np.std(a["e_ref"]) for a, b in pairs])
    h1 = np.array([np.std(b["e_ref"]) for a, b in pairs])
    # 가상 기하 대조군: 같은 도로들(σ 고정, e_ref=0)
    def virt(rs, seed0):
        rs2 = []
        for r in rs[:10]:
            r2 = dict(r); r2["e_ref"] = np.zeros_like(r2["v_ref"]); rs2.append(r2)
        env = DrivingEnv(rs2, dd=dd, record=True, steer_gain=GAIN)
        out = []
        for j in range(20):
            pol = p22.SpectralPolicy(ch["model"], ch["fr"], ch["A"], ch["sigma"],
                                     lib=ch["lib"], seed=seed0 + j)
            pol.reset()
            traj, _ = rollout(env, pol, j % len(rs2))
            if len(traj) > 60:
                out.append(float(np.std(p20.rl_signals(traj, gain=GAIN)["e"])))
        return np.array(out)
    g0 = virt([a for a, b in pairs], 100)
    g1 = virt([b for a, b in pairs], 300)
    r_geo = g1.mean() / g0.mean()
    rng = np.random.RandomState(0)
    ratios = []
    for _ in range(nb):
        i = rng.randint(len(pairs), size=len(pairs))
        ratios.append((h1[i].mean() / h0[i].mean()) / r_geo)
    r_net = float(np.median(ratios))
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    dif = h1 - h0
    return dict(r_sdlp=r_net, ci=[float(lo), float(hi)], r_human=float(h1.mean() / h0.mean()),
                r_geo=float(r_geo), d_paired=float(dif.mean() / dif.std(ddof=1)),
                n_pairs=len(pairs), source="2024 지상/지하 (조명류 최근접 대용물)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="", help="홍대 기하 CSV (curv,slope,lane_w,cw @2m)")
    ap.add_argument("--dry_run", action="store_true", help="대역 기하(남산 도로0)로 통주")
    ap.add_argument("--n_planned", type=int, default=30)
    ap.add_argument("--n_virtual", type=int, default=29)
    ap.add_argument("--tag", default="hongdae")
    args = ap.parse_args()
    tag = args.tag + ("_DRYRUN" if args.dry_run else "")

    ch = p34.load_champion(base=BASE)

    # ---- 1) 조명류 수정자 ----
    mod = lighting_modifier(ch)
    print(f"조명류 수정자: 사람비 {mod['r_human']:.3f} ÷ 기하비 {mod['r_geo']:.3f} "
          f"= 순수 {mod['r_sdlp']:.3f} [CI {mod['ci'][0]:.3f}~{mod['ci'][1]:.3f}] "
          f"(짝 d={mod['d_paired']:.2f}, n={mod['n_pairs']})", flush=True)

    # ---- 도로 ----
    if args.dry_run:
        roadsN, _, dd = load_roads(os.path.join(CACHE, "env_roads_namsan.npz"))
        road = dict(trim_roads(roadsN)[0])
        geo_src = "DRYRUN: 남산 도로0 대역"
    else:
        import pandas as pd
        g = pd.read_csv(args.geometry)
        road = dict(curv=g["curv"].to_numpy("float32"),
                    slope=g.get("slope", 0 * g["curv"]).to_numpy("float32"),
                    lane_w=g.get("lane_w", 0 * g["curv"] + 3.5).to_numpy("float32"),
                    cw=g.get("cw", 0 * g["curv"] + 7.0).to_numpy("float32"),
                    subject=0, cond=0, v_ref=None, e_ref=None)
        dd = GEN_DD
        geo_src = os.path.basename(args.geometry)
    road["e_ref"] = np.zeros(len(road["curv"]), "float32")
    kmax = float(np.abs(road["curv"]).max())
    packs = [p26.load_cvae(c) for c in ["wangsuk", "merge"]]
    vs = [p26.synth_vref(road, cvae, gs, bs, z_dim, np.random.RandomState(7))
          for (cvae, gs, bs, z_dim) in packs]
    road["v_ref"] = np.mean(vs, axis=0).astype("float32")
    print(f"기하: {geo_src} | {len(road['curv'])*dd/1000:.1f}km max|curv| {kmax:.4f} "
          f"| CVAE 속도 {np.mean(road['v_ref'])*3.6:.0f}km/h "
          f"{'[!]권한여유 미달' if kmax > GAIN/2.4 else ''}", flush=True)

    # ---- 2) 모집단 예측 (36의 은행+프로브맵 방식) ----
    Z = []
    for exp in ["2024", "namsan", "wangsuk"]:
        T = np.array(list(p36.subject_traits(exp)[0].values()))
        Z.append((T - T.mean(0)) / (T.std(0) + 1e-9))
    Z = np.vstack(Z)
    # 스케일 사전가정: 지하차도 55km/h급 = 남산 모멘트 (파일럿 없음 명시)
    Tn = np.array(list(p36.subject_traits("namsan")[0].values()))
    mu_t, sd_t = Tn.mean(0), Tn.std(0)

    env = DrivingEnv([road], dd=dd, record=True, steer_gain=GAIN)

    def probe(sig, n=5, seed=90):
        out = []
        for j in range(n):
            pol = p22.SpectralPolicy(ch["model"], ch["fr"], ch["A"], sig,
                                     lib=ch["lib"], seed=seed + j)
            pol.reset()
            traj, _ = rollout(env, pol, 0)
            if len(traj) > 60:
                out.append(float(np.std(p20.rl_signals(traj, gain=GAIN)["e"])))
        return float(np.mean(out))

    sig_grid = np.array([0.05, 0.2, 0.4])
    sd_grid = np.array([probe(s, seed=90 + i * 7) for i, s in enumerate(sig_grid)])
    b, a = np.polyfit(sig_grid, sd_grid, 1)
    print(f"프로브맵: {np.round(sd_grid,3).tolist()} (a={a:.3f} b={b:.3f})", flush=True)
    v_base = float(np.mean(road["v_ref"]))

    conds = {"C0_기준조명": 1.0, "C1_악화조명": mod["r_sdlp"]}
    rng = np.random.RandomState(1)
    zdraw = Z[rng.randint(len(Z), size=args.n_virtual)]
    base_t = mu_t + zdraw * sd_t
    pred = {}
    for cname, rmod in conds.items():
        vals = []
        for j, t in enumerate(base_t):
            tgt = float(np.clip(t[0] * rmod, 0.08, 0.6))
            sig_j = float(np.clip((tgt - a) / max(b, 1e-6), 0.02, 1.2))
            pol = p22.SpectralPolicy(ch["model"], ch["fr"], ch["A"], sig_j,
                                     b_bias=float(t[1]), v_scale=float(t[2]) / v_base,
                                     lib=ch["lib"], seed=1000 + j)
            pol.reset()
            traj, _ = rollout(env, pol, 0)
            if len(traj) > 60:
                s = p20.rl_signals(traj, gain=GAIN)
                vals.append([float(np.std(s["e"])), float(np.mean(s["e"])),
                             float(np.mean(traj[:, 2]))])
        V = np.array(vals)
        boot = np.array([V[np.random.RandomState(k).randint(len(V), size=len(V)), 0].mean()
                         for k in range(200)])
        pred[cname] = dict(n=len(V),
                           sdlp=dict(mean=float(V[:, 0].mean()), sd=float(V[:, 0].std()),
                                     ci95=[float(np.percentile(boot, 2.5)),
                                           float(np.percentile(boot, 97.5))]),
                           lpm=float(V[:, 1].mean()),
                           v_kmh=float(V[:, 2].mean() * 3.6))
        print(f"  {cname}: SDLP {pred[cname]['sdlp']['mean']:.3f}±{pred[cname]['sdlp']['sd']:.3f} "
              f"CI[{pred[cname]['sdlp']['ci95'][0]:.3f},{pred[cname]['sdlp']['ci95'][1]:.3f}] "
              f"v {pred[cname]['v_kmh']:.0f}km/h", flush=True)

    # 조건대비 예측 + 검정력
    s0, s1 = pred["C0_기준조명"], pred["C1_악화조명"]
    sp = np.sqrt((s0["sdlp"]["sd"] ** 2 + s1["sdlp"]["sd"] ** 2) / 2)
    d_pred = float((s1["sdlp"]["mean"] - s0["sdlp"]["mean"]) / (sp + 1e-12))
    power = {}
    for n in [10, 20, 30]:
        se = sp * np.sqrt(2 / n)
        power[n] = float(min(1.0, max(0.0, 1 - 0.5 * np.exp(
            -max(abs(d_pred) * np.sqrt(n / 2) - 1.96, 0)))))  # 근사 (정규)
    out = dict(experiment="홍대 지하차도 조명 실험 (전향적 사전등록)",
               created=datetime.datetime.now().isoformat(), dry_run=bool(args.dry_run),
               geometry=geo_src, model="champion v3.3 (rl_multi + library + CVAE ens)",
               assumptions=["조명 수정자는 2024 지상/지하 대용물(순수화: 가상 기하대조군)",
                            "스케일 사전가정 = 남산 모멘트(지하차도 55km/h급); 파일럿시 갱신",
                            "예측은 200m 구간 지표 아님 — 주행 전체 SDLP/LPM/속도"],
               lighting_modifier=mod, predictions=pred,
               contrast=dict(d_sdlp=d_pred,
                             direction="악화조명에서 SDLP 증가" if d_pred > 0 else "감소"),
               power_normal_approx=power, n_planned=args.n_planned)
    p = os.path.join(REP, f"preregister_{tag}.json")
    txt = json.dumps(out, ensure_ascii=False, indent=2)
    open(p, "w", encoding="utf-8").write(txt)
    # 내용 해시: 타임스탬프 제외 → 재실행 재현성 감사 가능 (커밋이 시점 공증 담당)
    core = {k: v for k, v in out.items() if k != "created"}
    sha = hashlib.sha256(json.dumps(core, ensure_ascii=False, sort_keys=True)
                         .encode("utf-8")).hexdigest()
    print(f"\n예측 대비: SDLP d={d_pred:+.2f} ({out['contrast']['direction']})", flush=True)
    print(f"SHA256: {sha}", flush=True)
    open(os.path.join(REP, f"preregister_{tag}.sha256"), "w").write(sha + "\n")
    print(f"saved {os.path.basename(p)} + .sha256 (커밋 = 타임스탬프 공증)", flush=True)


if __name__ == "__main__":
    main()
