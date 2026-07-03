# -*- coding: utf-8 -*-
"""
26_newroad_pipeline.py  --  A3: 진짜 새 도로 파이프라인 (사람 기록 0% 가정).

새 도로에는 사람 v_ref가 없다 → CVAE(기하→속도 프로파일)가 속도를 공급하고,
챔피언 v3.2(광권한 RL 조향 + 사람 청크 주입 + PD 속도)가 주행한다.
남산을 리허설 무대로: e_ref=0, v_ref=CVAE 생성 (진짜 블라인드), 남산 사람과 비교.

  python 26_newroad_pipeline.py --cvae merge --per_road 2
  (오라클 기준선 = newroad_namsan_wide.json의 제로샷)
"""
import os, json, argparse, importlib
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from common import ART, REP, CACHE, gen_split, GEN_DD, GEN_W, wasserstein1d
from datasets import Scaler
from models import CVAE
from driving_env import DrivingEnv, load_roads, rollout, trim_roads

p20 = importlib.import_module("20_profile_eval")
p21 = importlib.import_module("21_validation")
p22 = importlib.import_module("22_v3_spectral")
p24 = importlib.import_module("24_gail_seg")

for fam in ["Malgun Gothic", "NanumGothic", "Gulim"]:
    try:
        plt.rcParams["font.family"] = fam; break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(REP, "figs"); os.makedirs(FIG, exist_ok=True)
np.random.seed(0); torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
GAIN = 0.012


def load_cvae(exp):
    d = torch.load(os.path.join(ART, f"cvae_{exp}.pt"), map_location=DEV, weights_only=False)
    m = CVAE(beh_dim=d["beh_dim"], geo_dim=d["geo_dim"], z_dim=d["z_dim"],
             stochastic=bool(d.get("stochastic")), stoch_dim=d.get("stoch_dim"))
    m.load_state_dict(d["state"]); m.to(DEV).eval()
    return m, Scaler.from_dict(d["geo_scaler"]), Scaler.from_dict(d["beh_scaler"]), d["z_dim"]


def synth_vref(road, cvae, gs, bs, z_dim, rng):
    """기하만으로 속도 프로파일 생성 (창 256pt=512m, 이어붙임 + 50m 평활)."""
    N = len(road["curv"])
    geo = np.stack([road["curv"], road["slope"], np.zeros(N, np.float32),
                    road["lane_w"], road["cw"]], axis=1).astype("float32")
    geo = gs.transform(geo[None])[0]
    z = rng.randn(z_dim).astype("float32")          # 도로당 1개 = 일관된 속도 스타일
    v = np.zeros(N, np.float64); wsum = np.zeros(N, np.float64)
    starts = list(range(0, max(N - GEN_W, 1), GEN_W // 2)) + [max(N - GEN_W, 0)]
    with torch.no_grad():
        for a in starts:
            gw = torch.from_numpy(geo[None, a:a + GEN_W]).to(DEV)
            zt = torch.from_numpy(z[None]).to(DEV)
            mu, _ = cvae.decode_dist(zt, gw)
            beh = bs.inverse(mu.cpu().numpy())[0]   # (W,2): [offset, speed]
            w = np.hanning(len(beh)) + 1e-3
            v[a:a + len(beh)] += beh[:, 1] * w
            wsum[a:a + len(beh)] += w
    v = v / np.maximum(wsum, 1e-9)
    k = np.ones(25) / 25.0                          # 50 m 평활
    v = np.convolve(np.pad(v, (12, 12), mode="edge"), k, mode="valid")[:N]
    return np.clip(v, 3.0, 40.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_road", type=int, default=2)
    ap.add_argument("--cvaes", default="2024,merge")
    args = ap.parse_args()

    # ---- 챔피언 v3.2 자산 (2024) ----
    roads24, _, dd24 = load_roads(os.path.join(CACHE, "env_roads_2024.npz"))
    roads24 = trim_roads(roads24)
    subj24 = np.array([r["subject"] for r in roads24], "int64")
    tr, va, te = gen_split(subj24, seed=0)
    fit24 = [r for r, m in zip(roads24, tr | va) if m]
    fit_chunks = []
    for r in fit24:
        for ch in p21.chunk_signals(p20.human_signals(r, dd24)):
            fit_chunks.append(ch["e"])
    fr, A = p22.target_spectrum(fit_chunks)
    lib = []
    for e in fit_chunks:
        x = np.asarray(e, np.float64); x = x - x.mean()
        if len(x) >= 400 and x.std() > 1e-3:
            lib.append((x / x.std()).astype(np.float64))
    model = PPO.load(os.path.join(ART, "rl_2024_wide.zip"), device="cpu")
    cal = json.load(open(os.path.join(REP, "v3_library_2024_wide.json"), encoding="utf-8"))
    sigma = float(cal["sigma"])
    pool = [dict(sdlp=float(np.std(r["e_ref"])), lpm=float(np.mean(r["e_ref"])),
                 v=float(np.mean(r["v_ref"]))) for r in fit24]
    sdlp_pool = float(np.mean([t["sdlp"] for t in pool]))
    v_pool = float(np.mean([t["v"] for t in pool]))

    # ---- 남산 (기하만 사용한다고 가정) ----
    roadsN, _, ddN = load_roads(os.path.join(CACHE, "env_roads_namsan.npz"))
    roadsN = trim_roads(roadsN)
    hN = [p20.human_signals(r, ddN) for r in roadsN]
    XH = np.vstack([p21.seg_features(h) for h in hN])
    v_h_roads = np.array([float(np.mean(r["v_ref"])) for r in roadsN])
    print(f"namsan {len(roadsN)} roads | human v={v_h_roads.mean()*3.6:.0f}km/h", flush=True)

    oracle = json.load(open(os.path.join(REP, "newroad_namsan_wide.json"), encoding="utf-8"))

    out = dict(oracle_zero_shot=dict(auc=oracle["zero_shot"]["auc"],
                                     off=oracle["zero_shot"]["off"],
                                     tex=oracle["zero_shot"]["tex"]))
    bars = [("오라클 v_ref\n(사람속도)", oracle["zero_shot"]["auc"], "#888780")]
    for cv_exp in args.cvaes.split(","):
        cvae, gs, bs, z_dim = load_cvae(cv_exp)
        rng = np.random.RandomState(11)
        # 블라인드 도로: e_ref=0, v_ref=CVAE
        blind, v_pred_roads = [], []
        for r in roadsN:
            r2 = dict(r)
            r2["v_ref"] = synth_vref(r, cvae, gs, bs, z_dim, rng)
            r2["e_ref"] = np.zeros_like(r2["v_ref"])
            v_pred_roads.append(float(np.mean(r2["v_ref"])))
            blind.append(r2)
        v_pred_roads = np.array(v_pred_roads)
        w1_v = wasserstein1d(v_h_roads, v_pred_roads)
        print(f"[cvae_{cv_exp}] v pred={v_pred_roads.mean()*3.6:.0f}km/h "
              f"(사람 {v_h_roads.mean()*3.6:.0f}) | 도로평균속도 W1={w1_v:.2f} m/s", flush=True)

        env = DrivingEnv(blind, dd=ddN, record=True, steer_gain=GAIN)
        rng2 = np.random.RandomState(5)
        S, offs, n = [], 0, 0
        for k in range(len(blind)):
            for j in range(args.per_road):
                t = pool[rng2.randint(len(pool))]
                pol = p22.SpectralPolicy(model, fr, A,
                                         sigma=float(np.clip(sigma * t["sdlp"] / sdlp_pool, 0.03, 1.2)),
                                         b_bias=t["lpm"], v_scale=t["v"] / v_pool, lib=lib,
                                         seed=6000 + k * 20 + j)
                pol.reset()
                traj, o = rollout(env, pol, k)
                offs += int(o); n += 1
                if len(traj) > 60:
                    S.append(p20.rl_signals(traj, gain=GAIN))
            if (k + 1) % 30 == 0:
                print(f"  [{cv_exp}] road {k+1}/{len(blind)}", flush=True)
        off_rate = offs / max(n, 1)
        tex = dict(sdlp=float(np.mean([np.std(s["e"]) for s in S])),
                   wl=float(np.nanmean([p20.wavelength(s["e"]) for s in S])),
                   srr=float(np.mean([p20.srr(s["theta"], 0.5) for s in S])))
        XS = np.vstack([p21.seg_features(s) for s in S])
        rng3 = np.random.RandomState(2)
        nmin = min(len(XH), len(XS))
        X = np.vstack([XH[rng3.choice(len(XH), nmin, replace=False)],
                       XS[rng3.choice(len(XS), nmin, replace=False)]])
        yy = np.concatenate([np.zeros(nmin), np.ones(nmin)])
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        auc = p24.cv_auc(X, yy)
        print(f"[cvae_{cv_exp}] AUC={auc:.3f} off={off_rate:.2f} SDLP={tex['sdlp']:.3f} "
              f"wl={tex['wl']:.0f} SRR={tex['srr']:.1f}", flush=True)
        out[f"cvae_{cv_exp}"] = dict(auc=auc, off=off_rate, tex=tex, w1_v_road=float(w1_v),
                                     v_pred_mean=float(v_pred_roads.mean()),
                                     v_human_mean=float(v_h_roads.mean()))
        bars.append((f"CVAE {cv_exp}\nv_ref", auc, "#7F77DD"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    ax.bar([b[0] for b in bars], [b[1] for b in bars], color=[b[2] for b in bars])
    ax.axhline(0.5, ls=":", color="#185FA5", label="구별불가(0.5)")
    for i, b in enumerate(bars):
        ax.text(i, b[1], f"{b[1]:.3f}", ha="center", va="bottom")
    ax.set_ylim(0.4, 1.05); ax.set_ylabel("C2ST AUC"); ax.legend()
    ax.set_title("새도로 파이프라인 (남산 블라인드): 속도 공급원별")
    ax = axes[1]
    ax.hist(v_h_roads * 3.6, bins=20, alpha=0.6, label="남산 사람", color="#185FA5")
    for cv_exp in args.cvaes.split(","):
        vp = out[f"cvae_{cv_exp}"]["v_pred_mean"] * 3.6
        ax.axvline(vp, ls="--", lw=2, label=f"CVAE {cv_exp} 평균 {vp:.0f}")
    ax.set_xlabel("도로 평균속도 (km/h)"); ax.legend(); ax.set_title("속도 예측 vs 사람")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_newroad_pipeline.png"), dpi=120)
    plt.close(fig)
    json.dump(out, open(os.path.join(REP, "newroad_pipeline.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved fig_newroad_pipeline.png + newroad_pipeline.json", flush=True)


if __name__ == "__main__":
    main()
