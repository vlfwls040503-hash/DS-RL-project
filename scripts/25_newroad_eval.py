# -*- coding: utf-8 -*-
"""
25_newroad_eval.py  --  새 도로 일반화 검증: 2024 챔피언(v3.1)을 남산터널에 제로샷 투입.

원래 프로젝트 목적("새 도로 설계안을 가상 피실험자로 평가")의 첫 실전 시험.
2024 지하고속도로에서 학습·보정한 챔피언 드라이버 일체(RL 조향정책 + 사람 청크
라이브러리 + sigma 보정값 + 특성풀)를 전혀 다른 기하(남산터널, 29명 116주행)에 투입,
남산 사람 주행과 C2ST/텍스처로 비교.

  variant A (zero-shot): 남산 데이터 0% 사용 — 순수 전이
  variant B (few-shot) : 남산 피험자 20%(val)로 sigma 1노브만 재보정 — 파일럿 시나리오

  python 25_newroad_eval.py --per_road 3
"""
import os, json, argparse, importlib
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from common import ART, REP, CACHE, gen_split
from driving_env import DrivingEnv, load_roads, rollout, trim_roads

p20 = importlib.import_module("20_profile_eval")
p21 = importlib.import_module("21_validation")
p22 = importlib.import_module("22_v3_spectral")

for fam in ["Malgun Gothic", "NanumGothic", "Gulim"]:
    try:
        plt.rcParams["font.family"] = fam; break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(REP, "figs"); os.makedirs(FIG, exist_ok=True)
np.random.seed(0); torch.manual_seed(0)


def texture(sigs):
    return dict(sdlp=float(np.mean([np.std(s["e"]) for s in sigs])),
                wl=float(np.nanmean([p20.wavelength(s["e"]) for s in sigs])),
                srr=float(np.mean([p20.srr(s["theta"], 0.5) for s in sigs])),
                srr2=float(np.mean([p20.srr(s["theta"], 2.0) for s in sigs])))


def c2st(H_sigs, S_sigs):
    XH = np.vstack([p21.seg_features(h) for h in H_sigs])
    XS = np.vstack([p21.seg_features(s) for s in S_sigs])
    rng = np.random.RandomState(2)
    nmin = min(len(XH), len(XS))
    X = np.vstack([XH[rng.choice(len(XH), nmin, replace=False)],
                   XS[rng.choice(len(XS), nmin, replace=False)]])
    y = np.concatenate([np.zeros(nmin), np.ones(nmin)])
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return p22.cv_auc(X, y), nmin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_road", type=int, default=3)
    args = ap.parse_args()

    # ---- 소스 도메인(2024): 챔피언 자산 일체 ----
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
    model = PPO.load(os.path.join(ART, "rl_2024.zip"), device="cpu")
    cal = json.load(open(os.path.join(REP, "v3_library_2024.json"), encoding="utf-8"))
    sigma24 = float(cal["sigma"])
    pool = [dict(sdlp=float(np.std(r["e_ref"])), lpm=float(np.mean(r["e_ref"])),
                 v=float(np.mean(r["v_ref"]))) for r in fit24]
    sdlp_pool = float(np.mean([t["sdlp"] for t in pool]))
    v_pool = float(np.mean([t["v"] for t in pool]))
    print(f"champion assets: lib={len(lib)} chunks, sigma24={sigma24:.3f}, pool n={len(pool)}",
          flush=True)

    # ---- 타깃 도메인(남산): 완전 미학습 기하 ----
    roadsN, _, ddN = load_roads(os.path.join(CACHE, "env_roads_namsan.npz"))
    roadsN = trim_roads(roadsN)
    subjN = np.array([r["subject"] for r in roadsN], "int64")
    trN, vaN, teN = gen_split(subjN, seed=0)
    val_roads = [r for r, m in zip(roadsN, vaN) if m]       # few-shot sigma 재보정용
    rest_roads = [r for r, m in zip(roadsN, ~vaN) if m]     # few-shot 평가용
    kmax = max(float(np.abs(r["curv"]).max()) for r in roadsN)
    hN_all = [p20.human_signals(r, ddN) for r in roadsN]
    texH = texture(hN_all)
    vN = float(np.mean([np.mean(r["v_ref"]) for r in roadsN]))
    print(f"namsan: {len(roadsN)} roads / {len(set(subjN))} subj | "
          f"human SDLP={texH['sdlp']:.3f} wl={texH['wl']:.0f} SRR={texH['srr']:.1f} "
          f"v={vN*3.6:.0f}km/h | max|curv|={kmax:.4f} (조향한계 0.005)", flush=True)

    def run_cohort(target_roads, sigma, seed0, tag):
        env = DrivingEnv(target_roads, dd=ddN, record=True)
        rng = np.random.RandomState(seed0)
        S, offs, n = [], 0, 0
        for k, road in enumerate(target_roads):
            for j in range(args.per_road):
                t = pool[rng.randint(len(pool))]            # 특성은 2024 풀에서 전이
                pol = p22.SpectralPolicy(
                    model, fr, A,
                    sigma=float(np.clip(sigma * t["sdlp"] / sdlp_pool, 0.03, 1.2)),
                    b_bias=t["lpm"], v_scale=t["v"] / v_pool, lib=lib,
                    seed=seed0 + k * 20 + j)
                pol.reset()
                traj, o = rollout(env, pol, k)
                offs += int(o); n += 1
                if len(traj) > 60:
                    S.append(p20.rl_signals(traj))
            if (k + 1) % 20 == 0:
                print(f"  [{tag}] road {k+1}/{len(target_roads)}", flush=True)
        return S, offs / max(n, 1)

    # ---- A) 제로샷: 남산 정보 0, 2024 보정 그대로, 전체 116도로 ----
    S_zero, off_zero = run_cohort(roadsN, sigma24, 5000, "zero")
    auc_zero, n_zero = c2st(hN_all, S_zero)
    texZ = texture(S_zero)
    print(f"[zero-shot] AUC={auc_zero:.3f} (2024 내부 0.794) off={off_zero:.2f} "
          f"SDLP {texH['sdlp']:.3f}/{texZ['sdlp']:.3f} wl {texH['wl']:.0f}/{texZ['wl']:.0f} "
          f"SRR {texH['srr']:.1f}/{texZ['srr']:.1f}", flush=True)

    # ---- B) 퓨샷: 남산 val 피험자(20%)로 sigma 1노브 재보정 ----
    t_std = float(np.mean([np.std(p20.human_signals(r, ddN)["e"]) for r in val_roads]))
    env_v = DrivingEnv(val_roads, dd=ddN, record=True)

    def probe(sig):
        stds = []
        for k in range(len(val_roads)):
            pol = p22.SpectralPolicy(model, fr, A, sig, lib=lib, seed=k)
            pol.reset()
            traj, _ = rollout(env_v, pol, k)
            if len(traj) > 60:
                stds.append(np.std(p20.rl_signals(traj)["e"]))
        return float(np.mean(stds))

    sd1 = probe(t_std)
    sigmaN = float(np.clip(t_std * t_std / max(sd1, 1e-6), 0.03, 1.2))
    print(f"[few-shot] namsan 목표 e-std={t_std:.3f}, sigma {sigma24:.3f} -> {sigmaN:.3f}",
          flush=True)
    hN_rest = [p20.human_signals(r, ddN) for r in rest_roads]
    S_few, off_few = run_cohort(rest_roads, sigmaN, 7000, "few")
    auc_few, n_few = c2st(hN_rest, S_few)
    texF = texture(S_few)
    texHr = texture(hN_rest)
    print(f"[few-shot] AUC={auc_few:.3f} off={off_few:.2f} "
          f"SDLP {texHr['sdlp']:.3f}/{texF['sdlp']:.3f} wl {texHr['wl']:.0f}/{texF['wl']:.0f} "
          f"SRR {texHr['srr']:.1f}/{texF['srr']:.1f}", flush=True)

    # ---- 그림 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    names = ["2024 내부\n(기준)", "남산 제로샷", "남산 퓨샷\n(σ재보정)"]
    vals = [0.794, auc_zero, auc_few]
    ax.bar(names, vals, color=["#888780", "#7F77DD", "#1D9E75"])
    ax.axhline(0.5, ls=":", color="#185FA5", label="구별불가(0.5)")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    ax.set_ylim(0.4, 1.05); ax.set_ylabel("C2ST AUC")
    ax.set_title("새 도로(남산터널) 일반화: 판별자 AUC")
    ax = axes[1]
    labels = ["SDLP(m)", "파장/1000(m)", "SRR_0.5(/km)"]
    hvals = [texH["sdlp"], texH["wl"] / 1000, texH["srr"]]
    zvals = [texZ["sdlp"], texZ["wl"] / 1000, texZ["srr"]]
    fvals = [texF["sdlp"], texF["wl"] / 1000, texF["srr"]]
    x = np.arange(len(labels)); w = 0.26
    ax.bar(x - w, hvals, w, label="남산 사람", color="#185FA5")
    ax.bar(x, zvals, w, label="제로샷", color="#7F77DD")
    ax.bar(x + w, fvals, w, label="퓨샷", color="#1D9E75")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend()
    ax.set_title("텍스처 지표 (남산 test)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_newroad_namsan.png"), dpi=120)
    plt.close(fig)

    json.dump(dict(source="2024", target="namsan",
                   n_roads=len(roadsN), n_subj=int(len(set(subjN))), max_curv=kmax,
                   zero_shot=dict(auc=auc_zero, off=off_zero, n_seg=n_zero, tex=texZ),
                   few_shot=dict(auc=auc_few, off=off_few, n_seg=n_few, sigma=sigmaN,
                                 tex=texF),
                   human_tex=texH, sigma24=sigma24),
              open(os.path.join(REP, "newroad_namsan.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved fig_newroad_namsan.png + newroad_namsan.json", flush=True)


if __name__ == "__main__":
    main()
