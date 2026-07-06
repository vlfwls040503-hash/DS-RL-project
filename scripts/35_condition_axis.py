# -*- coding: utf-8 -*-
"""
35_condition_axis.py  --  조건축 확장: 조건 수정자 모델 + 조건효과 보존 검증 (남산 격벽).

질문: "이 설계에 격벽(조건)을 넣으면 행태가 얼마나 변하나"를 가상 군집이 예측할 수 있는가.
남산 = 같은 터널 기하 × 4조건 × 29명(균형) → 기하와 분리된 순수 조건효과 시험대.

방법 (전이 가능한 형태):
  1) train 피험자에서 조건 수정자 추정: 기준조건 대비 (SDLP비, 속도비, LPM차)
  2) 가상 군집(챔피언 v3.2, 블라인드: e_ref=0, v_ref=CVAE 앙상블)의 노브에 수정자 적용
  3) test 피험자의 조건 대비(짝지은 Cohen's d·비율)와 대조 — 절대값이 아니라
     '조건이 만드는 변화량'의 재현이 목표 (§9 조건효과 보존 방법론의 확장)

  python 35_condition_axis.py --per_cond 20
"""
import os, json, argparse, importlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import REP, CACHE, gen_split
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
GAIN = 0.012
REF = 0                       # 기준 조건


def paired_d(a, b):
    """짝지은 Cohen's d (같은 피험자 a→b 차이)."""
    d = np.asarray(b, float) - np.asarray(a, float)
    return float(d.mean() / (d.std(ddof=1) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_cond", type=int, default=20, help="조건당 가상 드라이버 수")
    ap.add_argument("--base", default="rl_2024_wide.zip",
                    help="조향 기반 (rl_multi.zip: 남산 코너링 바닥 0.36→0.11)")
    args = ap.parse_args()

    roads, _, dd = load_roads(os.path.join(CACHE, "env_roads_namsan.npz"))
    roads = trim_roads(roads)
    subj = np.array([r["subject"] for r in roads], "int64")
    conds = sorted(set(int(r["cond"]) for r in roads))
    tr, va, te = gen_split(subj, seed=0)
    te = va | te                                    # 검증측 통계력 확보 (train 19/test 10)
    subj_tr = sorted(set(subj[tr].tolist()))
    subj_te = sorted(set(subj[te].tolist()))
    print(f"namsan conds={conds} | 수정자 추정 {len(subj_tr)}명 / 검증 {len(subj_te)}명", flush=True)

    # 기하 동일성 확인 (순수 조건효과 주장의 전제)
    base = [r for r in roads if r["cond"] == REF]
    c1 = [r for r in roads if r["cond"] == conds[1]]
    n = min(len(base[0]["curv"]), len(c1[0]["curv"]))
    cc = float(np.corrcoef(base[0]["curv"][:n], c1[0]["curv"][:n])[0, 1])
    print(f"기하 동일성: cond{REF} vs cond{conds[1]} 곡률 상관 {cc:.3f}", flush=True)

    # ---- 사람 조건별 (피험자, 조건) 행렬 ----
    def subj_cond_stats(sub_list):
        out = {}
        for s in sub_list:
            for c in conds:
                rs = [r for r in roads if r["subject"] == s and r["cond"] == c]
                if not rs:
                    continue
                r = rs[0]
                out[(s, c)] = dict(sdlp=float(np.std(r["e_ref"])),
                                   lpm=float(np.mean(r["e_ref"])),
                                   v=float(np.mean(r["v_ref"])))
        return out

    H_tr, H_te = subj_cond_stats(subj_tr), subj_cond_stats(subj_te)

    # ---- 1) 조건 수정자 (train): 기준조건 대비 ----
    mods = {}
    for c in conds:
        rs, rv, dl = [], [], []
        for s in subj_tr:
            if (s, REF) in H_tr and (s, c) in H_tr:
                rs.append(H_tr[(s, c)]["sdlp"] / max(H_tr[(s, REF)]["sdlp"], 1e-6))
                rv.append(H_tr[(s, c)]["v"] / max(H_tr[(s, REF)]["v"], 1e-6))
                dl.append(H_tr[(s, c)]["lpm"] - H_tr[(s, REF)]["lpm"])
        mods[c] = dict(r_sdlp=float(np.median(rs)), r_v=float(np.median(rv)),
                       d_lpm=float(np.median(dl)))
        print(f"  수정자 cond{c}: SDLP×{mods[c]['r_sdlp']:.3f} v×{mods[c]['r_v']:.3f} "
              f"LPM{mods[c]['d_lpm']:+.3f}", flush=True)

    # ---- 2) 가상 군집: 블라인드 + 수정자 적용 ----
    ch = p34.load_champion(base=args.base)
    packs = [p26.load_cvae(c) for c in ["wangsuk", "merge"]]
    test_geo = [r for r in roads if r["subject"] in subj_te and r["cond"] == REF]
    rngv = np.random.RandomState(3)
    # 블라인드 기본 v_ref (기하→CVAE 앙상블, 조건 무관 기저)
    blind = []
    for r in test_geo:
        r2 = dict(r)
        vs = [p26.synth_vref(r, cvae, gs, bs, z_dim, rngv)
              for (cvae, gs, bs, z_dim) in packs]
        r2["v_base"] = np.mean(vs, axis=0).astype("float32")
        r2["e_ref"] = np.zeros_like(r2["v_ref"])
        blind.append(r2)

    # σ 역산: 코너링 바닥·속도 결합이 있어 노브(σ)가 아니라 **관측 SDLP 목표**에
    # 수정자를 적용해야 함 — 조건별 프로브 후 22와 같은 1노브 보정.
    def make_env(c):
        m = mods[c]
        rds = []
        for r in blind:
            r2 = dict(r)
            r2["v_ref"] = (r["v_base"] * m["r_v"]).astype("float32")
            rds.append(r2)
        return DrivingEnv(rds, dd=dd, record=True, steer_gain=GAIN), rds

    def probe_sdlp(env, rds, sigma, n=6, seed=77):
        out = []
        for j in range(n):
            pol = p22.SpectralPolicy(ch["model"], ch["fr"], ch["A"], sigma,
                                     lib=ch["lib"], seed=seed + j)
            pol.reset()
            traj, _ = rollout(env, pol, j % len(rds))
            if len(traj) > 60:
                out.append(float(np.std(p20.rl_signals(traj, gain=GAIN)["e"])))
        return float(np.mean(out))

    env0, rds0 = make_env(REF)
    s_base = probe_sdlp(env0, rds0, ch["sigma"])
    print(f"  프로브: cond{REF} 기저 SDLP={s_base:.3f} (σ={ch['sigma']:.3f})", flush=True)

    V = {c: dict(sdlp=[], lpm=[], v=[]) for c in conds}
    rng = np.random.RandomState(0)
    for c in conds:
        m = mods[c]
        env, rds = make_env(c)
        target = s_base * m["r_sdlp"]              # 관측 목표 = 기저측정 × 수정자비
        s_meas = probe_sdlp(env, rds, ch["sigma"], seed=99 + c)
        sigma_c = float(np.clip(ch["sigma"] * (target / max(s_meas, 1e-6)) ** 2, 0.02, 1.2))
        s_chk = probe_sdlp(env, rds, sigma_c, seed=55 + c)
        print(f"  cond{c}: 목표 {target:.3f} | probe {s_meas:.3f} -> σ {sigma_c:.3f} "
              f"-> 확인 {s_chk:.3f}", flush=True)
        for j in range(args.per_cond):
            k = j % len(rds)
            t = ch["pool"][rng.randint(len(ch["pool"]))]
            pol = p22.SpectralPolicy(
                ch["model"], ch["fr"], ch["A"],
                sigma=float(np.clip(sigma_c * t["sdlp"] / ch["sdlp_pool"], 0.02, 1.2)),
                b_bias=t["lpm"] + m["d_lpm"], v_scale=t["v"] / ch["v_pool"],
                lib=ch["lib"], seed=8000 + c * 500 + j)
            pol.reset()
            traj, off = rollout(env, pol, k)
            if len(traj) > 60:
                s = p20.rl_signals(traj, gain=GAIN)
                V[c]["sdlp"].append(float(np.std(s["e"])))
                V[c]["lpm"].append(float(np.mean(s["e"])))
                V[c]["v"].append(float(np.mean(traj[:, 2])))
        print(f"  가상 cond{c}: SDLP {np.mean(V[c]['sdlp']):.3f} "
              f"v {np.mean(V[c]['v'])*3.6:.0f}km/h (n={len(V[c]['sdlp'])})", flush=True)

    # ---- 3) 검증: test 사람 조건대비 vs 가상 조건대비 ----
    res = dict(mods=mods, contrasts={})
    print("\n조건 대비 (기준 cond%d, test 피험자):" % REF, flush=True)
    for c in conds:
        if c == REF:
            continue
        h_ref = [H_te[(s, REF)]["sdlp"] for s in subj_te if (s, REF) in H_te and (s, c) in H_te]
        h_c = [H_te[(s, c)]["sdlp"] for s in subj_te if (s, REF) in H_te and (s, c) in H_te]
        hv_ref = [H_te[(s, REF)]["v"] for s in subj_te if (s, REF) in H_te and (s, c) in H_te]
        hv_c = [H_te[(s, c)]["v"] for s in subj_te if (s, REF) in H_te and (s, c) in H_te]
        d_h = paired_d(h_ref, h_c)
        r_h = float(np.mean(h_c) / np.mean(h_ref))
        rv_h = float(np.mean(hv_c) / np.mean(hv_ref))
        # 가상: 독립표본 d (군집은 짝 없음)
        a, b = np.array(V[REF]["sdlp"]), np.array(V[c]["sdlp"])
        sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        d_v = float((b.mean() - a.mean()) / (sp + 1e-12))
        r_v = float(b.mean() / a.mean())
        rv_v = float(np.mean(V[c]["v"]) / np.mean(V[REF]["v"]))
        res["contrasts"][c] = dict(d_h=d_h, d_v=d_v, r_sdlp_h=r_h, r_sdlp_v=r_v,
                                   r_v_h=rv_h, r_v_v=rv_v,
                                   sign_ok=bool(np.sign(d_h) == np.sign(d_v)))
        print(f"  cond{REF}->{c}: SDLP d 사람 {d_h:+.2f} / 가상 {d_v:+.2f} "
              f"| SDLP비 {r_h:.3f}/{r_v:.3f} | 속도비 {rv_h:.3f}/{rv_v:.3f} "
              f"| 부호 {'OK' if np.sign(d_h)==np.sign(d_v) else 'X'}", flush=True)

    # ---- 그림 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    x = np.arange(len(conds)); w = 0.35
    hm = [np.mean([H_te[(s, c)]["sdlp"] for s in subj_te if (s, c) in H_te]) for c in conds]
    vm = [np.mean(V[c]["sdlp"]) for c in conds]
    ax.bar(x - w / 2, hm, w, label="사람(test)", color="#185FA5")
    ax.bar(x + w / 2, vm, w, label="가상(수정자)", color="#7F77DD")
    ax.set_xticks(x); ax.set_xticklabels([f"cond{c}" for c in conds])
    ax.set_ylabel("SDLP (m)"); ax.legend(); ax.set_title("조건별 SDLP: 사람 vs 가상")
    ax = axes[1]
    hv = [np.mean([H_te[(s, c)]["v"] for s in subj_te if (s, c) in H_te]) * 3.6 for c in conds]
    vv = [np.mean(V[c]["v"]) * 3.6 for c in conds]
    ax.bar(x - w / 2, hv, w, label="사람(test)", color="#185FA5")
    ax.bar(x + w / 2, vv, w, label="가상", color="#7F77DD")
    ax.set_xticks(x); ax.set_xticklabels([f"cond{c}" for c in conds])
    ax.set_ylabel("속도 (km/h)"); ax.legend(); ax.set_title("조건별 속도")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_condition_axis.png"), dpi=120)
    plt.close(fig)
    json.dump(res, open(os.path.join(REP, "condition_axis.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("saved fig_condition_axis.png + condition_axis.json", flush=True)


if __name__ == "__main__":
    main()
