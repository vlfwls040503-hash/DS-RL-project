# -*- coding: utf-8 -*-
"""
15_cross_validate.py  --  cross-experiment transfer test.
Model trained on X is evaluated against REAL human driving on experiment Y's roads
(same geometry feature intersection). Quantifies geometry->behavior transfer gap.

  python 15_cross_validate.py --smoke
  python 15_cross_validate.py --train_exp 2024 --val_exp namsan
"""
import os, sys, json, subprocess, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (ART, REP, CACHE, GEN_DD, build_smoke_dataset, gen_split,
                    wasserstein1d, ks_stat)
from datasets import Scaler
from models import CVAE

DEV = "cuda" if torch.cuda.is_available() else "cpu"
for fam in ["Malgun Gothic", "NanumGothic", "Gulim"]:
    try:
        plt.rcParams["font.family"] = fam; break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(REP, "figs"); os.makedirs(FIG, exist_ok=True)


def load_ckpt(exp):
    ck = torch.load(os.path.join(ART, f"cvae_{exp}.pt"), map_location=DEV, weights_only=False)
    m = CVAE(beh_dim=ck["beh_dim"], geo_dim=ck["geo_dim"], z_dim=ck["z_dim"])
    m.load_state_dict(ck["state"]); m.to(DEV).eval()
    return m, Scaler.from_dict(ck["geo_scaler"]), Scaler.from_dict(ck["beh_scaler"]), ck


def summaries(beh, dd):
    off, spd = beh[:, :, 0], beh[:, :, 1]
    return dict(sdlp=off.std(axis=1), mean_speed=spd.mean(axis=1), speed_std=spd.std(axis=1))


def decode_samples(model, gs, bs, Xg, z_dim, chunk=512):
    out = []
    with torch.no_grad():
        for s in range(0, len(Xg), chunk):
            geo = torch.from_numpy(gs.transform(Xg[s:s + chunk])).to(DEV)
            z = torch.randn(geo.shape[0], z_dim, device=DEV)
            out.append(bs.inverse(model.decode(z, geo).cpu().numpy()))
    return np.concatenate(out)


def build_val(val_exp, feat_geo, smoke):
    """Return (X_geo, Y_beh, dd) for the validation experiment with X's geo feature set."""
    if smoke:
        p = os.path.join(CACHE, "dataset_gen_smokeval.npz")
        build_smoke_dataset(p, geo_cols=feat_geo, seed=1, n_subj=8)   # seed shift = domain shift
    else:
        p = os.path.join(CACHE, f"dataset_gen_{val_exp}_for_xval.npz")
        cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "12_build_gen_dataset.py"),
               "--exp", val_exp, "--feat_set", ",".join(feat_geo), "--out", p]
        print("running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    d = np.load(p, allow_pickle=True)
    dd = float(d["dd"]) if "dd" in d else GEN_DD
    return d["X_geo"].astype("float32"), d["Y_beh"].astype("float32"), dd, [str(c) for c in d["feat_geo"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_exp", default="smoke")
    ap.add_argument("--val_exp", default="smokeval")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    train_exp = "smoke" if args.smoke else args.train_exp
    val_exp = "smokeval" if args.smoke else args.val_exp

    model, gs, bs, ck = load_ckpt(train_exp)
    feat_geo = [c for c in ck["feat_geo"] if not c.startswith("cond_")]
    if len(feat_geo) != ck["geo_dim"]:
        raise SystemExit("교차실험은 조건(cond_*) 미포함 generic 모델이어야 함. --use_condition 없이 학습한 모델을 쓰세요.")
    print(f"train={train_exp} -> val={val_exp} | geo feat={feat_geo}", flush=True)

    Xv, Yv, dd, val_feat = build_val(val_exp, feat_geo, args.smoke)
    if val_feat != feat_geo:
        raise SystemExit(f"geo feature 불일치: train {feat_geo} vs val {val_feat}")
    # generate over Y geometry using X's scalers
    gen = decode_samples(model, gs, bs, Xv, ck["z_dim"])
    rs, gsum = summaries(Yv, dd), summaries(gen, dd)

    cross = {}
    for k in ["sdlp", "mean_speed", "speed_std"]:
        cross[k] = dict(wasserstein=wasserstein1d(gsum[k], rs[k]), ks=ks_stat(gsum[k], rs[k]),
                        real_mean=float(np.mean(rs[k])), gen_mean=float(np.mean(gsum[k])))

    # within-experiment reference (if available)
    within = None
    wp = os.path.join(REP, f"eval_gen_{train_exp}.json")
    if os.path.exists(wp):
        within = json.load(open(wp, encoding="utf-8")).get("distribution")

    # figure: gen vs val-real distributions
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, k, t in zip(axes, ["sdlp", "mean_speed", "speed_std"], ["SDLP", "평균속도(m/s)", "속도std"]):
        lo, hi = np.percentile(np.concatenate([rs[k], gsum[k]]), [1, 99])
        b = np.linspace(lo, hi, 40)
        ax.hist(rs[k], bins=b, alpha=.5, density=True, label=f"{val_exp} 실제", color="#185FA5")
        ax.hist(gsum[k], bins=b, alpha=.5, density=True, label=f"{train_exp} 모델 생성", color="#1D9E75")
        ax.set_title(f"{t}  W1={cross[k]['wasserstein']:.3g}"); ax.legend()
    fig.suptitle(f"교차실험 전이: {train_exp} 모델 → {val_exp} 도로/실제"); fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"fig_cross_{train_exp}_to_{val_exp}.png"), dpi=120); plt.close(fig)

    json.dump(dict(train_exp=train_exp, val_exp=val_exp, n_val_windows=int(len(Xv)),
                   cross=cross, within=within),
              open(os.path.join(REP, f"cross_{train_exp}_to_{val_exp}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # append cross section to report
    L = ["\n\n---\n\n## 2. 교차실험 전이 검증\n",
         f"**{train_exp} 학습 모델 → {val_exp} 도로 위 실제 사람 주행**과 비교 (검증 윈도 {len(Xv)}).\n",
         "| 지표 | Cross W1 | Cross KS | (참고) Within W1 | 전이 gap |", "|---|---|---|---|---|"]
    for k, t in [("sdlp", "SDLP"), ("mean_speed", "평균속도"), ("speed_std", "속도std")]:
        cw = cross[k]["wasserstein"]; ww = within[k]["wasserstein"] if within else float("nan")
        gap = cw - ww if within else float("nan")
        L.append(f"| {t} | {cw:.4g} | {cross[k]['ks']:.3f} | {ww:.4g} | {gap:.4g} |")
    L.append(f"\n![교차실험 분포](figs/fig_cross_{train_exp}_to_{val_exp}.png)\n")
    L.append("**혼동요인 명시**: 실험 간 차이엔 도로기하뿐 아니라 *렌더링·속도역·실험셋업*이 섞여 있어, "
             "전이 gap이 순수 'geometry→behavior 전이실패'인지 *미모델링 도메인 시프트*인지 분리 불가. "
             "gap 자체를 **전이 한계**라는 발견으로 해석한다.\n")
    rep = os.path.join(REP, "report_generative.md")
    mode = "a" if os.path.exists(rep) else "w"
    open(rep, mode, encoding="utf-8").write("\n".join(L))

    print("cross W1: " + ", ".join(f"{k}={cross[k]['wasserstein']:.3g}" for k in cross), flush=True)
    print(f"appended 교차실험 section -> report_generative.md", flush=True)


if __name__ == "__main__":
    main()
