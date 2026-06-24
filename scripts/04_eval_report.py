# -*- coding: utf-8 -*-
"""
04_eval_report.py  --  evaluate trained BC checkpoints, make figures, write report.
Loads bc_mlp.pt / bc_gru.pt (whichever exist), runs inference on the held-out
test subjects, computes metrics, and renders plots + a Korean markdown report.
"""
import os, json, glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ART, REP, ACTIONS, FEATURES, VAL_SUBJECTS, TEST_SUBJECTS
from datasets import load_cache, split_masks, Scaler, SeqDataset
from models import MLP, GRUNet

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FIG = os.path.join(REP, "figs"); os.makedirs(FIG, exist_ok=True)

# Korean-capable font
for fam in ["Malgun Gothic", "NanumGothic", "Gulim"]:
    try:
        plt.rcParams["font.family"] = fam
        break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False
ACT_KO = {"steering": "조향", "throttle": "가속", "brake": "브레이크"}


def per_action_metrics(y_true, y_pred):
    out = {}
    for j, a in enumerate(ACTIONS):
        t, p = y_true[:, j], y_pred[:, j]
        ss_res = np.sum((t - p) ** 2); ss_tot = np.sum((t - t.mean()) ** 2) + 1e-12
        out[a] = dict(mae=float(np.mean(np.abs(t - p))),
                      rmse=float(np.sqrt(np.mean((t - p) ** 2))),
                      r2=float(1 - ss_res / ss_tot))
    return out


def load_model(ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEV, weights_only=False)
    xs = Scaler.from_dict(ck["x_scaler"]); ys = Scaler.from_dict(ck["y_scaler"])
    if ck["model"] == "mlp":
        m = MLP(len(FEATURES), len(ACTIONS))
    else:
        m = GRUNet(len(FEATURES), len(ACTIONS))
    m.load_state_dict(ck["state"]); m.to(DEV).eval()
    return m, xs, ys, ck


def infer_mlp(m, xs, ys, X, idx):
    Xs = torch.from_numpy(xs.transform(X[idx])).to(DEV)
    preds = []
    with torch.no_grad():
        for s in range(0, len(Xs), 65536):
            preds.append(m(Xs[s:s + 65536]).cpu().numpy())
    return ys.inverse(np.concatenate(preds)), idx  # pred (N,3), aligned to idx


def infer_gru(m, xs, ys, X, Y, run_id, mask, window, stride):
    Xs = xs.transform(X); Ys = ys.transform(Y)
    ds = SeqDataset(Xs, Ys, run_id, mask, L=window, stride=stride)
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=4096, shuffle=False, num_workers=0)
    preds = []
    with torch.no_grad():
        for xb, _ in dl:
            preds.append(m(xb.to(DEV)).cpu().numpy())
    p = ys.inverse(np.concatenate(preds))
    return p, ds.ends  # pred aligned to ds.ends (global indices)


def main():
    X, Y, run_id, subject, scenario = load_cache()
    tr, va, te = split_masks(subject)
    te_idx = np.where(te)[0]

    results = {}    # model -> dict(pred, idx, metrics, per_scen)
    for ck_path in sorted(glob.glob(os.path.join(ART, "bc_*.pt"))):
        name = os.path.basename(ck_path)[3:-3]  # bc_xxx.pt -> xxx
        m, xs, ys, ck = load_model(ck_path)
        if ck["model"] == "mlp":
            pred, idx = infer_mlp(m, xs, ys, X, te_idx)
        else:
            w = ck["args"]["window"]; s = ck["args"]["stride"]
            pred, idx = infer_gru(m, xs, ys, X, Y, run_id, te, w, s)
        ytrue = Y[idx]
        met = per_action_metrics(ytrue, pred)
        # per scenario
        scen = scenario[idx]; per_scen = {}
        for sc in [1, 2, 3, 4]:
            mlt = scen == sc
            if mlt.sum() > 0:
                per_scen[sc] = per_action_metrics(ytrue[mlt], pred[mlt])
        results[name] = dict(pred=pred, idx=idx, ytrue=ytrue, met=met, per_scen=per_scen,
                             run_id=run_id[idx], scen=scen)
        print(f"[{name}] test metrics:")
        for a in ACTIONS:
            print(f"   {a:9s} MAE={met[a]['mae']:.5f} RMSE={met[a]['rmse']:.5f} R2={met[a]['r2']:.3f}")

    # ---------------- Figures ----------------
    # 1) training curves
    plt.figure(figsize=(7, 4))
    for name in results:
        mp = os.path.join(REP, f"metrics_{name}.json")
        if os.path.exists(mp):
            h = json.load(open(mp, encoding="utf-8"))["history"]
            ep = [x["epoch"] for x in h]
            plt.plot(ep, [x["train"] for x in h], "--", label=f"{name} train")
            plt.plot(ep, [x["val"] for x in h], "-", label=f"{name} val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("학습 곡선"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fig_train_curves.png"), dpi=120); plt.close()

    # pick best model by mean test R2 over actions
    best = max(results, key=lambda n: np.mean([results[n]["met"][a]["r2"] for a in ACTIONS]))
    R = results[best]

    # 2) action distributions true vs pred
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for j, a in enumerate(ACTIONS):
        ax = axes[j]
        t, p = R["ytrue"][:, j], R["pred"][:, j]
        lo, hi = np.percentile(np.concatenate([t, p]), [0.5, 99.5])
        bins = np.linspace(lo, hi, 60)
        ax.hist(t, bins=bins, alpha=.5, label="실제", density=True)
        ax.hist(p, bins=bins, alpha=.5, label="예측", density=True)
        ax.set_title(f"{ACT_KO[a]} 분포"); ax.legend()
    fig.suptitle(f"행동 분포 비교 (모델: {best})"); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_action_hist.png"), dpi=120); plt.close(fig)

    # 3) pred vs true hexbin
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for j, a in enumerate(ACTIONS):
        ax = axes[j]
        t, p = R["ytrue"][:, j], R["pred"][:, j]
        ax.hexbin(t, p, gridsize=50, mincnt=1, cmap="viridis", bins="log")
        lo, hi = np.percentile(t, [0.5, 99.5])
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel("실제"); ax.set_ylabel("예측")
        ax.set_title(f"{ACT_KO[a]}  R²={R['met'][a]['r2']:.3f}")
    fig.suptitle(f"예측 vs 실제 ({best})"); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_scatter.png"), dpi=120); plt.close(fig)

    # 4) per-scenario MAE
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for j, a in enumerate(ACTIONS):
        ax = axes[j]
        scs = sorted(R["per_scen"].keys())
        vals = [R["per_scen"][sc][a]["mae"] for sc in scs]
        ax.bar([f"S{sc}" for sc in scs], vals, color="#4C72B0")
        ax.set_title(f"{ACT_KO[a]} 시나리오별 MAE")
    fig.suptitle(f"시나리오(격벽조건)별 오차 ({best})"); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_per_scenario_mae.png"), dpi=120); plt.close(fig)

    # 5) time-series for one representative test run (most braking activity)
    # choose run with highest brake std among test runs
    runs = np.unique(R["run_id"])
    brake_std = {r: R["ytrue"][R["run_id"] == r][:, 2].std() for r in runs}
    pick = max(brake_std, key=brake_std.get)
    sel = R["run_id"] == pick
    order = np.argsort(R["idx"][sel])
    tt = R["ytrue"][sel][order]; pp = R["pred"][sel][order]
    # plot a 1500-frame slice (~34s) for readability
    sl = slice(0, min(1500, len(tt)))
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for j, a in enumerate(ACTIONS):
        ax = axes[j]
        ax.plot(tt[sl, j], label="실제", lw=1.2)
        ax.plot(pp[sl, j], label="예측", lw=1.0, alpha=.8)
        ax.set_ylabel(ACT_KO[a]); ax.legend(loc="upper right")
    axes[-1].set_xlabel("프레임 (~44Hz)")
    fig.suptitle(f"테스트 주행 1개 구간: 실제 vs 예측 ({best}, run#{int(pick)})")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_timeseries.png"), dpi=120); plt.close(fig)

    # ---------------- consolidated metrics json ----------------
    summ = {name: dict(test=results[name]["met"], per_scenario=results[name]["per_scen"])
            for name in results}
    summ["_best_model"] = best
    json.dump(summ, open(os.path.join(REP, "eval_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("best model:", best)
    print("figures ->", FIG)
    return results, best


if __name__ == "__main__":
    main()
