# -*- coding: utf-8 -*-
"""
08_train_2024.py  --  PoC + ablation on 2024 underground dataset.

Trains a sequence policy under a chosen feature set and head:
  --feat_mode all | geo | ego
      all : every feature (road + ego-state + condition)
      geo : exogenous road geometry + 지상/지하 condition ONLY  (genuine geometry->behavior)
      ego : self-state ONLY (speed/accel/yaw/offset...) -- exposes "self-state continuity"
  --models both | point | gauss
      point : deterministic GRU (SmoothL1)
      gauss : probabilistic GRU (mean+std, NLL)

Outputs are suffixed by feat_mode:
  artifacts/bc2024_{mode}_point.pt / _gauss.pt
  reports/metrics_2024_{mode}.json
  cache/preds_2024_{mode}.npz   (only if gauss trained)
"""
import os, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import ART, REP, CACHE, ACTIONS
from datasets import Scaler, SeqDataset
from models import GRUNet, GRUGaussian, gaussian_nll

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)

TEST_SUBJ = [4, 8, 13, 19, 25, 30]
VAL_SUBJ = [2, 11, 17, 23, 28, 32]

GEO_FEATS = ["laneCurvature", "carriagewayWidth", "laneWidth",
             "roadLongitudinalSlope", "roadLateralSlope", "cond_ground", "cond_under"]
EGO_FEATS = ["speedInKmPerHour", "localAccelInMetresPerSecond2 X", "localAccelInMetresPerSecond2 Y",
             "bodyRotSpeedInRadsPerSecond Yaw", "offsetFromLaneCenter", "offsetFromRoadCenter",
             "distanceToLeftBorder", "distanceToRightBorder", "leftLaneOverLap", "rightLaneOverLap"]


def load():
    d = np.load(os.path.join(CACHE, "dataset_2024.npz"), allow_pickle=True)
    return (d["X"].astype("float32"), d["Y"].astype("float32"),
            d["run_id"].astype("int64"), d["subject"].astype("int64"),
            d["cond"].astype("int64"), [str(c) for c in d["feat_names"]])


def pa_metrics(y, p):
    out = {}
    for j, a in enumerate(ACTIONS):
        t, q = y[:, j], p[:, j]
        ssr = np.sum((t - q) ** 2); sst = np.sum((t - t.mean()) ** 2) + 1e-12
        out[a] = dict(mae=float(np.mean(np.abs(t - q))), rmse=float(np.sqrt(np.mean((t - q) ** 2))),
                      r2=float(1 - ssr / sst))
    return out


def make_loaders(Xs, Ys, run_id, tr, va, te, L, stride, batch):
    tr_ds = SeqDataset(Xs, Ys, run_id, tr, L=L, stride=stride)
    va_ds = SeqDataset(Xs, Ys, run_id, va, L=L, stride=stride)
    te_ds = SeqDataset(Xs, Ys, run_id, te, L=L, stride=stride)
    tl = DataLoader(tr_ds, batch_size=batch, shuffle=True, num_workers=0, pin_memory=(DEV == "cuda"))
    vl = DataLoader(va_ds, batch_size=4096, shuffle=False, num_workers=0)
    el = DataLoader(te_ds, batch_size=4096, shuffle=False, num_workers=0)
    return (tr_ds, va_ds, te_ds), (tl, vl, el)


def train_loop(model, tl, vl, is_gauss, lossfn, epochs, lr, patience, tag):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_state, bad, hist = 1e9, None, 0, []
    for ep in range(epochs):
        model.train(); tot = 0.0; ns = 0
        for xb, yb in tl:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad()
            if is_gauss:
                mu, ls = model(xb); loss = lossfn(mu, ls, yb)
            else:
                loss = lossfn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); tot += loss.item() * len(xb); ns += len(xb)
        sched.step()
        model.eval(); vt = 0.0; vn = 0
        with torch.no_grad():
            for xb, yb in vl:
                xb, yb = xb.to(DEV), yb.to(DEV)
                if is_gauss:
                    mu, ls = model(xb); l = lossfn(mu, ls, yb)
                else:
                    l = lossfn(model(xb), yb)
                vt += l.item() * len(xb); vn += len(xb)
        tr_l, va_l = tot / ns, vt / vn
        hist.append(dict(epoch=ep, train=tr_l, val=va_l))
        if va_l < best - 1e-5:
            best, best_state, bad = va_l, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        print(f"  [{tag}] ep{ep:3d} train={tr_l:.4f} val={va_l:.4f} best={best:.4f}", flush=True)
        if bad >= patience:
            print(f"  [{tag}] early stop @ep{ep}", flush=True); break
    model.load_state_dict(best_state)
    return model, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_mode", choices=["all", "geo", "ego"], default="all")
    ap.add_argument("--models", choices=["both", "point", "gauss"], default="both")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    args = ap.parse_args()
    print(f"device={DEV} feat_mode={args.feat_mode} models={args.models} batch={args.batch}", flush=True)

    X, Y, run_id, subject, cond, feats = load()
    if args.feat_mode == "geo":
        chosen = [c for c in GEO_FEATS if c in feats]
    elif args.feat_mode == "ego":
        chosen = [c for c in EGO_FEATS if c in feats]
    else:
        chosen = feats
    cidx = [feats.index(c) for c in chosen]
    X = X[:, cidx]
    print(f"features ({len(chosen)}): {chosen}", flush=True)

    te = np.isin(subject, TEST_SUBJ); va = np.isin(subject, VAL_SUBJ); tr = ~(te | va)
    print(f"rows train={tr.sum():,} val={va.sum():,} test={te.sum():,}", flush=True)
    xs = Scaler.fit(X[tr]); ys = Scaler.fit(Y[tr])
    Xs = xs.transform(X); Ys = ys.transform(Y)
    (tr_ds, va_ds, te_ds), (tl, vl, el) = make_loaders(Xs, Ys, run_id, tr, va, te,
                                                       args.window, args.stride, args.batch)
    print(f"windows train={len(tr_ds):,} val={len(va_ds):,} test={len(te_ds):,}", flush=True)

    y_orig = None
    met_point = met_gauss = comp = None
    mu_point = mu_gauss = sigma_gauss = None
    sigma_emp = None
    hist_p = hist_g = None
    suf = args.feat_mode

    if args.models in ("both", "point"):
        t0 = time.time()
        point = GRUNet(X.shape[1], len(ACTIONS)).to(DEV)
        point, hist_p = train_loop(point, tl, vl, False, nn.SmoothL1Loss(),
                                   args.epochs, args.lr, args.patience, f"point/{suf}")
        point.eval(); P, T = [], []
        with torch.no_grad():
            for xb, yb in el:
                P.append(point(xb.to(DEV)).cpu().numpy()); T.append(yb.numpy())
        mu_point = ys.inverse(np.concatenate(P)); y_orig = ys.inverse(np.concatenate(T))
        met_point = pa_metrics(y_orig, mu_point)
        sigma_emp = (y_orig - mu_point).std(axis=0)
        torch.save(dict(state=point.state_dict(), model="gru_point", feats=chosen,
                        x_scaler=xs.to_dict(), y_scaler=ys.to_dict(), args=vars(args)),
                   os.path.join(ART, f"bc2024_{suf}_point.pt"))
        print(f"point done {time.time()-t0:.0f}s", flush=True)

    if args.models in ("both", "gauss"):
        t1 = time.time()
        gauss = GRUGaussian(X.shape[1], len(ACTIONS)).to(DEV)
        gauss, hist_g = train_loop(gauss, tl, vl, True, gaussian_nll,
                                   args.epochs, args.lr, args.patience, f"gauss/{suf}")
        gauss.eval(); MU, LS, T = [], [], []
        with torch.no_grad():
            for xb, yb in el:
                mu, ls = gauss(xb.to(DEV)); MU.append(mu.cpu().numpy()); LS.append(ls.cpu().numpy()); T.append(yb.numpy())
        mu_gauss = ys.inverse(np.concatenate(MU))
        sigma_gauss = np.exp(np.concatenate(LS)) * ys.std
        if y_orig is None:
            y_orig = ys.inverse(np.concatenate(T))
        met_gauss = pa_metrics(y_orig, mu_gauss)
        torch.save(dict(state=gauss.state_dict(), model="gru_gauss", feats=chosen,
                        x_scaler=xs.to_dict(), y_scaler=ys.to_dict(), args=vars(args)),
                   os.path.join(ART, f"bc2024_{suf}_gauss.pt"))
        print(f"gauss done {time.time()-t1:.0f}s", flush=True)

    if met_point is not None and met_gauss is not None:
        comp = {}
        for j, a in enumerate(ACTIONS):
            resid = y_orig[:, j] - mu_gauss[:, j]
            comp[a] = dict(sigma_emp=float(sigma_emp[j]),
                           sigma_pred_mean=float(sigma_gauss[:, j].mean()),
                           cover_1sigma=float(np.mean(np.abs(resid) <= sigma_gauss[:, j])),
                           cover_2sigma=float(np.mean(np.abs(resid) <= 2 * sigma_gauss[:, j])))

    out_path = os.path.join(REP, f"metrics_2024_{suf}.json")
    rep = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else {}
    rep.update(feat_mode=args.feat_mode, features=chosen, n_features=len(chosen),
               rows=dict(train=int(tr.sum()), val=int(va.sum()), test=int(te.sum())),
               windows=dict(train=len(tr_ds), val=len(va_ds), test=len(te_ds)),
               test_subjects=TEST_SUBJ, val_subjects=VAL_SUBJ)
    if met_point is not None:
        rep["point"] = met_point; rep["hist_point"] = hist_p
    if met_gauss is not None:
        rep["gauss"] = met_gauss; rep["hist_gauss"] = hist_g
    if comp is not None:
        rep["variability"] = comp
    json.dump(rep, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    if mu_gauss is not None:
        np.savez_compressed(os.path.join(CACHE, f"preds_2024_{suf}.npz"),
                            y=y_orig.astype("float32"), mu_gauss=mu_gauss.astype("float32"),
                            mu_point=(mu_point.astype("float32") if mu_point is not None else mu_gauss.astype("float32")),
                            sigma_gauss=sigma_gauss.astype("float32"),
                            ends=te_ds.ends.astype("int64"),
                            cond=cond[te_ds.ends].astype("int8"), subject=subject[te_ds.ends].astype("int16"))

    print(f"\n=== feat_mode={args.feat_mode}  R2 (test) ===", flush=True)
    for a in ACTIONS:
        pp = f"{met_point[a]['r2']:.3f}" if met_point else "-"
        gg = f"{met_gauss[a]['r2']:.3f}" if met_gauss else "-"
        print(f"  {a:9s} point={pp} gauss={gg}", flush=True)
    if comp:
        print("=== variability ===", flush=True)
        for a in ACTIONS:
            c = comp[a]
            print(f"  {a:9s} sigma_emp={c['sigma_emp']:.4f} pred={c['sigma_pred_mean']:.4f} "
                  f"cov1={c['cover_1sigma']:.2f} cov2={c['cover_2sigma']:.2f}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
