# -*- coding: utf-8 -*-
"""
11_train_merge.py  --  full-input model on merge/diverge data (신월여의+복층).
Subject(person)-level train/val/test split. Trains point (+gauss) GRU, reports R²/MAE.
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


def load():
    d = np.load(os.path.join(CACHE, "dataset_merge.npz"), allow_pickle=True)
    return (d["X"].astype("float32"), d["Y"].astype("float32"),
            d["run_id"].astype("int64"), d["subject"].astype("int64"),
            d["is_jiha"].astype("int64"), [str(c) for c in d["feat_names"]])


def pa(y, p):
    out = {}
    for j, a in enumerate(ACTIONS):
        t, q = y[:, j], p[:, j]
        ssr = np.sum((t-q)**2); sst = np.sum((t-t.mean())**2)+1e-12
        out[a] = dict(mae=float(np.mean(np.abs(t-q))), rmse=float(np.sqrt(np.mean((t-q)**2))),
                      r2=float(1-ssr/sst))
    return out


def train_loop(model, tl, vl, is_gauss, lossfn, epochs, lr, patience, tag):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, bs, bad, hist = 1e9, None, 0, []
    for ep in range(epochs):
        model.train(); tot=0; n=0
        for xb, yb in tl:
            xb, yb = xb.to(DEV), yb.to(DEV); opt.zero_grad()
            loss = lossfn(*model(xb), yb) if is_gauss else lossfn(model(xb), yb)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); tot += loss.item()*len(xb); n += len(xb)
        sch.step()
        model.eval(); vt=0; vn=0
        with torch.no_grad():
            for xb, yb in vl:
                xb, yb = xb.to(DEV), yb.to(DEV)
                l = lossfn(*model(xb), yb) if is_gauss else lossfn(model(xb), yb)
                vt += l.item()*len(xb); vn += len(xb)
        tl_, vl_ = tot/n, vt/vn; hist.append(dict(epoch=ep, train=tl_, val=vl_))
        if vl_ < best-1e-5: best, bs, bad = vl_, {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}, 0
        else: bad += 1
        if ep%5==0 or ep==epochs-1: print(f"  [{tag}] ep{ep:3d} train={tl_:.4f} val={vl_:.4f} best={best:.4f}", flush=True)
        if bad>=patience: print(f"  [{tag}] early stop @ep{ep}", flush=True); break
    model.load_state_dict(bs); return model, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--models", choices=["both", "point"], default="both")
    args = ap.parse_args()

    X, Y, run_id, subject, is_jiha, feats = load()
    usubj = np.unique(subject); rng = np.random.RandomState(0); rng.shuffle(usubj)
    n = len(usubj); ntest = max(2, n//7); nval = max(2, n//7)
    test_s = set(usubj[:ntest].tolist()); val_s = set(usubj[ntest:ntest+nval].tolist())
    te = np.isin(subject, list(test_s)); va = np.isin(subject, list(val_s)); tr = ~(te|va)
    print(f"subjects total={n} train={n-ntest-nval} val={nval} test={ntest}", flush=True)
    print(f"rows train={tr.sum():,} val={va.sum():,} test={te.sum():,}  feats={len(feats)}", flush=True)

    xs = Scaler.fit(X[tr]); ys = Scaler.fit(Y[tr])
    Xs = xs.transform(X); Ys = ys.transform(Y)
    trd = SeqDataset(Xs, Ys, run_id, tr, L=args.window, stride=args.stride)
    vad = SeqDataset(Xs, Ys, run_id, va, L=args.window, stride=args.stride)
    ted = SeqDataset(Xs, Ys, run_id, te, L=args.window, stride=args.stride)
    print(f"windows train={len(trd):,} val={len(vad):,} test={len(ted):,}", flush=True)
    tl = DataLoader(trd, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=(DEV=="cuda"))
    vl = DataLoader(vad, batch_size=4096, shuffle=False); el = DataLoader(ted, batch_size=4096, shuffle=False)

    rep = dict(feats=feats, rows=dict(train=int(tr.sum()), val=int(va.sum()), test=int(te.sum())),
               windows=dict(train=len(trd), val=len(vad), test=len(ted)),
               n_subjects=int(n), test_subjects=int(ntest), val_subjects=int(nval))
    y_orig = None

    t0 = time.time()
    point = GRUNet(X.shape[1], len(ACTIONS)).to(DEV)
    point, hp = train_loop(point, tl, vl, False, nn.SmoothL1Loss(), args.epochs, 1e-3, 8, "point")
    point.eval(); P, T, JH = [], [], []
    with torch.no_grad():
        for xb, yb in el:
            P.append(point(xb.to(DEV)).cpu().numpy()); T.append(yb.numpy())
    mu_p = ys.inverse(np.concatenate(P)); y_orig = ys.inverse(np.concatenate(T))
    rep["point"] = pa(y_orig, mu_p); rep["hist_point"] = hp
    torch.save(dict(state=point.state_dict(), model="gru_point", feats=feats,
                    x_scaler=xs.to_dict(), y_scaler=ys.to_dict()), os.path.join(ART, "bc_merge_point.pt"))
    print(f"point done {time.time()-t0:.0f}s", flush=True)

    if args.models == "both":
        t1 = time.time()
        g = GRUGaussian(X.shape[1], len(ACTIONS)).to(DEV)
        g, hg = train_loop(g, tl, vl, True, gaussian_nll, args.epochs, 5e-4, 8, "gauss")
        g.eval(); MU, LS = [], []
        with torch.no_grad():
            for xb, yb in el:
                mu, ls = g(xb.to(DEV)); MU.append(mu.cpu().numpy()); LS.append(ls.cpu().numpy())
        mu_g = ys.inverse(np.concatenate(MU)); sig = np.exp(np.concatenate(LS))*ys.std
        rep["gauss"] = pa(y_orig, mu_g); rep["hist_gauss"] = hg
        comp = {}
        for j, a in enumerate(ACTIONS):
            r = y_orig[:, j]-mu_g[:, j]
            comp[a] = dict(sigma_emp=float((y_orig[:, j]-mu_p[:, j]).std()),
                           sigma_pred=float(sig[:, j].mean()),
                           cover1=float(np.mean(np.abs(r)<=sig[:, j])), cover2=float(np.mean(np.abs(r)<=2*sig[:, j])))
        rep["variability"] = comp
        torch.save(dict(state=g.state_dict(), model="gru_gauss", feats=feats,
                        x_scaler=xs.to_dict(), y_scaler=ys.to_dict()), os.path.join(ART, "bc_merge_gauss.pt"))
        np.savez_compressed(os.path.join(CACHE, "preds_merge.npz"),
                            y=y_orig.astype("float32"), mu_point=mu_p.astype("float32"),
                            mu_gauss=mu_g.astype("float32"), sigma_gauss=sig.astype("float32"),
                            ends=ted.ends.astype("int64"), is_jiha=is_jiha[ted.ends].astype("int8"))
        print(f"gauss done {time.time()-t1:.0f}s", flush=True)

    json.dump(rep, open(os.path.join(REP, "metrics_merge.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n=== MERGE full-input (test) ===", flush=True)
    for a in ACTIONS:
        line = f"  {a:9s} point R2={rep['point'][a]['r2']:.3f} MAE={rep['point'][a]['mae']:.4f}"
        if "gauss" in rep: line += f" | gauss R2={rep['gauss'][a]['r2']:.3f}"
        print(line, flush=True)
    print("saved metrics_merge.json", flush=True)


if __name__ == "__main__":
    main()
