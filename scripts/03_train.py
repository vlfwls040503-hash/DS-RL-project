# -*- coding: utf-8 -*-
"""
03_train.py  --  train a behavioral-cloning model (full driving control).

  python 03_train.py --model mlp
  python 03_train.py --model gru --window 24 --stride 2

Saves:
  artifacts/bc_{model}.pt           best checkpoint (weights + scalers + config)
  reports/metrics_{model}.json      history + final train/val/test metrics
  cache/preds_{model}.npz           test-set y_true / y_pred (original units)
"""
import os, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import ART, REP, CACHE, ACTIONS, FEATURES, VAL_SUBJECTS, TEST_SUBJECTS
from datasets import load_cache, split_masks, Scaler, FrameDataset, SeqDataset
from models import MLP, GRUNet

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)


def per_action_metrics(y_true, y_pred):
    """y_* in ORIGINAL units, shape (N,3). Returns dict per action."""
    out = {}
    for j, a in enumerate(ACTIONS):
        t, p = y_true[:, j], y_pred[:, j]
        mae = float(np.mean(np.abs(t - p)))
        rmse = float(np.sqrt(np.mean((t - p) ** 2)))
        ss_res = float(np.sum((t - p) ** 2))
        ss_tot = float(np.sum((t - t.mean()) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        out[a] = dict(mae=mae, rmse=rmse, r2=r2)
    return out


# ----------------------------- MLP path (GPU-resident) -----------------------------
def train_mlp(X, Y, tr, va, te, xs, ys, args):
    Xt = torch.from_numpy(xs.transform(X)).to(DEV)
    Yt = torch.from_numpy(ys.transform(Y)).to(DEV)
    tr_i = torch.from_numpy(np.where(tr)[0]).to(DEV)
    va_i = torch.from_numpy(np.where(va)[0]).to(DEV)
    te_i = torch.from_numpy(np.where(te)[0]).to(DEV)

    model = MLP(X.shape[1], len(ACTIONS), p=args.dropout).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.SmoothL1Loss()

    def eval_idx(idx):
        model.eval()
        with torch.no_grad():
            preds = []
            for s in range(0, len(idx), 65536):
                b = idx[s:s + 65536]
                preds.append(model(Xt[b]))
            p = torch.cat(preds)
            l = lossf(p, Yt[idx]).item()
            # original units
            p_o = ys.inverse(p.cpu().numpy())
            t_o = Y[idx.cpu().numpy()]
        return l, p_o, t_o

    best_val, best_state, bad = 1e9, None, 0
    hist = []
    nb = args.batch
    for ep in range(args.epochs):
        model.train()
        perm = tr_i[torch.randperm(len(tr_i), device=DEV)]
        tot = 0.0
        for s in range(0, len(perm), nb):
            b = perm[s:s + nb]
            opt.zero_grad()
            out = model(Xt[b])
            loss = lossf(out, Yt[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        tr_loss = tot / len(perm)
        val_loss, _, _ = eval_idx(va_i)
        hist.append(dict(epoch=ep, train=tr_loss, val=val_loss))
        if val_loss < best_val - 1e-5:
            best_val, best_state, bad = val_loss, {k: v.detach().cpu().clone()
                                                   for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  ep{ep:3d} train={tr_loss:.4f} val={val_loss:.4f} best={best_val:.4f}")
        if bad >= args.patience:
            print(f"  early stop @ep{ep}")
            break

    model.load_state_dict(best_state)
    val_loss, vp, vt = eval_idx(va_i)
    te_loss, tp, tt = eval_idx(te_i)
    return model, hist, dict(val=per_action_metrics(vt, vp), test=per_action_metrics(tt, tp)), (tt, tp, te_i.cpu().numpy())


# ----------------------------- GRU path (windowed) -----------------------------
def train_gru(X, Y, run_id, tr, va, te, xs, ys, args):
    Xs = xs.transform(X); Ys = ys.transform(Y)
    tr_ds = SeqDataset(Xs, Ys, run_id, tr, L=args.window, stride=args.stride)
    va_ds = SeqDataset(Xs, Ys, run_id, va, L=args.window, stride=args.stride)
    te_ds = SeqDataset(Xs, Ys, run_id, te, L=args.window, stride=args.stride)
    print(f"  windows: train={len(tr_ds):,} val={len(va_ds):,} test={len(te_ds):,}")
    tl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=(DEV == "cuda"))
    vl = DataLoader(va_ds, batch_size=4096, shuffle=False, num_workers=0)
    el = DataLoader(te_ds, batch_size=4096, shuffle=False, num_workers=0)

    model = GRUNet(X.shape[1], len(ACTIONS), p=args.dropout).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.SmoothL1Loss()

    def eval_loader(loader):
        model.eval(); ps, ts = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(DEV)
                p = model(xb).cpu().numpy()
                ps.append(p); ts.append(yb.numpy())
        p = np.concatenate(ps); t = np.concatenate(ts)
        l = float(np.mean(np.abs(p - t)))
        return l, ys.inverse(p), ys.inverse(t)

    best_val, best_state, bad, hist = 1e9, None, 0, []
    for ep in range(args.epochs):
        model.train(); tot = 0.0; nseen = 0
        for xb, yb in tl:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(xb); nseen += len(xb)
        sched.step()
        tr_loss = tot / max(nseen, 1)
        val_loss, _, _ = eval_loader(vl)
        hist.append(dict(epoch=ep, train=tr_loss, val=val_loss))
        if val_loss < best_val - 1e-5:
            best_val, best_state, bad = val_loss, {k: v.detach().cpu().clone()
                                                   for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        print(f"  ep{ep:3d} train={tr_loss:.4f} val={val_loss:.4f} best={best_val:.4f}")
        if bad >= args.patience:
            print(f"  early stop @ep{ep}"); break

    model.load_state_dict(best_state)
    _, vp, vt = eval_loader(vl)
    _, tp, tt = eval_loader(el)
    return model, hist, dict(val=per_action_metrics(vt, vp), test=per_action_metrics(tt, tp)), (tt, tp, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mlp", "gru"], default="mlp")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()
    print(f"device={DEV}  model={args.model}")

    X, Y, run_id, subject, scenario = load_cache()
    tr, va, te = split_masks(subject)
    print(f"rows train={tr.sum():,} val={va.sum():,} test={te.sum():,}")
    print(f"  val subjects={VAL_SUBJECTS} test subjects={TEST_SUBJECTS}")

    xs = Scaler.fit(X[tr]); ys = Scaler.fit(Y[tr])

    t0 = time.time()
    if args.model == "mlp":
        model, hist, metrics, (tt, tp, te_idx) = train_mlp(X, Y, tr, va, te, xs, ys, args)
    else:
        model, hist, metrics, (tt, tp, te_idx) = train_gru(X, Y, run_id, tr, va, te, xs, ys, args)
    dt = time.time() - t0

    # naive baselines on test (original units): predict train mean
    base = {}
    tr_mean = Y[tr].mean(axis=0)
    base["predict_train_mean"] = per_action_metrics(tt, np.tile(tr_mean, (len(tt), 1)))

    ckpt = dict(state=model.state_dict(), feat_names=FEATURES, act_names=ACTIONS,
                x_scaler=xs.to_dict(), y_scaler=ys.to_dict(),
                model=args.model, args=vars(args))
    torch.save(ckpt, os.path.join(ART, f"bc_{args.model}.pt"))

    report = dict(model=args.model, device=DEV, seconds=dt, args=vars(args),
                  val_subjects=VAL_SUBJECTS, test_subjects=TEST_SUBJECTS,
                  rows=dict(train=int(tr.sum()), val=int(va.sum()), test=int(te.sum())),
                  metrics=metrics, baseline=base, history=hist)
    with open(os.path.join(REP, f"metrics_{args.model}.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    # save test predictions for plotting
    save = dict(y_true=tt.astype("float32"), y_pred=tp.astype("float32"))
    if te_idx is not None:
        save["subject"] = subject[te_idx].astype("int16")
        save["scenario"] = scenario[te_idx].astype("int8")
    np.savez_compressed(os.path.join(CACHE, f"preds_{args.model}.npz"), **save)

    print(f"\nDONE {args.model} in {dt:.1f}s")
    print("TEST metrics (original units):")
    for a in ACTIONS:
        m = metrics["test"][a]; b = base["predict_train_mean"][a]
        print(f"  {a:9s} MAE={m['mae']:.5f} RMSE={m['rmse']:.5f} R2={m['r2']:.3f}  "
              f"(baseline MAE={b['mae']:.5f})")


if __name__ == "__main__":
    main()
