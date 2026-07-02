# -*- coding: utf-8 -*-
"""
13_train_cvae.py  --  train the conditional VAE on distance-indexed trajectories.

  python 13_train_cvae.py --smoke
  python 13_train_cvae.py --exp 2024 --z_dim 16 --beta 0.5 --kl_warmup 10

β-VAE with KL warmup (0 -> beta over kl_warmup epochs) to avoid posterior collapse.
Saves: artifacts/cvae_{exp}.pt, reports/metrics_cvae_{exp}.json
"""
import os, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from common import ART, REP, CACHE, build_smoke_dataset, gen_split
from datasets import Scaler
from models import CVAE, cvae_loss

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)


def load_gen(exp):
    p = os.path.join(CACHE, f"dataset_gen_{exp}.npz")
    if not os.path.exists(p) and exp == "smoke":
        build_smoke_dataset(p)
    d = np.load(p, allow_pickle=True)
    return (d["X_geo"].astype("float32"), d["Y_beh"].astype("float32"),
            d["win_subject"].astype("int64"), [str(c) for c in d["feat_geo"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="smoke")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--z_dim", type=int, default=16)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--kl_warmup", type=int, default=10)
    ap.add_argument("--stochastic", action="store_true", help="per-step Gaussian decoder (NLL recon) -> restores SDLP")
    ap.add_argument("--stoch_dim", type=int, default=-1, help="# leading behavior channels stochastic (offset,speed); -1=all, 1=offset only")
    ap.add_argument("--window", type=int, default=0)   # informational (data already windowed)
    ap.add_argument("--stride", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    exp = "smoke" if args.smoke else args.exp
    if args.smoke:                                       # fast self-test, low beta to show z usage
        args.epochs, args.batch, args.kl_warmup, args.beta = 25, 16, 5, 0.02
    print(f"device={DEV} exp={exp} z_dim={args.z_dim} beta={args.beta} warmup={args.kl_warmup}", flush=True)

    Xg, Yb, wsub, feat_geo = load_gen(exp)
    G = Xg.shape[2]
    tr, va, te = gen_split(wsub, seed=0)
    print(f"windows={len(Xg)} geo_dim={G} | train={tr.sum()} val={va.sum()} test={te.sum()}", flush=True)

    gs = Scaler.fit(Xg[tr].reshape(-1, G)); bs = Scaler.fit(Yb[tr].reshape(-1, 2))
    Xs = gs.transform(Xg); Ys = bs.transform(Yb)
    tl = DataLoader(TensorDataset(torch.from_numpy(Ys[tr]), torch.from_numpy(Xs[tr])),
                    batch_size=args.batch, shuffle=True)
    vl = DataLoader(TensorDataset(torch.from_numpy(Ys[va]), torch.from_numpy(Xs[va])),
                    batch_size=512, shuffle=False)

    _sd = None if args.stoch_dim < 0 else args.stoch_dim
    model = CVAE(beh_dim=2, geo_dim=G, z_dim=args.z_dim, stochastic=args.stochastic, stoch_dim=_sd).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    recon_fn = nn.SmoothL1Loss()
    best, best_state, bad, hist = 1e9, None, 0, []

    for ep in range(args.epochs):
        beta_t = args.beta * min(1.0, (ep + 1) / max(1, args.kl_warmup))
        model.train(); tot = rec_s = kl_s = n = 0
        for beh, geo in tl:
            beh, geo = beh.to(DEV), geo.to(DEV)
            opt.zero_grad()
            rec_mu, rec_logstd, mu_z, logvar = model(beh, geo)
            loss, rec, kl = cvae_loss(rec_mu, rec_logstd, beh, mu_z, logvar, beta_t, recon_fn)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            tot += loss.item() * len(beh); rec_s += rec.item() * len(beh); kl_s += kl.item() * len(beh); n += len(beh)
        sch.step()
        # val (use final beta for comparability)
        model.eval(); vtot = vrec = vkl = vn = 0
        with torch.no_grad():
            for beh, geo in vl:
                beh, geo = beh.to(DEV), geo.to(DEV)
                rec_mu, rec_logstd, mu_z, logvar = model(beh, geo)
                loss, rec, kl = cvae_loss(rec_mu, rec_logstd, beh, mu_z, logvar, args.beta, recon_fn)
                vtot += loss.item() * len(beh); vrec += rec.item() * len(beh); vkl += kl.item() * len(beh); vn += len(beh)
        hist.append(dict(epoch=ep, beta=beta_t, train=tot / n, train_recon=rec_s / n, train_kl=kl_s / n,
                         val=vtot / vn, val_recon=vrec / vn, val_kl=vkl / vn))
        if vtot / vn < best - 1e-6:
            best, best_state, bad = vtot / vn, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  ep{ep:3d} beta={beta_t:.3f} val={vtot/vn:.4f} recon={vrec/vn:.4f} KL={vkl/vn:.3f}", flush=True)
        if bad >= 8:
            print(f"  early stop @ep{ep}", flush=True); break

    model.load_state_dict(best_state)
    final_kl = hist[-1]["val_kl"]
    collapse = final_kl < 0.05
    torch.save(dict(state=model.state_dict(), exp=exp, z_dim=args.z_dim, geo_dim=G, beh_dim=2,
                    stochastic=args.stochastic, stoch_dim=model.stoch_dim,
                    feat_geo=feat_geo, geo_scaler=gs.to_dict(), beh_scaler=bs.to_dict(),
                    args=vars(args)), os.path.join(ART, f"cvae_{exp}.pt"))
    json.dump(dict(exp=exp, z_dim=args.z_dim, beta=args.beta, geo_dim=G, feat_geo=feat_geo,
                   windows=dict(train=int(tr.sum()), val=int(va.sum()), test=int(te.sum())),
                   final_val_KL=final_kl, posterior_collapse=bool(collapse), history=hist),
              open(os.path.join(REP, f"metrics_cvae_{exp}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\nDONE  final val KL={final_kl:.3f}  posterior_collapse={collapse} "
          f"(KL>0 이면 z가 쓰임)\nsaved -> artifacts/cvae_{exp}.pt", flush=True)


if __name__ == "__main__":
    main()
