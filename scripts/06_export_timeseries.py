# -*- coding: utf-8 -*-
"""Export one test run's time-ordered actual-vs-predicted (GRU) to JSON for the widget."""
import os, json
import numpy as np
import torch
from common import ART, REP, ACTIONS
from datasets import load_cache, split_masks, Scaler, SeqDataset
from models import GRUNet
from torch.utils.data import DataLoader

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    X, Y, run_id, subject, scenario = load_cache()
    tr, va, te = split_masks(subject)

    ck = torch.load(os.path.join(ART, "bc_gru.pt"), map_location=DEV, weights_only=False)
    xs = Scaler.from_dict(ck["x_scaler"]); ys = Scaler.from_dict(ck["y_scaler"])
    w = ck["args"]["window"]; s = ck["args"]["stride"]
    m = GRUNet(X.shape[1], len(ACTIONS)); m.load_state_dict(ck["state"]); m.to(DEV).eval()

    # choose the test run with the most braking activity (so all 3 actions are interesting)
    test_runs = np.unique(run_id[te])
    brake_sum = {int(r): float(Y[run_id == r][:, 2].sum()) for r in test_runs}
    pick = max(brake_sum, key=brake_sum.get)
    subj = int(subject[run_id == pick][0]); scen = int(scenario[run_id == pick][0])
    print(f"picked run#{pick} subject={subj} scenario=S{scen}  brakeSum={brake_sum[pick]:.1f}")

    mask = (run_id == pick)
    Xs = xs.transform(X); Ys = ys.transform(Y)
    ds = SeqDataset(Xs, Ys, run_id, mask, L=w, stride=s)
    dl = DataLoader(ds, batch_size=4096, shuffle=False, num_workers=0)
    preds = []
    with torch.no_grad():
        for xb, _ in dl:
            preds.append(m(xb.to(DEV)).cpu().numpy())
    pred = ys.inverse(np.concatenate(preds))         # (M,3) aligned to ds.ends
    ends = ds.ends
    order = np.argsort(ends)
    ends, pred = ends[order], pred[order]
    true = Y[ends]

    # downsample to ~500 points, take a ~30s window from the start of predictions
    hz = 43.7
    max_pts = 500
    n = len(ends)
    step = max(1, n // max_pts)
    idx = np.arange(0, n, step)[:max_pts]
    t_sec = (np.arange(n) / hz)[idx]

    out = dict(run=int(pick), subject=subj, scenario=f"S{scen}", hz=hz,
               t=[round(float(x), 2) for x in t_sec])
    for j, a in enumerate(ACTIONS):
        out[a] = dict(true=[round(float(v), 4) for v in true[idx, j]],
                      pred=[round(float(v), 4) for v in pred[idx, j]])
    json.dump(out, open(os.path.join(REP, "timeseries_run.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    print("wrote reports/timeseries_run.json  points:", len(idx))
    # quick correlation per action on the full run for context
    for j, a in enumerate(ACTIONS):
        r = np.corrcoef(true[:, j], pred[:, j])[0, 1]
        print(f"  {a:9s} corr={r:.3f}  range true [{true[:,j].min():.3f},{true[:,j].max():.3f}]")


if __name__ == "__main__":
    main()
