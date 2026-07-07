# -*- coding: utf-8 -*-
"""
16_build_env_roads.py  --  build road profiles + human refs for the RL env.

  python 16_build_env_roads.py --smoke
  python 16_build_env_roads.py --exp 2024

Reuses 12_build_gen_dataset's exp resolvers + common.reindex_run.
Output: cache/env_roads_{exp}.npz  (concatenated channels + ptr; see driving_env.save_roads)
"""
import os, argparse, importlib
import numpy as np
from common import CACHE, GEN_DD, GEN_GEO_CANDIDATES, reindex_run, make_smoke_roads
from driving_env import save_roads

MIN_GRID_PTS = 200          # skip runs shorter than ~400 m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["namsan", "2024", "merge", "wangsuk", "icing", "underpass21", "smoke"], default="smoke")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    exp = "smoke" if args.smoke else args.exp
    out = args.out or os.path.join(CACHE, f"env_roads_{exp}.npz")

    if exp == "smoke":
        roads = make_smoke_roads(n=8, seed=0)
        save_roads(out, roads)
        print(f"[smoke] roads={len(roads)} subjects={len(set(r['subject'] for r in roads))}")
        print("saved ->", out)
        return

    b12 = importlib.import_module("12_build_gen_dataset")
    geo_cols = list(GEN_GEO_CANDIDATES)   # [laneCurvature, longSlope, latSlope, laneWidth, cwWidth]
    roads, skipped = [], 0
    for df, subj, rid, cond in b12.RESOLVERS[exp](geo_cols):
        uv = df[df["type"] == "uv"] if "type" in df.columns else df
        r = reindex_run(uv, geo_cols, GEN_DD)
        if r is None:
            skipped += 1; continue
        beh, G = r
        if len(beh) < MIN_GRID_PTS:
            skipped += 1; continue
        roads.append(dict(curv=G[:, 0], slope=G[:, 1], lane_w=G[:, 3], cw=G[:, 4],
                          e_ref=beh[:, 0], v_ref=beh[:, 1], subject=subj, cond=cond))
        if len(roads) % 20 == 0:
            print(f"  {len(roads)} roads...", flush=True)
    save_roads(out, roads)
    lens = [len(r["curv"]) for r in roads]
    print(f"roads={len(roads)} (skipped {skipped})  subjects={len(set(r['subject'] for r in roads))}")
    print(f"length: median {np.median(lens)*GEN_DD/1000:.1f} km, "
          f"min {min(lens)*GEN_DD/1000:.1f} / max {max(lens)*GEN_DD/1000:.1f} km")
    print("saved ->", out)


if __name__ == "__main__":
    main()
