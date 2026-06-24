# -*- coding: utf-8 -*-
"""Deep profile of the 2024 long-distance underground dataset (PoC base)."""
import os, glob, re
import numpy as np
import pandas as pd

D = r"<NAS_PATH set your own>"


def hdr(path):
    return list(pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns)


def main():
    g_sang = glob.glob(os.path.join(D, "피실험자1_지상주행.csv"))[0]
    g_ha = glob.glob(os.path.join(D, "피실험자1_지하주행.csv"))[0]
    h1, h2 = hdr(g_sang), hdr(g_ha)
    print("지상 cols:", len(h1), "  지하 cols:", len(h2), "  identical:", h1 == h2)

    print("\n=== FULL COLUMN LIST (지하) ===")
    for i, c in enumerate(h2):
        print(f"{i:3d}  {c}")

    print("\n=== sample stats from 지하주행 피실험자1 (first 20000 rows) ===")
    df = pd.read_csv(g_ha, nrows=20000, encoding="utf-8-sig", low_memory=False)
    if "type" in df.columns:
        print("type values:", df["type"].value_counts(dropna=False).to_dict())
        df = df[df["type"] == "uv"]
    if "drivingMode" in df.columns:
        print("drivingMode:", df["drivingMode"].value_counts(dropna=False).to_dict())
    if "automaticControl" in df.columns:
        print("automaticControl:", df["automaticControl"].value_counts(dropna=False).to_dict())
    # sampling
    for tcol in ["scenarioTime", "scenario.Time", "time", "X...time."]:
        if tcol in df.columns:
            t = pd.to_numeric(df[tcol], errors="coerce").dropna().to_numpy()
            d = np.diff(t); d = d[(d > 0) & (d < 1)]
            if len(d):
                print(f"sampling via '{tcol}': dt~{np.median(d):.4f}s  ~{1/np.median(d):.1f}Hz")
            break
    # key columns ranges
    keys = ["steering", "throttle", "brake", "speedInKmPerHour", "offsetFromLaneCenter",
            "standardDeviationFromLaneCenter", "distanceToFrontVehicle", "TTCToFrontVehicle",
            "laneCurvature", "drivingMode"]
    print("\n=== key column ranges (uv rows) ===")
    for k in keys:
        if k in df.columns:
            s = pd.to_numeric(df[k], errors="coerce")
            print(f"  {k:34s} min={np.nanmin(s):.4g} max={np.nanmax(s):.4g} "
                  f"nan={int(s.isna().sum())}")
        else:
            print(f"  {k:34s} (ABSENT)")


if __name__ == "__main__":
    main()
