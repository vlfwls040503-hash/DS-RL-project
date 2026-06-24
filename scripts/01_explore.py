# -*- coding: utf-8 -*-
"""
01_explore.py
Deep exploration of the driving-simulator raw CSVs to lock down the schema
for behavioral cloning (full driving control: steering / throttle / brake).

Outputs:
  - reports/explore_summary.json   (machine-readable stats)
  - reports/explore_summary.md     (human-readable summary)
Processes files one at a time to keep memory bounded.
"""
import os, re, json, glob, math
import numpy as np
import pandas as pd

RAW_DIR = r"<DATA_DIR set your own>"
OUT_DIR = r"D:\driving_bc\reports"
os.makedirs(OUT_DIR, exist_ok=True)

# Only main runs: 남산터널_피실험자{N}_S{1..4}.csv  (exclude 예비주행, 남산일반, 궤적비교)
PAT = re.compile(r"남산터널_피실험자(\d+)_S([1-4])\.csv$")

# Candidate action columns (we will inspect ranges and pick the human-input ones)
ACTION_CANDIDATES = [
    "steering", "appliedSteering", "rawSteering", "steeringVelocity",
    "throttle", "appliedThrottle", "rawThrottle",
    "brake", "appliedBrake", "rawBrake",
]
# Candidate state columns
STATE_CANDIDATES = [
    "speedInKmPerHour", "speedInMetresPerSecond",
    "localAccelInMetresPerSecond2 X", "localAccelInMetresPerSecond2 Y", "localAccelInMetresPerSecond2 Z",
    "bodyRotSpeedInRadsPerSecond Yaw",
    "turningCurvature", "laneCurvature",
    "offsetFromLaneCenter", "offsetFromRoadCenter", "standardDeviationFromLaneCenter",
    "distanceToLeftBorder", "distanceToRightBorder", "carriagewayWidth", "laneWidth",
    "distanceToFrontVehicle", "TTCToFrontVehicle",
    "roadLongitudinalSlope", "roadLateralSlope",
    "speedLimit", "laneNumber",
    "leftLaneOverLap", "rightLaneOverLap",
]
META = ["time", "scenarioTime", "type", "drivingMode", "automaticControl", "inIntersection",
        "road", "distanceAlongRoad"]

ALL_COLS = list(dict.fromkeys(ACTION_CANDIDATES + STATE_CANDIDATES + META))


def numstats(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").astype("float64")
    finite = s.replace([np.inf, -np.inf], np.nan)
    n = int(s.shape[0])
    n_nan = int(s.isna().sum())
    n_inf = int(np.isinf(s.to_numpy(dtype="float64", na_value=np.nan)).sum())
    d = dict(n=n, n_nan=n_nan, n_inf=n_inf)
    f = finite.dropna()
    if len(f):
        d.update(min=float(f.min()), max=float(f.max()), mean=float(f.mean()),
                 std=float(f.std()), p01=float(f.quantile(.01)), p50=float(f.quantile(.5)),
                 p99=float(f.quantile(.99)))
    return d


def merge_stats(acc, st):
    """Accumulate min/max/sum/sumsq/count + nan/inf across files for a column."""
    a = acc
    a["n"] += st["n"]; a["n_nan"] += st["n_nan"]; a["n_inf"] += st["n_inf"]
    if "min" in st:
        a["min"] = st["min"] if a["min"] is None else min(a["min"], st["min"])
        a["max"] = st["max"] if a["max"] is None else max(a["max"], st["max"])
        a["_vals"].append((st.get("p01"), st.get("p50"), st.get("p99")))
        a["_means"].append(st["mean"]); a["_cnt"].append(st["n"] - st["n_nan"] - st["n_inf"])


def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    runs = []
    for f in files:
        m = PAT.search(os.path.basename(f))
        if m:
            runs.append((f, int(m.group(1)), int(m.group(2))))
    print(f"main-run files: {len(runs)}")

    col_acc = {c: dict(n=0, n_nan=0, n_inf=0, min=None, max=None, _vals=[], _means=[], _cnt=[])
               for c in ALL_COLS}
    per_scenario_rows = {1: 0, 2: 0, 3: 0, 4: 0}
    per_subject = {}
    dt_samples = []
    drivingmode_counts = {}
    header_cols = None
    file_reports = []

    for i, (f, subj, scen) in enumerate(runs):
        try:
            # read full file but only keep needed columns where present
            df = pd.read_csv(f, low_memory=False)
        except Exception as e:
            print("READ FAIL", f, e); continue
        if header_cols is None:
            header_cols = list(df.columns)
        uv = df[df["type"] == "uv"].copy()
        nrows = len(uv)
        per_scenario_rows[scen] += nrows
        per_subject.setdefault(subj, 0)
        per_subject[subj] += nrows

        # sampling dt
        if "time" in uv:
            t = pd.to_numeric(uv["time"], errors="coerce").dropna().to_numpy()
            if len(t) > 10:
                dt = np.diff(t)
                dt = dt[(dt > 0) & (dt < 1)]
                if len(dt):
                    dt_samples.append(float(np.median(dt)))
        # drivingMode distribution
        if "drivingMode" in uv:
            for k, v in uv["drivingMode"].value_counts(dropna=False).items():
                drivingmode_counts[str(k)] = drivingmode_counts.get(str(k), 0) + int(v)

        for c in ALL_COLS:
            if c in uv.columns:
                merge_stats(col_acc[c], numstats(uv[c]))

        file_reports.append(dict(file=os.path.basename(f), subject=subj, scenario=scen, uv_rows=nrows))
        if (i + 1) % 20 == 0:
            print(f"  processed {i+1}/{len(runs)}")
        del df, uv

    # finalize
    summary = dict(
        n_files=len(runs),
        subjects=sorted(per_subject.keys()),
        n_subjects=len(per_subject),
        per_scenario_uv_rows=per_scenario_rows,
        total_uv_rows=int(sum(per_scenario_rows.values())),
        median_dt_sec=float(np.median(dt_samples)) if dt_samples else None,
        approx_hz=(1.0 / float(np.median(dt_samples))) if dt_samples else None,
        drivingMode_counts=drivingmode_counts,
        header_cols=header_cols,
    )
    cols_out = {}
    for c, a in col_acc.items():
        present = a["n"] > 0
        d = dict(present=present, n=a["n"], n_nan=a["n_nan"], n_inf=a["n_inf"],
                 min=a["min"], max=a["max"])
        if a["_means"] and sum(a["_cnt"]) > 0:
            w = np.array(a["_cnt"], dtype=float)
            d["mean"] = float(np.average(np.array(a["_means"]), weights=w))
            p50s = [v[1] for v in a["_vals"] if v[1] is not None]
            if p50s:
                d["median_of_file_medians"] = float(np.median(p50s))
        cols_out[c] = d
    summary["columns"] = cols_out
    summary["per_subject_uv_rows"] = per_subject

    with open(os.path.join(OUT_DIR, "explore_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    # markdown
    lines = []
    lines.append("# 데이터 탐색 요약\n")
    lines.append(f"- 본주행 파일: **{summary['n_files']}개**, 피실험자: **{summary['n_subjects']}명** "
                 f"{summary['subjects']}")
    lines.append(f"- 시나리오별 자차(uv) 행수: {summary['per_scenario_uv_rows']}")
    lines.append(f"- 총 자차 행수: **{summary['total_uv_rows']:,}** "
                 f"(샘플링 ~{summary['approx_hz']:.1f}Hz, dt≈{summary['median_dt_sec']:.4f}s)")
    lines.append(f"- drivingMode 분포: {summary['drivingMode_counts']}")
    lines.append("\n## 행동(Action) 후보 컬럼\n")
    lines.append("| col | present | min | max | mean | nan | inf |")
    lines.append("|---|---|---|---|---|---|---|")
    for c in ACTION_CANDIDATES:
        d = cols_out[c]
        if d["present"]:
            lines.append(f"| {c} | Y | {d['min']:.4g} | {d['max']:.4g} | "
                         f"{d.get('mean', float('nan')):.4g} | {d['n_nan']} | {d['n_inf']} |")
        else:
            lines.append(f"| {c} | N | | | | | |")
    lines.append("\n## 상태(State) 후보 컬럼\n")
    lines.append("| col | present | min | max | mean | nan | inf |")
    lines.append("|---|---|---|---|---|---|---|")
    for c in STATE_CANDIDATES:
        d = cols_out[c]
        if d["present"]:
            lines.append(f"| {c} | Y | {d['min']:.4g} | {d['max']:.4g} | "
                         f"{d.get('mean', float('nan')):.4g} | {d['n_nan']} | {d['n_inf']} |")
        else:
            lines.append(f"| {c} | N | | | | | |")
    with open(os.path.join(OUT_DIR, "explore_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("DONE -> reports/explore_summary.{json,md}")
    print(f"total uv rows: {summary['total_uv_rows']:,}  approx_hz={summary['approx_hz']}")


if __name__ == "__main__":
    main()
