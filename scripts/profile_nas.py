# -*- coding: utf-8 -*-
"""
profile_nas.py  --  conservative compatibility scan of multiple NAS experiment folders.
Reads only the header + first ~80 data rows of one representative CSV per folder
(plus a file count) so it is light on the network. Writes a compatibility report.

Output: reports/nas_inventory.md  and  reports/nas_inventory.json
"""
import os, re, csv, json, glob, io

PATHS = [
    (r"<NAS_PATH set your own>", "결빙2022_S1"),
    (r"<NAS_PATH set your own>", "결빙2022_S2"),
    (r"<NAS_PATH set your own>", "결빙2022_S3"),
    (r"<NAS_PATH set your own>", "지하유출입2021_S1"),
    (r"<NAS_PATH set your own>", "지하유출입2021_S2"),
    (r"<NAS_PATH set your own>", "지하유출입2021_S3"),
    (r"<NAS_PATH set your own>", "신월여의2022_DS"),
    (r"<NAS_PATH set your own>", "신월여의2022_DSYei"),
    (r"<NAS_PATH set your own>", "복층터널2022_DS"),
    (r"<NAS_PATH set your own>", "지하고속도로2024_장거리"),
]

# canonical needed columns (normalized name -> role)
NEED = {
    "steering": ("action", "steering"),
    "throttle": ("action", "throttle"),
    "brake": ("action", "brake"),
    "speedinkmperhour": ("state", "speed"),
    "offsetfromlanecenter": ("state", "laneoffset"),
    "lanecurvature": ("state", "lanecurv"),
    "distancetoleftborder": ("state", "leftborder"),
    "distancetorightborder": ("state", "rightborder"),
    "localaccelinmetrespersecond2x": ("state", "accx"),
    "bodyrotspeedinradspersecondyaw": ("state", "yawrate"),
    "roadlongitudinalslope": ("state", "slope"),
    "speedlimit": ("state", "speedlimit"),
    "distancetofrontvehicle": ("front", "frontdist"),
    "ttctofrontvehicle": ("front", "ttc"),
    "standarddeviationfromlanecenter": ("front", "sdlp"),
    "type": ("meta", "type"),
    "drivingmode": ("meta", "drivingmode"),
    "scenariotime": ("meta", "scenariotime"),
}


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def sniff_read(path, max_rows=80):
    """Return (encoding, sep, header_list, rows) reading only the top of the file."""
    raw = None
    for enc in ("utf-8-sig", "cp949", "utf-8", "latin-1"):
        try:
            with io.open(path, "r", encoding=enc, errors="strict") as f:
                head = [next(f) for _ in range(max_rows + 1)]
            raw, used = head, enc
            break
        except (UnicodeDecodeError, StopIteration) as e:
            if isinstance(e, StopIteration):
                # short file; re-read all
                try:
                    with io.open(path, "r", encoding=enc, errors="strict") as f:
                        raw, used = f.readlines(), enc
                    break
                except UnicodeDecodeError:
                    continue
            continue
    if raw is None:
        with io.open(path, "r", encoding="latin-1", errors="replace") as f:
            raw = [next(f) for _ in range(max_rows + 1)]
        used = "latin-1?"
    sample = raw[0]
    sep = max([",", "\t", ";"], key=lambda c: sample.count(c))
    rows = list(csv.reader(raw, delimiter=sep))
    header = [h.strip().strip('"') for h in rows[0]]
    return used, sep, header, rows[1:]


def pick_representative(folder):
    cands = glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
    if not cands:
        return None, 0
    # prefer raw logs (Master/Log) and largest; avoid 'summary'
    def score(p):
        b = os.path.basename(p).lower()
        s = os.path.getsize(p)
        if "summary" in b:
            s *= 0.001
        if "master" in b or "log" in b or "distance_based" in b:
            s *= 1000
        return s
    rep = max(cands, key=score)
    return rep, len(cands)


def main():
    out = []
    for folder, label in PATHS:
        rec = {"label": label, "path": folder}
        if not os.path.isdir(folder):
            rec["status"] = "PATH NOT FOUND"
            out.append(rec)
            print(f"[{label}] NOT FOUND")
            continue
        rep, ncsv = pick_representative(folder)
        rec["n_csv"] = ncsv
        if rep is None:
            rec["status"] = "no CSV"
            out.append(rec)
            print(f"[{label}] no CSV")
            continue
        rec["sample_file"] = os.path.basename(rep)
        try:
            enc, sep, header, rows = sniff_read(rep)
        except Exception as e:
            rec["status"] = f"read error: {e}"
            out.append(rec)
            print(f"[{label}] read error {e}")
            continue
        nh = {norm(h): h for h in header}
        present, missing = {}, []
        for k, (role, lab) in NEED.items():
            if k in nh:
                present[lab] = nh[k]
            else:
                missing.append(lab)
        rec.update(encoding=enc, sep=("comma" if sep == "," else repr(sep)),
                   n_cols=len(header), enc_ok=True)
        rec["actions_ok"] = all(a in present for a in ("steering", "throttle", "brake"))
        rec["core_state"] = sum(1 for a in ("speed", "laneoffset", "lanecurv", "leftborder",
                                            "rightborder", "accx", "yawrate", "slope", "speedlimit")
                                if a in present)
        rec["front_veh"] = [a for a in ("frontdist", "ttc", "sdlp") if a in present]
        rec["has_type_col"] = "type" in present
        rec["missing_key"] = [m for m in missing if m in ("steering", "throttle", "brake",
                                                          "speed", "laneoffset")]
        # type values & sampling from sampled rows
        if "type" in present and rows:
            ti = header.index(present["type"])
            vals = set()
            for r in rows:
                if len(r) > ti:
                    vals.add(r[ti].strip().strip('"'))
            rec["type_values"] = sorted(vals)[:6]
        # dt from scenariotime if present
        if "scenariotime" in present and len(rows) > 5:
            ci = header.index(present["scenariotime"])
            try:
                ts = [float(r[ci]) for r in rows[:60] if len(r) > ci and r[ci].strip()]
                difs = [b - a for a, b in zip(ts, ts[1:]) if 0 < (b - a) < 5]
                if difs:
                    rec["dt_sec"] = round(sorted(difs)[len(difs) // 2], 4)
                    rec["approx_hz"] = round(1 / rec["dt_sec"], 1)
            except Exception:
                pass
        rec["status"] = "OK" if rec["actions_ok"] else "MISSING ACTIONS"
        out.append(rec)
        print(f"[{label}] files={ncsv} cols={len(header)} enc={enc} "
              f"actions={'OK' if rec['actions_ok'] else 'NO'} core={rec['core_state']}/9 "
              f"front={rec['front_veh']} hz={rec.get('approx_hz')}")

    os.makedirs(r"D:\driving_bc\reports", exist_ok=True)
    json.dump(out, open(r"D:\driving_bc\reports\nas_inventory.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    L = ["# NAS 다실험 데이터 호환성 진단\n",
         "각 폴더의 대표 CSV 1개 헤더 + 상위 행만 읽어 보수적으로 진단.\n",
         "| 실험 | CSV수 | 컬럼 | 인코딩 | 구분자 | 행동(S/T/B) | 핵심상태 | 앞차/SDLP | type | Hz | 상태 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in out:
        if r.get("status") == "PATH NOT FOUND":
            L.append(f"| {r['label']} | - | - | - | - | - | - | - | - | - | **{r['status']}** |")
            continue
        L.append("| {lab} | {n} | {c} | {e} | {s} | {a} | {cs}/9 | {fv} | {ty} | {hz} | {st} |".format(
            lab=r["label"], n=r.get("n_csv", "-"), c=r.get("n_cols", "-"),
            e=r.get("encoding", "-"), s=r.get("sep", "-"),
            a="✅" if r.get("actions_ok") else "❌",
            cs=r.get("core_state", 0), fv=",".join(r.get("front_veh", [])) or "없음",
            ty="✅" if r.get("has_type_col") else "—", hz=r.get("approx_hz", "?"),
            st=r.get("status", "?")))
    L.append("\n## 경로별 상세\n")
    for r in out:
        L.append(f"### {r['label']}")
        L.append(f"- 경로: `{r['path']}`")
        if "sample_file" in r:
            L.append(f"- 대표파일: `{r['sample_file']}`")
        if "type_values" in r:
            L.append(f"- type 값: {r['type_values']}")
        if r.get("missing_key"):
            L.append(f"- ⚠️ 누락(핵심): {r['missing_key']}")
        L.append("")
    open(r"D:\driving_bc\reports\nas_inventory.md", "w", encoding="utf-8").write("\n".join(L))
    print("\nwrote reports/nas_inventory.{md,json}")


if __name__ == "__main__":
    main()
