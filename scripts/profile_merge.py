# -*- coding: utf-8 -*-
"""Profile filename structure of the merge/diverge datasets (names only, no content read)."""
import os, re, glob
from collections import Counter

PATHS = {
    "신월여의_본실험": r"<NAS_PATH set your own>",
    "신월여의_추가": r"<NAS_PATH set your own>",
    "복층터널_지하도로DS": r"<NAS_PATH set your own>",
}

# Log_<ts>_<road>_<person>-<cond>_0_0_0.User_Master.csv   (road/person/cond vary)
def parse(name):
    base = name.replace(".User_Master.csv", "").replace(".csv", "")
    # underground/ground tag
    tag = "지하" if "지하" in base else ("지상" if ("지상" in base or "(상)" in base) else "?")
    # person: Korean name token before a hyphen (e.g., NAME-condition)
    person = None
    m = re.search(r"_([가-힣]{2,4})-", base)
    if m:
        person = m.group(1)
    # road token: right after Log_<digits>_
    road = None
    m2 = re.match(r"Log_\d+_([^_]+)", base)
    if m2:
        road = m2.group(1).strip()
    return tag, person, road


def main():
    for label, p in PATHS.items():
        files = glob.glob(os.path.join(p, "**", "*.csv"), recursive=True)
        files = [os.path.basename(f) for f in files]
        tags, persons, roads = Counter(), Counter(), Counter()
        for f in files:
            t, person, road = parse(f)
            tags[t] += 1
            if person:
                persons[person] += 1
            if road:
                roads[road] += 1
        print(f"\n=== {label}  (files={len(files)}) ===")
        print(f"  지상/지하 태그: {dict(tags)}")
        print(f"  고유 인물수(피실험자 후보): {len(persons)}  ->", list(persons.keys())[:20])
        print(f"  고유 도로토큰수: {len(roads)}")
        for r, c in roads.most_common(12):
            print(f"      [{c:3d}] {r}")
        # files with no person parsed
        nop = sum(1 for f in files if parse(f)[1] is None)
        print(f"  인물 파싱 실패: {nop}/{len(files)}")


if __name__ == "__main__":
    main()
