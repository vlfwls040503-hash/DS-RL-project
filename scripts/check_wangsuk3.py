# -*- coding: utf-8 -*-
import os, glob, re
from collections import Counter

D = r"<NAS_PATH set your own>"

files = glob.glob(os.path.join(D, "**", "*.csv"), recursive=True)
by_dir = Counter(os.path.relpath(os.path.dirname(f), D) for f in files)
print("dir counts:")
for d, c in by_dir.most_common():
    print("  ", repr(d), c)

names = [os.path.basename(x) for x in files]
# 패턴 후보들
pats = {
    "No.X_SY": re.compile(r"No\.(\d+)_S(\d+)"),
    "SY only": re.compile(r"_S(\d+)_"),
    "NoX only": re.compile(r"No\.?(\d+)"),
}
for k, p in pats.items():
    print(k, sum(1 for n in names if p.search(n)))

print("\nsample names (40, stride):")
for n in names[:: max(1, len(names) // 40)]:
    print("  ", n)
