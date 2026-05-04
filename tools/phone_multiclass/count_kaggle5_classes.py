from pathlib import Path
from collections import Counter

root = Path("datasets/phone_multiclass_kaggle_v1/labels_raw")
counter = Counter()

for txt in root.rglob("*.txt"):
    with open(txt, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            parts = s.split()
            if not parts or not parts[0].isdigit():
                continue
            counter[int(parts[0])] += 1

print("Class counts in labels_raw:")
for k in sorted(counter):
    print(k, counter[k])

print("\nExpected classes:")
print("0 phone")
print("1 walkie_talkie")
print("2 mouse")
print("3 cigarette_pack")
print("4 remote")