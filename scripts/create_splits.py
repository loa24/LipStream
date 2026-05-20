import os
import json
from pathlib import Path

DATA_DIR = Path("data/processed/lipread/train")
SPLIT_DIR = Path("data/splits")

SPLIT_DIR.mkdir(parents=True, exist_ok=True)

train, val, test = [], [], []

for sample in sorted(os.listdir(DATA_DIR)):
    meta_path = DATA_DIR / sample / "meta.json"

    with open(meta_path) as f:
        meta = json.load(f)

    speaker = meta["speaker"]
    import re
    num = int(re.search(r'\d+', speaker).group())
    

    if num <= 24:
        train.append(sample)
    elif num <= 29:
        val.append(sample)
    else:
        test.append(sample)

with open(SPLIT_DIR / "train.txt", "w") as f:
    f.write("\n".join(train))

with open(SPLIT_DIR / "val.txt", "w") as f:
    f.write("\n".join(val))

with open(SPLIT_DIR / "test.txt", "w") as f:
    f.write("\n".join(test))

print("Train:", len(train))
print("Val:", len(val))
print("Test:", len(test))

