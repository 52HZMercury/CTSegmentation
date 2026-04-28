import json
import os
import numpy as np
import nibabel as nib

json_path = "metadata/lower_cta.json"

with open(json_path, "r") as f:
    data = json.load(f)

for split in ["training", "validation"]:
    print(f"\n===== {split} =====")
    for i, item in enumerate(data[split]):
        label_path = item["label"]
        arr = np.asanyarray(nib.load(label_path).dataobj)
        vals = np.unique(arr)

        print(
            split,
            i,
            os.path.basename(label_path),
            "min =", vals.min(),
            "max =", vals.max(),
            "unique =", vals[:20],
            "num_unique =", len(vals),
        )

        if vals.min() < 0 or vals.max() >= 2 or not set(vals.tolist()).issubset({0, 1}):
            print("  >>> BAD LABEL VALUE FOUND!")