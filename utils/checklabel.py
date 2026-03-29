import nibabel as nib
import numpy as np
from pathlib import Path

# 指向你的 label 文件夹
label_dir = Path("/workdir2/cn24/data/SCU_dataset/CA")
empty_files = []

print("正在检查标签文件...")
for lbl_path in label_dir.glob("*.nii.gz"):
    data = nib.load(str(lbl_path)).get_fdata()
    if np.max(data) <= 0:
        empty_files.append(lbl_path.name)

if empty_files:
    print(f"❌ 发现 {len(empty_files)} 个空标签文件:")
    for f in empty_files:
        print(f"  - {f}")
else:
    print("✅ 所有标签文件都包含前景目标。")