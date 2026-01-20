import os
from glob import glob
import nrrd  # pip install pynrrd, if pynrrd is not already installed
import nibabel as nib  # pip install nibabel, if nibabel is not already installed
import numpy as np

baseDir = os.path.normpath('/workdir2/cn24/data/AI-ERCP-labled')
outputDir = os.path.normpath('/workdir2/cn24/data/AI-ERCP-labeled_niigz')

# 递归查找所有 .nrrd 文件
files = glob(os.path.join(baseDir, '**', '*.nrrd'), recursive=True)

for file in files:
    # load nrrd
    _nrrd = nrrd.read(file)
    data = _nrrd[0]
    header = _nrrd[1]

    # 获取相对于基础目录的路径
    rel_path = os.path.relpath(file, baseDir)

    # 构建输出文件的完整路径
    output_file_path = os.path.join(outputDir, rel_path)

    # 确保输出子目录存在
    output_subdir = os.path.dirname(output_file_path)
    os.makedirs(output_subdir, exist_ok=True)

    # 将文件扩展名从 .nrrd 改为 .nii.gz
    output_file_path = output_file_path[:-5] + '.nii.gz'  # 移除 .nrrd 的最后5个字符(.nrrd)，添加.nii.gz

    # save nifti
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, output_file_path)

    print(f"Converted: {file} -> {output_file_path}")