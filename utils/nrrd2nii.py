import os
from glob import glob
import SimpleITK as sitk  # pip install SimpleITK

baseDir = os.path.normpath('/workdir2/cn24/data/AI-ERCP-labled')
outputDir = os.path.normpath('/workdir2/cn24/data/AI-ERCP-labeled_nii_gz')

files = glob(os.path.join(baseDir, '**', '*.nrrd'), recursive=True)

for file in files:
    # 1. 使用 SimpleITK 读取，它会自动处理 Header
    image = sitk.ReadImage(file)

    # 2. 构建输出路径
    rel_path = os.path.relpath(file, baseDir)
    output_file_path = os.path.join(outputDir, rel_path)
    output_subdir = os.path.dirname(output_file_path)
    os.makedirs(output_subdir, exist_ok=True)

    # 3. 修改后缀名
    output_file_path = os.path.splitext(output_file_path)[0] + '.nii.gz'

    # 4. 直接保存
    # SimpleITK 会自动处理从 NRRD(LPS) 到 NIfTI(RAS) 的坐标转换逻辑
    sitk.WriteImage(image, output_file_path)

    print(f"Converted: {file} -> {output_file_path}")


# import os
# import glob
# import SimpleITK as sitk
# from tqdm import tqdm
#
#
# def convert_nrrd_to_nii(base_dir, output_dir):
#     """
#     将 base_dir 下所有的 .nrrd 文件转换为 .nii.gz，并保持原有文件夹结构存储到 output_dir。
#     """
#
#     # 1. 规范化路径，防止不同操作系统的路径分隔符问题
#     base_dir = os.path.normpath(base_dir)
#     output_dir = os.path.normpath(output_dir)
#
#     print(f"[-] 正在查找 NRRD 文件...")
#     print(f"    源目录: {base_dir}")
#
#     # 2. 递归查找所有 nrrd 文件
#     # 注意：glob 返回的是绝对路径
#     files = glob.glob(os.path.join(base_dir, '**', '*.nrrd'), recursive=True)
#
#     total_files = len(files)
#     print(f"[-] 共发现 {total_files} 个文件。准备开始转换...")
#
#     if total_files == 0:
#         print("[!] 未找到 .nrrd 文件，请检查路径。")
#         return
#
#     # 3. 遍历并转换 (使用 tqdm 显示进度条)
#     success_count = 0
#     fail_count = 0
#     errors = []
#
#     for file_path in tqdm(files, desc="Converting", unit="file"):
#         try:
#             # --- A. 计算相对路径以保留目录结构 ---
#             # 例如: /work/data/patient01/image.nrrd -> patient01/image.nrrd
#             rel_path = os.path.relpath(file_path, base_dir)
#
#             # --- B. 构建输出文件路径 ---
#             # 替换扩展名 .nrrd -> .nii.gz
#             # 注意：处理 .nrrd 这种单后缀比较简单，如果是 .seg.nrrd 需要特别注意
#             rel_path_no_ext = os.path.splitext(rel_path)[0]
#             out_file_path = os.path.join(output_dir, rel_path_no_ext + ".nii.gz")
#
#             # --- C. 创建目标文件夹 ---
#             # 获取输出文件的父目录
#             out_folder = os.path.dirname(out_file_path)
#             if not os.path.exists(out_folder):
#                 os.makedirs(out_folder, exist_ok=True)
#
#             # --- D. SimpleITK 读取与写入 ---
#             # 读取
#             image = sitk.ReadImage(file_path)
#
#             # 写入 (使用 image compress 来生成 .nii.gz)
#             sitk.WriteImage(image, out_file_path)
#
#             success_count += 1
#
#         except Exception as e:
#             fail_count += 1
#             errors.append(f"File: {file_path} | Error: {str(e)}")
#
#     # 4. 输出统计报告
#     print("\n" + "=" * 30)
#     print(f"处理完成 Summary:")
#     print(f"成功: {success_count}")
#     print(f"失败: {fail_count}")
#     print(f"输出目录: {output_dir}")
#     print("=" * 30)
#
#     if errors:
#         print("\n[!] 错误日志:")
#         for err in errors:
#             print(err)
#
#
# if __name__ == "__main__":
#     # 配置路径
#     # 注意：Python字符串中，建议使用原始字符串 r'' 或者正斜杠 / 来避免转义问题
#     BASE_DIR = r'/workdir2/cn24/data/AI-ERCP-labled'
#     OUTPUT_DIR = r'/workdir2/cn24/data/new_AI-ERCP-labeled_nii_gz'
#
#     # 执行转换
#     convert_nrrd_to_nii(BASE_DIR, OUTPUT_DIR)