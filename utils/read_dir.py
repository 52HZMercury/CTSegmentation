import os
import json
import random


def read_directory(path, result):
    paths = os.listdir(path)
    for i, item in enumerate(paths):
        sub_path = os.path.join(path, item)
        if os.path.isdir(sub_path):
            result[item] = {}
            read_directory(sub_path, result[item])
        else:
            result[item] = item


if __name__ == "__main__":
    fpath = '/workdir2/cn24/data/Takayasu_rename'
    # part = 'Abdominal_Infrarenal'
    # part = 'Abdominal_Suprarenal'
    # part = 'Ascending_Arch'
    # part = 'Descending_Thoracic'
    # part = 'DIS'
    # part = 'DIS_Abdominal_Infrarenal'
    # part = 'DIS_Abdominal_Suprarenal'
    # part = 'DIS_Ascending_Arch'
    part = 'DIS_Descending_Thoracic'
    filename = f'metadata/{part}.json'
    output = []

    # 定义图像和标签文件夹路径
    img_path = os.path.join(fpath, "Takayasu_img")
    lab_path = os.path.join(fpath, "Takayasu_lab")

    # 检查文件夹是否存在
    if not os.path.exists(img_path) or not os.path.exists(lab_path):
        print("Takayasu_img 或 Takayasu_lab 文件夹不存在")
        exit(1)

    # 获取图像文件夹中的所有子文件夹
    img_dirs = os.listdir(img_path)

    # 遍历每个图像文件夹
    for img_dir in img_dirs:
        img_dir_path = os.path.join(img_path, img_dir)
        if os.path.isdir(img_dir_path):
            # 构造对应的标签文件夹路径
            lab_dir_path = os.path.join(lab_path, img_dir)

            # 检查标签文件夹是否存在
            if os.path.exists(lab_dir_path) and os.path.isdir(lab_dir_path):
                # 检查part.nii.gz文件是否存在
                part_label_path = os.path.join(lab_dir_path, f"{part}.nii.gz")
                if os.path.exists(part_label_path):
                    # 添加到输出列表
                    output.append({
                        "image": f"{fpath}/Takayasu_img/{img_dir}/arterial_phase.nii.gz",
                        "label": f"{fpath}/Takayasu_lab/{img_dir}/{part}.nii.gz",
                        "all_lab": f"{fpath}/Takayasu_lab/{img_dir}/ALL.nii.gz"
                    })

    random.shuffle(output)

    # 按8:2比例分割为training和validation
    total_count = len(output)
    train_count = int(total_count * 0.8)

    training_data = output[:train_count]
    validation_data = output[train_count:]

    # 构造最终的JSON结构
    result_json = {
        "training": training_data,
        "validation": validation_data
    }

    print(f"Total: {total_count}, Training: {len(training_data)}, Validation: {len(validation_data)}")
    json_res = json.dumps(result_json, indent=2)
    with open(filename, 'w') as fp:
        fp.write(json_res)
