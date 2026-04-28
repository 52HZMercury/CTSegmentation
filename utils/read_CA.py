import os
import json
import random
from pathlib import Path


def generate_dataset_json(root_path, train_ratio=0.8):
    # 1. 初始化路径
    root = Path(root_path)
    img_dir = root / "image"
    label_dir = root / "label"
    output_file = "../metadata/lower_cta.json"

    # 2. 获取所有图像文件 (假设是 .nii.gz)
    images = sorted([f for f in img_dir.glob("*.nii.gz")])

    data_list = []

    print(f"🔍 正在匹配文件，总计图像数: {len(images)}")

    for img_path in images:
        # 构造对应的 label 文件名：原文件名 + "_label"
        # 例如: 001.nii.gz -> 001_label.nii.gz
        file_id = img_path.name.replace(".nii.gz", "_label")
        label_name = f"{file_id}.nii.gz"
        label_path = label_dir / label_name

        # 检查 label 是否存在
        if label_path.exists():
            data_list.append({
                "image": str(img_path.absolute()),
                "label": str(label_path.absolute())
            })
        else:
            print(f"⚠️ 警告: 未找到对应的标签文件: {label_name}")

    # 3. 随机打乱数据
    random.seed(42)  # 固定随机种子，确保实验可重复
    random.shuffle(data_list)

    # 4. 划分训练集和验证集
    split_idx = int(len(data_list) * train_ratio)
    training_set = data_list[:split_idx]
    validation_set = data_list[split_idx:]

    # 5. 写入 JSON
    output_data = {
        "training": training_set,
        "validation": validation_set
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"✅ 处理完成！")
    print(f"📝 训练集数量: {len(training_set)}")
    print(f"📝 验证集数量: {len(validation_set)}")
    print(f"💾 JSON 文件已保存至: {output_file}")


if __name__ == "__main__":
    # 根据你的描述设置根目录
    DATA_ROOT = "/workdir2/cn24/data/lower_cta"
    generate_dataset_json(DATA_ROOT)