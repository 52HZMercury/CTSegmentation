import os
import json
from pathlib import Path
import random


def read_neck_files(base_dir):
    """
    读取neck文件夹和neck_label_STA文件夹，匹配图像和标签文件

    Args:
        base_dir: 基础目录路径，默认为 /workdir2/cn24/data/neck_label_STA

    Returns:
        list: 包含图像和标签路径的字典列表
    """
    neck_dir = os.path.join(base_dir, "image")
    label_dir = os.path.join(base_dir, "label")

    if not os.path.exists(neck_dir):
        print(f"错误: image目录不存在: {neck_dir}")
        return []

    if not os.path.exists(label_dir):
        print(f"错误: label目录不存在: {label_dir}")
        return []

    result = []

    image_files = sorted([f for f in os.listdir(neck_dir) if f.endswith('.nii.gz')])

    for img_file in image_files:
        base_name = img_file.replace('.nii.gz', '')

        img_path = os.path.join(neck_dir, img_file)
        label_lsta_path = os.path.join(label_dir, f"{base_name}_label.nii.gz")
        # label_rsta_path = os.path.join(label_dir, f"{base_name}_RSTA.nii.gz")
        # label_sta_path = os.path.join(label_dir, f"{base_name}_STA.nii.gz")

        # if all(os.path.exists(p) for p in [label_lsta_path, label_rsta_path, label_sta_path]):
        #     result.append({
        #         "image": img_path,
        #         "label_1": label_lsta_path,
        #         "label_2": label_rsta_path,
        #         "label_3": label_sta_path
        #     })
        #     print(f"匹配成功: {img_file}")

        if all(os.path.exists(p) for p in [label_lsta_path]):
            result.append({
                "image": img_path,
                "label": label_lsta_path,
            })
            print(f"匹配成功: {img_file}")
        else:
            missing = []
            if not os.path.exists(label_lsta_path):
                missing.append("LSTA")
            print(f"警告: {img_file} 缺少标签: {', '.join(missing)}")

    return result


def split_dataset(data, train_ratio=0.8, seed=42):
    """
    将数据集划分为训练集和测试集

    Args:
        data: 完整数据列表
        train_ratio: 训练集比例，默认0.8
        seed: 随机种子，默认42

    Returns:
        dict: 包含'training'和'validation'键的字典
    """
    random.seed(seed)

    data_copy = data.copy()
    random.shuffle(data_copy)

    split_idx = int(len(data_copy) * train_ratio)

    training_data = data_copy[:split_idx]
    validation_data = data_copy[split_idx:]

    return {
        "training": training_data,
        "validation": validation_data
    }


def save_to_json(data, output_path):
    """
    将数据保存为JSON文件

    Args:
        data: 要保存的数据
        output_path: 输出JSON文件路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"\nJSON文件已保存到: {output_path}")
    if isinstance(data, dict) and 'training' in data and 'validation' in data:
        print(f"训练集: {len(data['training'])} 条数据")
        print(f"验证集: {len(data['validation'])} 条数据")
        print(f"总计: {len(data['training']) + len(data['validation'])} 条数据")
    else:
        print(f"共包含 {len(data)} 条数据")


if __name__ == "__main__":
    base_directory = "/workdir2/cn24/data/Upper_limb_vessels"

    output_json_path = "../metadata/Upper_limb_vessels.json"

    print(f"开始读取目录: {base_directory}")
    print("-" * 60)

    dataset = read_neck_files(base_directory)

    if dataset:
        print(f"\n总共找到 {len(dataset)} 条匹配数据")
        print("正在按8:2比例划分训练集和验证集...")

        split_data = split_dataset(dataset, train_ratio=0.8, seed=42)

        save_to_json(split_data, output_json_path)

        print("\n训练集示例数据:")
        print(json.dumps(split_data['training'][0], indent=4, ensure_ascii=False))
        if split_data['validation']:
            print("\n验证集示例数据:")
            print(json.dumps(split_data['validation'][0], indent=4, ensure_ascii=False))
    else:
        print("\n未找到匹配的文件")
