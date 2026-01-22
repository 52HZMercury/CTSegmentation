import os
import json
import random


def generate_dataset_json(root_dir, output_file, train_ratio=0.8):
    data_list = []

    # --- 核心修复：确保输出目录存在 ---
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        print(f"检测到目录不存在，正在创建: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    # -------------------------------

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith("-label.nii.gz"):
                label_path = os.path.abspath(os.path.join(root, file))
                image_name = file.replace("-label.nii.gz", ".nii.gz")
                image_path = os.path.abspath(os.path.join(root, image_name))

                if os.path.exists(image_path):
                    data_list.append({
                        "image": image_path,
                        "label": label_path
                    })

    random.seed(42)
    random.shuffle(data_list)

    split_idx = int(len(data_list) * train_ratio)
    dataset_json = {
        "training": data_list[:split_idx],
        "validation": data_list[split_idx:]
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_json, f, indent=4, ensure_ascii=False)

    print(f"处理完成！JSON 已保存至: {output_file}")


if __name__ == "__main__":
    # 确保这里的路径与你报错的路径一致
    INPUT_DIR = "/workdir2/cn24/data/new_AI-ERCP-labeled_nii_gz"
    OUTPUT_JSON = "../metadata/new_AI-ERCP.json"

    generate_dataset_json(INPUT_DIR, OUTPUT_JSON)