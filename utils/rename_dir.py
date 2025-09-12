import os
import re


def rename_directories(root_path):
    """
    重命名指定路径下的目录，将类似 N001_pu bu pu che_0008705308_214514_20180201 的文件夹重命名为 N001

    :param root_path: 包含Takayasu_img和Takayasu_lab文件夹的根目录路径
    """
    # 定义需要处理的子目录
    sub_dirs = ['Takayasu_img', 'Takayasu_lab']

    # 遍历Takayasu_img和Takayasu_lab目录
    for sub_dir in sub_dirs:
        full_path = os.path.join(root_path, sub_dir)

        # 检查目录是否存在
        if not os.path.exists(full_path):
            print(f"目录 {full_path} 不存在，跳过...")
            continue

        if not os.path.isdir(full_path):
            print(f"{full_path} 不是一个目录，跳过...")
            continue

        # 获取目录下的所有文件夹
        for item in os.listdir(full_path):
            item_path = os.path.join(full_path, item)

            # 确保是一个目录
            if os.path.isdir(item_path):
                # 使用正则表达式匹配 Nxxx_xxx 格式
                match = re.match(r'([NT]\d+)_', item)
                if match:
                    new_name = match.group(1)
                    new_path = os.path.join(full_path, new_name)

                    # 检查目标目录是否已存在
                    if os.path.exists(new_path):
                        print(f"目标目录 {new_path} 已存在，跳过 {item}")
                        continue

                    # 重命名目录
                    try:
                        os.rename(item_path, new_path)
                        print(f"成功重命名: {item} -> {new_name}")
                    except Exception as e:
                        print(f"重命名 {item} 失败: {str(e)}")


if __name__ == "__main__":
    # 假设脚本在项目根目录运行，Takayasu_rename是根目录下的文件夹
    root_directory = os.path.join("/workdir2/cn24/data", "Takayasu_rename")

    # 如果Takayasu_rename目录不存在，则使用当前目录
    if not os.path.exists(root_directory):
        root_directory = os.getcwd()

    print(f"正在处理目录: {root_directory}")
    rename_directories(root_directory)
    print("重命名操作完成!")
