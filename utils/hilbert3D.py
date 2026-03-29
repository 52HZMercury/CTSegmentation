import math


class Hilbert3DMapper:
    def __init__(self):
        """
        初始化并预计算常见尺寸的 Hilbert 3D 映射表。
        字典的 Key 为立方体边长 (例如 4, 8, 16...)
        字典的 Value 为映射列表 [(x,y,z), (x,y,z), ...]
        """
        self.hilbert3dMaps = {}

        # 预计算常见尺寸的映射表
        # 分别对应 128*128*128, 64*64*64, 32*32*32 等立方体
        for side_length in [128, 64, 32, 16, 8, 4]:
            # 根据边长反推迭代次数: side_length = 2 ** (iterations + 1)
            iterations = int(math.log2(side_length)) - 1
            print(f"正在预计算边长 {side_length} (迭代次数 {iterations}) 的 Hilbert 映射...")
            self.hilbert3dMaps[side_length] = self._generate_1d_to_3d_mapping(iterations)

        print("所有预计算完成！\n" + "=" * 40)

    def _generate_1d_to_3d_mapping(self, iterations):
        """
        核心生成算法 (原 get_1d_to_3d_mapping)
        """

        def hilbert3d(center, size, it, v0=0, v1=1, v2=2, v3=3, v4=4, v5=5, v6=6, v7=7):
            half = size / 2.0
            x, y, z = center

            vec_s = [
                (x - half, y + half, z - half),
                (x - half, y + half, z + half),
                (x - half, y - half, z + half),
                (x - half, y - half, z - half),
                (x + half, y - half, z - half),
                (x + half, y - half, z + half),
                (x + half, y + half, z + half),
                (x + half, y + half, z - half)
            ]

            vec = [
                vec_s[v0], vec_s[v1], vec_s[v2], vec_s[v3],
                vec_s[v4], vec_s[v5], vec_s[v6], vec_s[v7]
            ]

            it -= 1
            if it >= 0:
                tmp = []
                tmp.extend(hilbert3d(vec[0], half, it, v0, v3, v4, v7, v6, v5, v2, v1))
                tmp.extend(hilbert3d(vec[1], half, it, v0, v7, v6, v1, v2, v5, v4, v3))
                tmp.extend(hilbert3d(vec[2], half, it, v0, v7, v6, v1, v2, v5, v4, v3))
                tmp.extend(hilbert3d(vec[3], half, it, v2, v3, v0, v1, v6, v7, v4, v5))
                tmp.extend(hilbert3d(vec[4], half, it, v2, v3, v0, v1, v6, v7, v4, v5))
                tmp.extend(hilbert3d(vec[5], half, it, v4, v3, v2, v5, v6, v1, v0, v7))
                tmp.extend(hilbert3d(vec[6], half, it, v4, v3, v2, v5, v6, v1, v0, v7))
                tmp.extend(hilbert3d(vec[7], half, it, v6, v5, v2, v1, v0, v3, v4, v7))
                return tmp

            return vec

        initial_size = 2 ** iterations
        raw_points = hilbert3d((0.0, 0.0, 0.0), initial_size, iterations)

        min_x = min(p[0] for p in raw_points)
        min_y = min(p[1] for p in raw_points)
        min_z = min(p[2] for p in raw_points)

        mapped_sequence = []
        for p in raw_points:
            ix = int(round(p[0] - min_x))
            iy = int(round(p[1] - min_y))
            iz = int(round(p[2] - min_z))
            mapped_sequence.append((ix, iy, iz))

        return mapped_sequence

    def get_mapping(self, side_length):
        """
        获取指定边长的 1D 到 3D 映射表。
        如果请求的尺寸未在预计算中，则会实时计算并缓存。
        """
        if side_length not in self.hilbert3dMaps:
            iterations = int(math.log2(side_length)) - 1
            self.hilbert3dMaps[side_length] = self._generate_1d_to_3d_mapping(iterations)
        return self.hilbert3dMaps[side_length]


# ==========================================
# 测试与验证
# ==========================================
if __name__ == "__main__":
    # 实例化类（这会自动触发 __init__ 中的预计算过程）
    mapper = Hilbert3DMapper()

    # 直接从预计算好的字典中获取 4x4x4 的映射表
    target_side_length = 4
    sequence_4 = mapper.get_mapping(target_side_length)

    print(f"=== 获取边长 {target_side_length} 的映射表 (共 {len(sequence_4)} 个体素) ===")
    print("一维空间索引 -> 三维空间整数坐标 (x, y, z)")

    # 打印前 10 个体素的对应关系
    for i in range(10):
        print(f"索引 {i:02d} -> {sequence_4[i]}")

    # 测试未预计算的动态生成和缓存
    # sequence_2 = mapper.get_mapping(2)