import numpy as np
import algorithms
import sys


def merge_and_save_binary(list1, list2, output_file):
    """
    合并两个 int64 列表并保存为二进制文件

    参数:
        list1 (list[int]): 第一个整数列表
        list2 (list[int]): 第二个整数列表
        output_file (str): 输出二进制文件路径
    """
    # 合并两个列表
    merged_array = np.concatenate([list1[:], list2[:]], dtype=np.int64)

    print(merged_array)
    # 保存为二进制文件
    merged_array.tofile(output_file)
    print(f"成功保存到 {output_file} (共 {len(merged_array)} 个 int64 数据)")


if __name__ == "__main__":
    """Example usage of the GraphReader class."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <array len> <graph_path> <output_path>")
        sys.exit(1)

    gt_array1 = algorithms.read_cpp_binary_array(sys.argv[1], 'q')
    gt_array1 = np.array(gt_array1, dtype=np.int64)

    gt_array2 = algorithms.read_cpp_binary_array(sys.argv[2], 'q')
    gt_array2 = np.array(gt_array2, dtype=np.int64)

    merge_and_save_binary(gt_array1, gt_array2, sys.argv[3])
