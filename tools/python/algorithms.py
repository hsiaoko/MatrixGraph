import struct
import numpy as np
import torch
import yaml


def read_cpp_binary_array(filename, dtype='i', endianness='@'):
    """
    读取C++写入的二进制数组

    参数:
        filename: 文件名
        dtype: 数据类型格式字符(默认'i'表示32位整数)
               'i': 32位整数, 'f': 32位浮点, 'd': 64位浮点
               'q': 64位整数, 'h': 16位整数等
        endianness: 字节顺序
                   '@': 本机顺序, '=': 本机标准
                   '<': 小端, '>': 大端, '!': 网络顺序(大端)
    """
    with open(filename, 'rb') as f:
        # 读取二进制数据
        data = f.read()

    # 计算元素数量(每个元素占4字节)
    element_size = struct.calcsize(dtype)
    num_elements = len(data) // element_size
    print("read ", num_elements)
    print("[read_cpp_binary_array]: num_elements ", num_elements)
    print("[read_cpp_binary_array]: elements_size ", element_size)

    # 解包二进制数据
    format_str = f'{endianness}{num_elements}{dtype}'
    array = struct.unpack(format_str, data)

    return array


def list_to_binary_array(input_list, N):
    """
    将列表转换为二进制数组，列表中的值对应的位置设为1，其余为0

    参数:
        input_list (list): 包含要设为1的索引的列表
        N (int): 输出数组的长度

    返回:
        np.array: 生成的二进制数组
    """
    output_array = np.zeros(N, dtype=int)  # 初始化全零数组
    for k in input_list:
        if 0 <= k < N:  # 确保k在有效范围内
            output_array[k] = 1
    return output_array


def calculate_metrics(y_true, y_pred):
    """
    计算二分类任务的precision, recall和f1分数

    参数:
    y_true -- 真实标签数组 (0或1)
    y_pred -- 预测标签数组 (0或1)

    返回:
    precision, recall, f1
    """
    # 将输入转换为numpy数组以防万一
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算真正例(TP)、假正例(FP)、假负例(FN)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    # 计算precision, recall和f1
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("f1: ", f1)
    print("prefision: ", precision)
    print("recall: ", recall)

    return precision, recall, f1


def bitmap_calculate_metrics(gt_array1, gt_array2):
    """
    计算 TP, FP, FN, TN, Precision, Recall, F1

    参数:
        gt_array1 (np.array): Ground Truth 标签 (0或1)
        gt_array2 (np.array): 预测标签 (0或1)

    返回:
        dict: 包含 TP, FP, FN, TN, Precision, Recall, F1 的字典
    """
    # 确保输入是 numpy 数组
    gt_array1 = np.array(gt_array1)
    gt_array2 = np.array(gt_array2)

    # 计算 TP, FP, FN, TN
    tp = np.sum((gt_array1 == 1) & (gt_array2 == 1))
    fp = np.sum((gt_array1 == 0) & (gt_array2 == 1))
    fn = np.sum((gt_array1 == 1) & (gt_array2 == 0))
    tn = np.sum((gt_array1 == 0) & (gt_array2 == 0))

    # 计算 Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


def save_tensor_for_cpp(tensor: torch.Tensor, root_path: str):
    """保存Tensor为C++可读格式"""
    assert tensor.is_contiguous(), "Tensor must be contiguous"

    bin_path = root_path + "embedding.bin"
    meta_path = root_path + "meta.yaml"
    # 保存二进制
    tensor = tensor.float()
    print(tensor)
    np_array = tensor.detach().numpy().astype(np.float32)
    np_array.tofile(bin_path)

    # 保存元数据

    x = tensor.shape[0]
    y = 0
    if tensor.dim() == 2:
        y = tensor.shape[1]
    else:
        y = 1
    metadata = {
        "dtype": str(tensor.dtype).split(".")[-1],  # 如 "float32"
        "x": x,
        "y": y,
        "endian": "little",  # 或 "big"
        "order": "row_major"  # PyTorch默认行优先存储
    }

    # 保存为YAML
    with open(meta_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
