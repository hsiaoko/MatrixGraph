import numpy as np
import algorithms
import sys


def PrintStat(gt_path1, gt_path2):
    gt_array1 = algorithms.read_cpp_binary_array(gt_path1, 'q')
    gt_array1 = np.array(gt_array1, dtype=np.int64)

    gt_array2 = algorithms.read_cpp_binary_array(gt_path2, 'q')
    gt_array2 = np.array(gt_array2, dtype=np.int64)

    print(gt_array1)
    print(gt_array2)
    # tp = algorithms.count_common_elements(gt_array1, gt_array2)
    gt_array1 = algorithms.list_to_binary_array(gt_array1, 65536000)
    gt_array2 = algorithms.list_to_binary_array(gt_array2, 65536000)

    metrics = algorithms.bitmap_calculate_metrics(gt_array1, gt_array2)
    print(f"TP: {metrics['TP']}")

    print(f"FP: {metrics['FP']}")
    print(f"FN: {metrics['FN']}")
    print(f"TN: {metrics['TN']}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"F1 Score: {metrics['F1']:.4f}")


if __name__ == "__main__":
    """Example usage of the GraphReader class."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <array len> <graph_path> <output_path>")
        sys.exit(1)

    PrintStat(sys.argv[1], sys.argv[2])
