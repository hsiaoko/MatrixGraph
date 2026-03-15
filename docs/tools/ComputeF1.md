# Compute F1

## Overview

Python tool for comparing two binary arrays and computing classification metrics (Precision, Recall, F1). Used to evaluate ML filter predictions against ground truth in SubIso.

## Functionality

- Reads two C++-compatible binary array files
- Computes TP, FP, FN, TN, Precision, Recall, F1 Score

## Parameters

| Argument | Description |
|----------|-------------|
| `path1` | First binary array (e.g. ground truth) |
| `path2` | Second binary array (e.g. predictions) |

## Input Format

C++ binary array files (e.g. `uint64_t` arrays).

## Source

`tools/python/compute_f1.py`

## Example

```bash
python tools/python/compute_f1.py <ground_truth.bin> <predictions.bin>
```

Output: TP, FP, FN, TN, Precision, Recall, F1 Score.
