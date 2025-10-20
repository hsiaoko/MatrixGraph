# Binary Array Metrics Calculator

A Python tool for comparing two binary arrays and calculating performance metrics including Precision, Recall, and F1 Score.

## Description

This script reads two binary arrays from C++ compatible binary files and computes various classification metrics to evaluate the similarity between them. It's particularly useful for comparing ground truth data with prediction results in graph processing and machine learning applications.

## Features

- Read C++ binary array files
- Convert lists to binary arrays
- Calculate classification metrics:
    - True Positives (TP), False Positives (FP)
    - False Negatives (FN), True Negatives (TN)
    - Precision, Recall, and F1 Score

## Usage

```bash
python compute_f1.py <binary_file1> <binary_file2>
```
