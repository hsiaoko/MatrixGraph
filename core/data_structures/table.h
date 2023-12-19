#ifndef HYPERBLOCKER_CORE_DATA_STRUCTURES_TABLE_H_
#define HYPERBLOCKER_CORE_DATA_STRUCTURES_TABLE_H_

#include <cstring>
#include <cuda_runtime.h>

#include <rapidcsv.h>

#include "core/util/atomic.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace data_structures {

class SerializedTable {
public:
  void Show(size_t n_rows = 3) const {
    std::cout << "ShowTable: " << std::endl;
    n_rows = n_rows < n_rows_ ? n_rows : n_rows_;
    for (size_t j = 0; j < n_rows; j++) {
      for (size_t i = 0; i < n_cols_; i++) {
        std::cout << (data_ + aligned_tuple_size_ * j + col_offset_[i]) << "|";
      }
      std::cout << std::endl;
    }
  }

  char *get_data_base_ptr() const { return data_; }
  size_t *get_col_size_base_ptr() const { return col_size_; }
  size_t *get_col_offset_base_ptr() const { return col_offset_; }

  char *get_term_ptr(size_t col, size_t row) const {
    return data_ + aligned_tuple_size_ * row + col_offset_[col];
  }

  size_t get_aligned_tuple_size() const { return aligned_tuple_size_; }
  size_t get_n_rows() const { return n_rows_; }
  size_t get_n_cols() const { return n_cols_; }

private:
  friend class SerializableTable;

  size_t n_rows_ = 0;
  size_t n_cols_ = 0;
  size_t aligned_tuple_size_ = 0;
  size_t *col_size_;
  size_t *col_offset_;
  char *data_;
};

class SerializableTable {
public:
  SerializableTable(const std::vector<std::vector<std::string>> &cols)
      : cols_(cols) {
    n_cols_ = cols.size();
    n_rows_ = cols[0].size();
    // Serialize();
    Serialize2PinnedMemory();
  }

  void Serialize() {
    if (is_complete_)
      return;

    serialized_table_.n_cols_ = n_cols_;
    serialized_table_.n_rows_ = n_rows_;

    // First traversal to get the max length of the strings.
    auto max_length_col = new unsigned[n_cols_]();
    serialized_table_.col_size_ = new size_t[n_cols_]();
    serialized_table_.col_offset_ = new size_t[n_cols_]();
    for (size_t i = 0; i < n_cols_; i++) {
      auto &&col = cols_[i];
      std::for_each(col.begin(), col.end(),
                    [&max_length_col, i](std::string &s) {
                      sics::matrixgraph::core::util::WriteMax(
                          (max_length_col + i), (unsigned)s.length());
                    });
      serialized_table_.col_size_[i] = max_length_col[i] + 1;
    }

    // Compute mata data of the table.
    for (size_t i = 0; i < n_cols_; i++) {
      serialized_table_.aligned_tuple_size_ += serialized_table_.col_size_[i];
      if (i > 0)
        serialized_table_.col_offset_[i] =
            serialized_table_.col_offset_[i - 1] +
            serialized_table_.col_size_[i - 1];
    }

    serialized_table_.aligned_tuple_size_ =
        (((serialized_table_.aligned_tuple_size_ + 1) >> 6) << 6) + 64;
    serialized_table_.data_ = new char[serialized_table_.aligned_tuple_size_ *
                                       n_rows_ * sizeof(char)]();

    // Second traversal to fill the data.
    for (size_t i = 0; i < n_cols_; i++) {
      auto &&col = cols_[i];
      for (size_t j = 0; j < n_rows_; j++) {
        std::memcpy(serialized_table_.data_ +
                        serialized_table_.aligned_tuple_size_ * j +
                        serialized_table_.col_offset_[i],
                    col[j].c_str(), col[j].length() * sizeof(char));
      }
    }
    delete[] max_length_col;
    is_complete_ = true;
  }

  void Serialize2PinnedMemory() {
    if (is_complete_)
      return;

    serialized_table_.n_cols_ = n_cols_;
    serialized_table_.n_rows_ = n_rows_;

    // First traversal to get the max length of the strings.
    auto max_length_col = new unsigned[n_cols_]();
    // serialized_table_.col_size_ = new size_t[n_cols_]();
    // serialized_table_.col_offset_ = new size_t[n_cols_]();

    cudaHostAlloc(&(serialized_table_.col_size_), sizeof(size_t) * n_cols_,
                  cudaHostAllocPortable);
    cudaHostAlloc(&(serialized_table_.col_offset_), sizeof(size_t) * n_cols_,
                  cudaHostAllocPortable);

    for (size_t i = 0; i < n_cols_; i++) {
      auto &&col = cols_[i];
      std::for_each(col.begin(), col.end(),
                    [&max_length_col, i](std::string &s) {
                      sics::matrixgraph::core::util::WriteMax(
                          (max_length_col + i), (unsigned)s.length());
                    });
      serialized_table_.col_size_[i] = max_length_col[i] + 1;
    }

    // Compute mata data of the table.
    for (size_t i = 0; i < n_cols_; i++) {
      serialized_table_.aligned_tuple_size_ += serialized_table_.col_size_[i];
      if (i > 0)
        serialized_table_.col_offset_[i] =
            serialized_table_.col_offset_[i - 1] +
            serialized_table_.col_size_[i - 1];
    }

    serialized_table_.aligned_tuple_size_ =
        (((serialized_table_.aligned_tuple_size_ + 1) >> 6) << 6) + 64;
    // serialized_table_.data_ = new char[serialized_table_.aligned_tuple_size_
    // *
    //                                    n_rows_ * sizeof(char)]();

    cudaHostAlloc(&(serialized_table_.data_),
                  sizeof(char) * n_rows_ *
                      serialized_table_.aligned_tuple_size_,
                  cudaHostAllocPortable);

    // Second traversal to fill the data.
    for (size_t i = 0; i < n_cols_; i++) {
      auto &&col = cols_[i];
      for (size_t j = 0; j < n_rows_; j++) {
        std::memcpy(serialized_table_.data_ +
                        serialized_table_.aligned_tuple_size_ * j +
                        serialized_table_.col_offset_[i],
                    col[j].c_str(), col[j].length() * sizeof(char));
      }
    }
    delete[] max_length_col;
    is_complete_ = true;
  }

  SerializedTable GetSerializedTable() {
    if (is_complete_)
      return serialized_table_;
    else {
      Serialize();
      return serialized_table_;
    }
  }

  size_t get_n_rows() const { return n_rows_; }
  size_t get_n_cols() const { return n_cols_; }

  std::vector<std::vector<std::string>> get_cols() const { return cols_; }

private:
  bool is_complete_ = false;

  SerializedTable serialized_table_;

  size_t n_cols_ = 0;
  size_t n_rows_ = 0;

  std::vector<std::vector<std::string>> cols_;
};

} // namespace data_structures
} // namespace core
} // namespace matrixgraph
} // namespace sics
#endif // HYPERBLOCKER_CORE_DATA_STRUCTURES_TABLE_H_