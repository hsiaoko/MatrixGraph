#ifndef HYPERBLOCKER_CORE_COMPONENTS_DATA_MNGR_H_
#define HYPERBLOCKER_CORE_COMPONENTS_DATA_MNGR_H_

#include <climits>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

#include <rapidcsv.h>

#include "core/common/types.h"
#include "core/components/execution_plan_generator.h"
#include "core/data_structures/table.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace components {

using sics::matrixgraph::core::data_structures::SerializableTable;
using sics::matrixgraph::core::data_structures::SerializedTable;

class DataMngr {
public:
  DataMngr(const std::string &data_path, const std::string &sep = ",",
           bool read_header = false) {

    rapidcsv::Document doc(data_path, rapidcsv::LabelParams(0, -1));

    auto n_rows = doc.GetRowCount();
    auto n_cols = doc.GetColumnCount();

    std::vector<std::vector<std::string>> cols;
    cols.reserve(n_cols);

    for (size_t i = 0; i < n_cols; i++) {
      auto &&col = doc.GetColumn<std::string>(i);
      cols.push_back(col);
    }

    SerializableTable tb(cols);
    serializable_table_vec_l_.push_back(tb);
  }

  DataMngr(const std::string &data_path_l, const std::string &data_path_r,
           const std::string &sep = ",", bool read_header = false) {
    rapidcsv::Document doc_l(data_path_l, rapidcsv::LabelParams(-1, -1),
                             rapidcsv::SeparatorParams(*sep.c_str()));
    rapidcsv::Document doc_r(data_path_r, rapidcsv::LabelParams(-1, -1),
                             rapidcsv::SeparatorParams(*sep.c_str()));

    auto n_rows_l = doc_l.GetRowCount();
    auto n_cols_l = doc_l.GetColumnCount();
    auto n_rows_r = doc_r.GetRowCount();
    auto n_cols_r = doc_r.GetColumnCount();

    std::vector<std::vector<std::string>> cols_l;
    cols_l.reserve(n_cols_l);

    for (size_t i = 0; i < n_cols_l; i++) {
      auto &&col_l = doc_l.GetColumn<std::string>(i);
      cols_l.push_back(col_l);
    }

    SerializableTable tb_l(cols_l);
    serializable_table_vec_l_.push_back(tb_l);

    std::vector<std::vector<std::string>> cols_r;
    cols_r.reserve(n_cols_r);

    for (size_t i = 0; i < n_cols_r; i++) {
      auto &&col_r = doc_r.GetColumn<std::string>(i);
      cols_r.push_back(col_r);
    }

    SerializableTable tb_r(cols_r);
    serializable_table_vec_r_.push_back(tb_r);
  }

  void DataPartitioning(const SerializedExecutionPlan &serialized_ep,
                        size_t n_partitions, int prefix_hash_predicate_index) {
    if (is_complete_)
      return;

    if (serializable_table_vec_r_.size() == 0) {
      // one table.
      auto serializable_table = serializable_table_vec_l_.front();
      serializable_table_vec_l_.erase(serializable_table_vec_l_.begin(),
                                      serializable_table_vec_l_.end());

      std::vector<std::vector<std::vector<std::string>>> bucket;
      bucket.resize(n_partitions);
      for (size_t i = 0; i < n_partitions; i++)
        bucket[i].resize(serializable_table.get_n_cols());

      std::vector<int> ivec(serializable_table.get_n_rows());
      std::iota(std::begin(ivec), std::end(ivec), 0);

      // Perform data partitioning.
      std::for_each(ivec.begin(), ivec.end(), [&](auto &i) {
        auto bucket_id = 0;
        for (size_t j = 0; j < (serialized_ep.length) && j < MAX_HASH_TIMES;
             j++) {
          if (serialized_ep.pred_type[j] == EQUALITIES) {
            auto pred_index = serialized_ep.pred_index[j];
            bucket_id = (std::hash<std::string>{}(
                             serializable_table.get_cols()[pred_index][i]) +
                         std::hash<int>{}(bucket_id)) %
                        n_partitions;
          }
        }
        for (size_t k = 0; k < serializable_table.get_n_cols(); k++) {
          bucket[bucket_id][k].push_back(serializable_table.get_cols()[k][i]);
        }
      });

      for (size_t i = 0; i < n_partitions; i++) {
        serializable_table_vec_l_.push_back(SerializableTable(bucket[i]));
      }
    } else {
      // two table.
      auto serializable_table_l = serializable_table_vec_l_.front();
      serializable_table_vec_l_.erase(serializable_table_vec_l_.begin(),
                                      serializable_table_vec_l_.end());
      auto serializable_table_r = serializable_table_vec_r_.front();
      serializable_table_vec_r_.erase(serializable_table_vec_r_.begin(),
                                      serializable_table_vec_r_.end());

      // Init.
      std::vector<std::vector<std::vector<std::string>>> bucket_l;
      bucket_l.resize(n_partitions);
      std::vector<std::vector<std::vector<std::string>>> bucket_r;
      bucket_r.resize(n_partitions);

      for (size_t i = 0; i < n_partitions; i++)
        bucket_l[i].resize(serializable_table_l.get_n_cols());
      for (size_t i = 0; i < n_partitions; i++)
        bucket_r[i].resize(serializable_table_r.get_n_cols());

      std::vector<int> ivec_l(serializable_table_l.get_n_rows());
      std::iota(std::begin(ivec_l), std::end(ivec_l), 0);
      std::vector<int> ivec_r(serializable_table_r.get_n_rows());
      std::iota(std::begin(ivec_r), std::end(ivec_r), 0);

      auto &&cols_l = serializable_table_l.get_cols();
      auto &&cols_r = serializable_table_r.get_cols();

      // Perform data partitioning for table 1.
      std::for_each(ivec_l.begin(), ivec_l.end(), [&](auto &i) {
        auto bucket_id = 0;
        for (size_t j = 0; j < (serialized_ep.length) && j < MAX_HASH_TIMES;
             j++) {
          if (serialized_ep.pred_type[j] == EQUALITIES) {
            auto pred_index = serialized_ep.pred_index[j];
            auto local_bucket_id =
                (std::hash<std::string>{}(cols_l[pred_index][i])) %
                n_partitions;
            bucket_id = (bucket_id + local_bucket_id) % n_partitions;
          }
        }

        if (prefix_hash_predicate_index != INT_MAX) {
          if (cols_l[prefix_hash_predicate_index][i].length() > 2) {
            auto two_d_bucket_id =
                (std::hash<std::string>{}(
                     cols_l[prefix_hash_predicate_index][i].substr(2)) +
                 std::hash<int>{}(bucket_id)) %
                (n_partitions);
            bucket_id = (bucket_id + two_d_bucket_id) % n_partitions;
          }
        }

        for (size_t k = 0; k < serializable_table_l.get_n_cols(); k++) {
          bucket_l[bucket_id][k].push_back(cols_l[k][i]);
        }
      });

      // Perform data partitioning for table 2.
      std::for_each(ivec_r.begin(), ivec_r.end(), [&](auto &i) {
        auto bucket_id = 0;
        for (size_t j = 0; j < (serialized_ep.length) && j < MAX_HASH_TIMES;
             j++) {
          if (serialized_ep.pred_type[j] == EQUALITIES) {
            auto pred_index = serialized_ep.pred_index[j];
            auto local_bucket_id =
                (std::hash<std::string>{}(cols_r[pred_index][i])) %
                n_partitions;
            bucket_id = (bucket_id + local_bucket_id) % n_partitions;
          }
        }

        if (prefix_hash_predicate_index != INT_MAX) {
          if (cols_r[prefix_hash_predicate_index][i].length() > 2) {
            auto two_d_bucket_id =
                (std::hash<std::string>{}(
                     cols_r[prefix_hash_predicate_index][i].substr(2)) +
                 std::hash<int>{}(bucket_id)) %
                (n_partitions);
            bucket_id = (bucket_id + two_d_bucket_id) % n_partitions;
          }
        }

        for (size_t k = 0; k < serializable_table_r.get_n_cols(); k++) {
          bucket_r[bucket_id][k].push_back(cols_r[k][i]);
        }
      });

      for (size_t i = 0; i < n_partitions; i++) {
        serializable_table_vec_l_.push_back(SerializableTable(bucket_l[i]));
        serializable_table_vec_r_.push_back(SerializableTable(bucket_r[i]));
      }
    }

    is_complete_ = true;
  }

  std::pair<SerializedTable, SerializedTable> GetPartition(size_t i) {
    assert(i < serializable_table_vec_l_.size());

    if (serializable_table_vec_r_.size() > 0) {
      auto serialized_table_l =
          serializable_table_vec_l_[i].GetSerializedTable();
      auto serialized_table_r =
          serializable_table_vec_r_[i].GetSerializedTable();
      return std::make_pair(serialized_table_l, serialized_table_r);
    } else {
      auto serialized_table_l =
          serializable_table_vec_l_[i].GetSerializedTable();
      auto serialized_table_r =
          serializable_table_vec_l_[i].GetSerializedTable();
      return std::make_pair(serialized_table_l, serialized_table_r);
    }
  }

  size_t get_n_partitions() const { return serializable_table_vec_l_.size(); }

private:
  bool is_complete_ = false;

  std::vector<SerializableTable> serializable_table_vec_l_;
  std::vector<SerializableTable> serializable_table_vec_r_;
};

} // namespace components
} // namespace core
} // namespace matrixgraph
} // namespace sics
#endif // HYPERBLOCKER_CORE_COMPONENTS_DATA_MNGR_H_
