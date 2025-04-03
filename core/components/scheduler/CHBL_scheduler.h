#ifndef HYPERBLOCKER_CORE_COMPONENTS_SCHEDULER_CHBL_SCHEDULER_H_
#define HYPERBLOCKER_CORE_COMPONENTS_SCHEDULER_CHBL_SCHEDULER_H_

#include "core/components/scheduler/scheduler.h"

//#include <cuda_runtime.h>

#include <iostream>
#include <unordered_map>
#include <utility>

namespace sics {
namespace matrixgraph {
namespace core {
namespace components {
namespace scheduler {

class CHBLScheduler : public Scheduler {
public:
  CHBLScheduler(int n_device) : Scheduler(n_device) {
    std::cout << "Scheduler: CHBL." << std::endl;
  }

  ~CHBLScheduler() { std::cout << "Deconstruct: CHBL" << std::endl; }

  int GetBinID(int ball_id = 0) override {
    auto bin_id = Hash(ball_id) % get_n_device();

    int i = get_n_device();
    bool full = false;
    while (GetAvailableThreads(bin_id) < 0) {
      bin_id = (bin_id + 1) % get_n_device();
      if (--i < 0) {
        full = true;
        break;
      }
    }

    // find bin with minimum workload
    if (full) {
      int allocated_n_threads = INT_MIN;
      for (auto &iter : available_threads_by_ball_id_) {
        if (allocated_n_threads < iter.second) {
          allocated_n_threads = iter.second;
          bin_id = iter.first;
        }
      }
    }

    this->set_bin_id_by_ball_id(ball_id, bin_id);
    return bin_id;
  }

  int GetAvailableThreads(int bin_id = 0) {
    auto iter = available_threads_by_ball_id_.find(bin_id);
    if (iter != available_threads_by_ball_id_.end()) {
      // std::cout << iter->second << std::endl;
      return iter->second;
    }
    return 0;
  }

private:
  int Hash(int key) {
    unsigned int seed = 174321;
    unsigned int hash = (key * seed) >> 3;
    return (hash & 0x7FFFFFFF) % get_n_device();
  }
};

} // namespace scheduler
} // namespace components
} // namespace core
} // namespace matrixgraph
} // namespace sics
#endif // HYPERBLOCKER_CORE_COMPONENTS_SCHEDULER_CHBL_SCHEDULER_H_
