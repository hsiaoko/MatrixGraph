#ifndef HYPERBLOCKER_CORE_COMPONENTS_SCHEDULER_SCHEDULER_H_
#define HYPERBLOCKER_CORE_COMPONENTS_SCHEDULER_SCHEDULER_H_

#include <unordered_map>
#include <utility>

namespace sics {
namespace matrixgraph {
namespace core {
namespace components {
namespace scheduler {

enum SchedulerType {
  kEvenSplit,
  kCHBL, // default
  kRoundRobin,
};

class Scheduler {
public:
  Scheduler(int n_device) : n_device_(n_device) {
    for (int i = 0; i < n_device; i++) {
      available_threads_by_ball_id_.insert(std::make_pair(i, 80 * 2048));
      consume_threads_by_ball_id_.insert(std::make_pair(i, 0));
    }
  }

  ~Scheduler() = default;

  virtual int GetBinID(int ball_id) = 0;

  int get_n_device() const { return n_device_; }

  void Release(int ball_id, int bin_id) {
    int n_threads;
    auto iter_consume = consume_threads_by_ball_id_.find(ball_id);
    if (iter_consume == consume_threads_by_ball_id_.end()) {
      n_threads = 1024 * 256;
    } else {
      n_threads = iter_consume->second;
    }

    auto iter_available = available_threads_by_ball_id_.find(bin_id);
    if (iter_available != available_threads_by_ball_id_.end()) {
      iter_available->second += n_threads;
    }
  }

  void RecordConsume(int ball_id, int n_threads) {
    consume_threads_by_ball_id_.insert(std::make_pair(ball_id, n_threads));
  }

  void Consume(int bin_id, int n_threads) {
    auto iter = available_threads_by_ball_id_.find(bin_id);
    if (iter != available_threads_by_ball_id_.end()) {
      iter->second -= n_threads;
    }
  }

  int get_bin_id_by_ball_id(int ball_id) const {
    auto iter = ball_id_by_bin_id_.find(ball_id);
    if (iter == ball_id_by_bin_id_.end()) {
      return INT_MAX;
    } else {
      return iter->second;
    }
  }

  void set_bin_id_by_ball_id(int ball_id, int bin_id) {
    ball_id_by_bin_id_.insert(std::make_pair(ball_id, bin_id));
  }

protected:
  int n_device_ = 0;

  std::unordered_map<int, int> ball_id_by_bin_id_;
  std::unordered_map<int, int> available_threads_by_ball_id_;
  std::unordered_map<int, int> consume_threads_by_ball_id_;
};

} // namespace scheduler
} // namespace components
} // namespace core
} // namespace matrixgraph
} // namespace sics
#endif // HYPERBLOCKER_CORE_COMPONENTS_SCHEDULER_SCHEDULER_H_
