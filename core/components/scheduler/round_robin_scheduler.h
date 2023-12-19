#ifndef HYPERBLOCKER_CORE_COMPONENTS_SCHEDULER_ROUND_ROBIN_SCHEDULER_H_
#define HYPERBLOCKER_CORE_COMPONENTS_SCHEDULER_ROUND_ROBIN_SCHEDULER_H_

#include "core/components/scheduler/scheduler.h"
#include "core/util/bitmap.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace components {
namespace scheduler {

class RoundRobinScheduler : public Scheduler {
public:
  RoundRobinScheduler(int n_device) : Scheduler(n_device) {
    std::cout << "Scheduler: RoundRobin" << std::endl;
  }

  int GetBinID(int ball_id = 0) override { return ball_id % get_n_device(); }
};

} // namespace scheduler
} // namespace components
} // namespace core
} // namespace matrixgraph
} // namespace sics
#endif // HYPERBLOCKER_CORE_COMPONENTS_SCHEDULER_ROUND_ROBIN_SCHEDULER_H_