#ifndef HYPERBLOCKER_CORE_COMPONENTS_SCHEDULER_H_
#define HYPERBLOCKER_CORE_COMPONENTS_SCHEDULER_H_

class Scheduler {
public:
  Scheduler(size_t n_device) : n_device_(n_device) {}

  virtual size_t GetDeviceID() = 0;

private:
  size_t n_device_ = 0;
};

#endif // ER_HYPERBLOCKER_CORE_COMPONENTS_SCHEDULER_H_
