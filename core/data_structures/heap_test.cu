#include <gtest/gtest.h>

namespace sics {
namespace matrixgraph {
namespace core {
namespace task {
namespace kernel {

class MiniHeapTest : public ::testing::Test {
 protected:
  MiniHeapTest() = default;
  ~MiniHeapTest() override = default;

  // MiniHeapTest() {
  //::testing::FLAGS_gtest_death_test_style = "threadsafe";
  //}

  // TEST_F(MiniHeapTest, DefaultConstructor) {}
};

}  // namespace kernel
}  // namespace task
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics