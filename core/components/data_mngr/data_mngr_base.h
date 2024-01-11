#ifndef MATRIXGRAPH_CORE_COMPONENTS_DATA_MNGR_H_
#define MATRIXGRAPH_CORE_COMPONENTS_DATA_MNGR_H_

namespace sics {
namespace matrixgraph {
namespace core {
namespace components {

class DataMngrBase {
public:
  virtual void GetData(void *data) = 0;
};

} // namespace components
} // namespace core
} // namespace matrixgraph
} // namespace sics
#endif // MATRIXGRAPH_CORE_COMPONENTS_DATA_MNGR_H_
