#ifndef MATRIXGRAPH_CORE_IO_ATTRIBUTES_IO_H_
#define MATRIXGRAPH_CORE_IO_ATTRIBUTES_IO_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/data_structures/attributes.h"

namespace sics {
namespace matrixgraph {
namespace core {
namespace io {

// Columnar disk IO for data_structures::Attributes.
//
// Directory layout (one directory per entity table):
//
//   <dir>/meta.yaml                AttributesMetadata manifest
//   <dir>/<name>.bin               flat data buffer
//   <dir>/<name>.offsets.bin       uint32[n_rows+1]  (list types and String)
//   <dir>/<name>.elem_offsets.bin  uint32[n_elements+1]  (StringList only)
//   <dir>/<name>.valid.bin         uint8[n_rows]  (when a valid mask exists)
//
// Row semantics follow the contract used with ImmutableCSR:
//   vertex table (entity_id = 0): row i == local vertex id i
//   edge table   (entity_id = 1): row i == outgoing_edges slot i
//
// String columns are serialized as a byte blob plus row offsets; on Read the
// StringView array is rebuilt into Storage.
class AttributesIO {
 public:
  using Attribute = sics::matrixgraph::core::data_structures::Attribute;

  struct ColumnManifest {
    std::string name;
    sics::matrixgraph::core::data_structures::ValueType type;
    uint32_t n_rows = 0;
    uint32_t n_elements = 0;
    bool has_offsets = false;
    bool has_elem_offsets = false;
    bool has_valid = false;
  };

  // Owning storage backing the Attribute pointers returned by Read.
  struct Storage {
    uint32_t entity_id = 0;
    std::vector<ColumnManifest> manifest;
    std::vector<Attribute> columns;
    std::vector<std::unique_ptr<uint8_t[]>> blobs;
  };

  // Writes columns (name/type/n_rows/n_elements/data/offsets/valid) into
  // `dir`. String/StringList columns are flattened to blob + offsets.
  static void Write(const std::string& dir, uint32_t entity_id,
                    const Attribute* columns, uint32_t n_columns);

  // Reads a directory written by Write into `storage`. Pointers inside
  // storage->columns reference buffers owned by `storage`.
  static void Read(const std::string& dir, Storage* storage);

  // Byte size of one primitive element of a (non-list, non-string) type.
  static uint32_t ElementSize(
      sics::matrixgraph::core::data_structures::ValueType type);
};

}  // namespace io
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // MATRIXGRAPH_CORE_IO_ATTRIBUTES_IO_H_
