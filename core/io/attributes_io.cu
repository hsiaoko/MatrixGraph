#include "core/io/attributes_io.cuh"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace sics {
namespace matrixgraph {
namespace core {
namespace io {

using sics::matrixgraph::core::data_structures::StringView;
using sics::matrixgraph::core::data_structures::ValueType;

namespace {

bool IsList(ValueType t) {
  return t == ValueType::kFloatList || t == ValueType::kIntList ||
         t == ValueType::kStringList || t == ValueType::kTimeList;
}

ValueType ListElementType(ValueType t) {
  switch (t) {
    case ValueType::kFloatList: return ValueType::kFloat64;
    case ValueType::kIntList: return ValueType::kInt;
    case ValueType::kTimeList: return ValueType::kTime;
    case ValueType::kStringList: return ValueType::kString;
    default: return t;
  }
}

void WriteBuffer(const std::string& path, const void* data, uint64_t bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("AttributesIO: cannot write " + path);
  if (bytes > 0) {
    out.write(reinterpret_cast<const char*>(data),
              static_cast<std::streamsize>(bytes));
  }
}

std::unique_ptr<uint8_t[]> ReadBuffer(const std::string& path,
                                      uint64_t bytes) {
  auto buf = std::make_unique<uint8_t[]>(bytes > 0 ? bytes : 1);
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("AttributesIO: cannot open " + path);
  if (bytes > 0) {
    in.read(reinterpret_cast<char*>(buf.get()),
            static_cast<std::streamsize>(bytes));
    if (!in) {
      throw std::runtime_error("AttributesIO: failed reading " + path);
    }
  }
  return buf;
}

}  // namespace

uint32_t AttributesIO::ElementSize(ValueType type) {
  switch (type) {
    case ValueType::kFloat64: return 8;
    case ValueType::kFloat32: return 4;
    case ValueType::kInt: return 8;
    case ValueType::kTime: return 8;
    case ValueType::kBool: return 1;
    case ValueType::kString: return 0;  // variable, blob + offsets
    default:
      throw std::runtime_error("AttributesIO: ElementSize on non-scalar type");
  }
}

void AttributesIO::Write(const std::string& dir, uint32_t entity_id,
                         const Attribute* columns, uint32_t n_columns) {
  std::filesystem::create_directories(dir);

  YAML::Node meta;
  meta["AttributesMetadata"]["entity_id"] = entity_id;
  meta["AttributesMetadata"]["num_columns"] = n_columns;
  YAML::Node cols(YAML::NodeType::Sequence);

  for (uint32_t c = 0; c < n_columns; ++c) {
    const Attribute& a = columns[c];
    const std::string name(a.name);
    const std::string base = dir + "/" + name;
    const bool is_string = a.type == ValueType::kString;
    const bool is_strlist = a.type == ValueType::kStringList;
    const bool has_offsets = IsList(a.type) || is_string;

    YAML::Node m;
    m["name"] = name;
    m["type"] = static_cast<uint32_t>(a.type);
    m["n_rows"] = a.n_rows;
    // For String columns the element count is the row count (one StringView
    // per row); Read relies on this to rebuild the StringView array.
    m["n_elements"] = is_string ? a.n_rows : a.n_elements;
    m["has_offsets"] = has_offsets;
    m["has_elem_offsets"] = is_strlist;
    m["has_valid"] = a.valid != nullptr;

    if (is_string || is_strlist) {
      // Flatten StringView elements into a byte blob plus element offsets.
      const auto n_elems = is_string ? a.n_rows : a.n_elements;
      const auto* views = static_cast<const StringView*>(a.data);
      std::vector<uint32_t> elem_off(n_elems + 1, 0);
      uint64_t blob_bytes = 0;
      for (uint32_t i = 0; i < n_elems; ++i) {
        blob_bytes += views[i].len;
        elem_off[i + 1] = static_cast<uint32_t>(blob_bytes);
      }
      auto blob = std::make_unique<char[]>(blob_bytes > 0 ? blob_bytes : 1);
      uint64_t pos = 0;
      for (uint32_t i = 0; i < n_elems; ++i) {
        if (views[i].len > 0) {
          std::memcpy(blob.get() + pos, views[i].data, views[i].len);
          pos += views[i].len;
        }
      }
      WriteBuffer(base + ".bin", blob.get(), blob_bytes);
      WriteBuffer(base + ".elem_offsets.bin", elem_off.data(),
                  sizeof(uint32_t) * (n_elems + 1));
    } else {
      const uint64_t bytes =
          static_cast<uint64_t>(a.n_elements) *
          ElementSize(IsList(a.type) ? ListElementType(a.type) : a.type);
      WriteBuffer(base + ".bin", a.data, bytes);
    }
    if (IsList(a.type) || is_string) {
      WriteBuffer(base + ".offsets.bin", a.offsets,
                  sizeof(uint32_t) * (a.n_rows + 1));
    }
    if (a.valid != nullptr) {
      WriteBuffer(base + ".valid.bin", a.valid, a.n_rows);
    }
    cols.push_back(m);
  }
  meta["AttributesMetadata"]["columns"] = cols;

  std::ofstream out(dir + "/meta.yaml");
  if (!out) {
    throw std::runtime_error("AttributesIO: cannot write " + dir +
                             "/meta.yaml");
  }
  out << meta << "\n";
}

void AttributesIO::Read(const std::string& dir, Storage* storage) {
  YAML::Node meta;
  try {
    meta = YAML::LoadFile(dir + "/meta.yaml");
  } catch (const YAML::Exception& e) {
    throw std::runtime_error("AttributesIO: cannot parse " + dir +
                             "/meta.yaml: " + e.what());
  }
  const auto root = meta["AttributesMetadata"];
  storage->entity_id = root["entity_id"].as<uint32_t>();
  storage->columns.clear();
  storage->blobs.clear();
  storage->manifest.clear();

  for (const auto& m : root["columns"]) {
    ColumnManifest cm;
    cm.name = m["name"].as<std::string>();
    cm.type = static_cast<ValueType>(m["type"].as<uint32_t>());
    cm.n_rows = m["n_rows"].as<uint32_t>();
    cm.n_elements = m["n_elements"].as<uint32_t>();
    cm.has_offsets = m["has_offsets"].as<bool>();
    cm.has_elem_offsets = m["has_elem_offsets"].as<bool>();
    cm.has_valid = m["has_valid"].as<bool>();

    const std::string base = dir + "/" + cm.name;
    Attribute a{};
    std::snprintf(a.name, sizeof(a.name), "%s", cm.name.c_str());
    a.type = cm.type;
    a.n_rows = cm.n_rows;
    a.n_elements = cm.n_elements;

    if (cm.has_valid) {
      storage->blobs.push_back(ReadBuffer(base + ".valid.bin", cm.n_rows));
      a.valid = storage->blobs.back().get();
    }
    if (cm.has_offsets) {
      storage->blobs.push_back(ReadBuffer(
          base + ".offsets.bin", sizeof(uint32_t) * (cm.n_rows + 1)));
      a.offsets =
          reinterpret_cast<const uint32_t*>(storage->blobs.back().get());
    }

    const bool is_string = cm.type == ValueType::kString;
    const bool is_strlist = cm.type == ValueType::kStringList;
    if (is_string || is_strlist) {
      storage->blobs.push_back(ReadBuffer(
          base + ".elem_offsets.bin",
          sizeof(uint32_t) * (cm.n_elements + 1)));
      const auto* elem_off = reinterpret_cast<const uint32_t*>(
          storage->blobs.back().get());
      storage->blobs.push_back(
          ReadBuffer(base + ".bin", elem_off[cm.n_elements]));
      const char* blob =
          reinterpret_cast<const char*>(storage->blobs.back().get());
      // Rebuild the StringView array.
      auto views = std::make_unique<StringView[]>(
          cm.n_elements > 0 ? cm.n_elements : 1);
      for (uint32_t i = 0; i < cm.n_elements; ++i) {
        views[i].data = blob + elem_off[i];
        views[i].len = elem_off[i + 1] - elem_off[i];
      }
      // StringView is POD; storing the array as a byte blob keeps ownership
      // in one place (the deleter is compatible with trivially-destructible
      // element types).
      storage->blobs.emplace_back(
          reinterpret_cast<uint8_t*>(views.release()));
      a.data = reinterpret_cast<const StringView*>(
          storage->blobs.back().get());
    } else {
      const uint64_t bytes = static_cast<uint64_t>(cm.n_elements) *
                             ElementSize(IsList(cm.type)
                                             ? ListElementType(cm.type)
                                             : cm.type);
      storage->blobs.push_back(ReadBuffer(base + ".bin", bytes));
      a.data = storage->blobs.back().get();
    }

    storage->manifest.push_back(cm);
    storage->columns.push_back(a);
  }
}

}  // namespace io
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics
