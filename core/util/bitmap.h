#ifndef CORE_UTIL_BITMAP_H_
#define CORE_UTIL_BITMAP_H_

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>

namespace sics {
namespace matrixgraph {
namespace core {
namespace util {

#define WORD_OFFSET(i) (i >> 6)
#define BIT_OFFSET(i) (i & 0x3f)

// @DESCRIPTION
//
// Bitmap is a mapping from integers~(indexes) to bits. If the unit is
// occupied, the bit is a nonzero integer constant and if it is empty, the bit
// is zero.
// make sure the pointer is created by new[].
class Bitmap {
 protected:
  Bitmap() = default;

  virtual ~Bitmap() {}

  void Init(size_t size) {
    delete[] data_;
    size_ = size;
    data_ = new uint64_t[WORD_OFFSET(size) + 1]();
  }

  // init data pointer from call function， and own the pointer of data
  // delete pointer in decstructor
  virtual void Init(size_t size, uint64_t* data) {
    delete[] data_;
    size_ = size;
    data_ = data;
  }

 public:
  void Clear() {
    size_t bm_size = WORD_OFFSET(size_);
    for (size_t i = 0; i <= bm_size; i++) data_[i] = 0;
  }

  bool IsEmpty() const {
    size_t bm_size = WORD_OFFSET(size_);
    for (size_t i = 0; i <= bm_size; i++)
      if (data_[i] != 0) return false;
    return true;
  }

  bool IsEqual(Bitmap& b) const {
    if (size_ != b.size_) return false;
    size_t bm_size = WORD_OFFSET(size_);
    for (size_t i = 0; i <= bm_size; i++)
      if (data_[i] != b.data_[i]) return false;
    return true;
  }

  void Fill() {
    size_t bm_size = WORD_OFFSET(size_);
    for (size_t i = 0; i < bm_size; i++) {
      data_[i] = 0xffffffffffffffff;
    }
    data_[bm_size] = 0;
    for (size_t i = (bm_size << 6); i < size_; i++) {
      data_[bm_size] |= 1ul << BIT_OFFSET(i);
    }
  }

  bool GetBit(size_t i) const {
    if (i > size_) return 0;
    return data_[WORD_OFFSET(i)] & (1ul << BIT_OFFSET(i));
  }

  void SetBit(size_t i) {
    if (i > size_) return;
    __sync_fetch_and_or(data_ + WORD_OFFSET(i), 1ul << BIT_OFFSET(i));
  }

  void ClearBit(const size_t i) {
    if (i > size_) return;
    __sync_fetch_and_and(data_ + WORD_OFFSET(i), ~(1ul << BIT_OFFSET(i)));
  }

  size_t Count() const {
    size_t count = 0;
    for (size_t i = 0; i <= WORD_OFFSET(size_); i++) {
      auto x = data_[i];
      x = (x & (0x5555555555555555)) + ((x >> 1) & (0x5555555555555555));
      x = (x & (0x3333333333333333)) + ((x >> 2) & (0x3333333333333333));
      x = (x & (0x0f0f0f0f0f0f0f0f)) + ((x >> 4) & (0x0f0f0f0f0f0f0f0f));
      x = (x & (0x00ff00ff00ff00ff)) + ((x >> 8) & (0x00ff00ff00ff00ff));
      x = (x & (0x0000ffff0000ffff)) + ((x >> 16) & (0x0000ffff0000ffff));
      x = (x & (0x00000000ffffffff)) + ((x >> 32) & (0x00000000ffffffff));
      count += (size_t)x;
    }
    return count;
  }

  size_t PreElementCount(size_t idx) const {
    if (idx > size_) return Count();
    size_t count = 0;
    size_t bm_size = WORD_OFFSET(idx);
    size_t idx_offset = WORD_OFFSET(idx);
    size_t idx_bit_offset = BIT_OFFSET(idx);

    for (size_t i = 0; i <= bm_size; i++) {
      uint64_t x = 0;
      if (i == idx_offset) {
        uint64_t mask = (1ul << idx_bit_offset) - 1;
        x = data_[i] & mask;
      } else {
        x = data_[i];
      }
      x = (x & (0x5555555555555555)) + ((x >> 1) & (0x5555555555555555));
      x = (x & (0x3333333333333333)) + ((x >> 2) & (0x3333333333333333));
      x = (x & (0x0f0f0f0f0f0f0f0f)) + ((x >> 4) & (0x0f0f0f0f0f0f0f0f));
      x = (x & (0x00ff00ff00ff00ff)) + ((x >> 8) & (0x00ff00ff00ff00ff));
      x = (x & (0x0000ffff0000ffff)) + ((x >> 16) & (0x0000ffff0000ffff));
      x = (x & (0x00000000ffffffff)) + ((x >> 32) & (0x00000000ffffffff));
      count += (size_t)x;
    }

    return count;
  }

  size_t size() const { return size_; }

  uint64_t* data() const { return data_; }

  void SetSize(size_t size) { size_ = size; }

  void SetDataPtr(uint64_t* ptr) { data_ = ptr; }

  size_t GetBufferSize() const {
    return sizeof(uint64_t) * (WORD_OFFSET(size_) + 1);
  }

  uint64_t* GetDataBasePointer() const { return data_; }

  uint64_t GetMaxWordOffset() const { return WORD_OFFSET(size_); }

  // @return the fragment of the bitmap at index i
  // @param i the index point of bitmap.
  uint64_t GetFragment(int i) const { return *(data_ + WORD_OFFSET(i)); }

  // @return the pointer of fragment of the bitmap at index i
  // @param i the index point of bitmap.
  uint64_t* GetPFragment(int i) const { return data_ + WORD_OFFSET(i); }

 public:
  size_t size_ = 0;
  uint64_t* data_ = nullptr;
};

}  // namespace util
}  // namespace core
}  // namespace matrixgraph
}  // namespace sics

#endif  // CORE_UTIL_BITMAP_H_