/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 * SPDX-FileCopyrightText: 2019 Tobias Grosser <tobias@grosser.es>
 * SPDX-FileCopyrightText: 2019 Maximilian Falkenstein <falkensm@ethz.ch>
 */

#include <cstdlib>
#include <new>

#ifndef LIBINT_ALIGNED_ALLOCATOR_H
#define LIBINT_ALIGNED_ALLOCATOR_H

namespace mlir {
namespace analysis {
namespace presburger {

// A C++17 aligned memory allocator
template <class T, size_t Alignment>
struct AlignedAllocator {
  typedef T value_type;

  // The rebind is not needed from C++17 onwards, but it seems earlier stls
  // still require it. Hence, leave it in for now.
  template <typename U>
  struct rebind {
    typedef AlignedAllocator<U, Alignment> other;
  };

  constexpr unsigned to_next_alignment_multiple(unsigned alignment,
                                                unsigned size) {
    if ((size % alignment) == 0) {
      return size;
    } else {
      return size - (size % alignment) + alignment;
    }
  }

  AlignedAllocator() = default;
  template <class U, size_t Y>
  constexpr AlignedAllocator(const AlignedAllocator<U, Y> &) noexcept {}
  T *allocate(std::size_t Elements) {
    if (auto p = static_cast<T *>(aligned_alloc(
            Alignment,
            to_next_alignment_multiple(Alignment, Elements * sizeof(T)))))
      return p;
    std::abort();
  }
  void deallocate(T *Pointer, std::size_t) noexcept { std::free(Pointer); }
};
template <class T, class U, size_t X, size_t Y>
bool operator==(const AlignedAllocator<T, X> &,
                const AlignedAllocator<U, Y> &) {
  return true;
}
template <class T, class U, size_t X, size_t Y>
bool operator!=(const AlignedAllocator<T, X> &,
                const AlignedAllocator<U, Y> &) {
  return false;
}

} // namespace presburger
} // namespace analysis
} // namespace mlir

#endif // LIBINT_ALIGNED_ALLOCATOR_H
