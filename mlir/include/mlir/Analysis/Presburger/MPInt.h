//===- MPInt.h - MLIR MPInt Class -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a simple class to represent arbitrary precision signed integers.
// Unlike APInt, one does not have to specify a fixed maximum size, and the
// integer can take on any arbitrary values. This is optimized for small-values
// by providing fast-paths for the cases when the value stored fits in 64-bits.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_MPINT_H
#define MLIR_ANALYSIS_PRESBURGER_MPINT_H

#include "mlir/Analysis/Presburger/SlowMPInt.h"
#include "mlir/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <unistd.h>
#include <variant>

namespace mlir {
namespace presburger {

/// This class provides support for multi-precision arithmetic.
///
/// Unlike APInt, this extends the precision as necessary to prevent overflows
/// and supports operations between objects with differing internal precisions.
///
/// This is optimized for small-values by providing fast-paths for the cases
/// when the value stored fits in 64-bits. We annotate all fastpaths by using
/// the LLVM_LIKELY/LLVM_UNLIKELY annotations. Removing these would result in
/// a 1.2x performance slowdown.
///
/// We always_inline all operations; removing these results in a 1.5x
/// performance slowdown.
///
/// When holdsSlow is true, a SlowMPInt is held in the union. If it is false,
/// the int64_t is held. Using std::variant instead would significantly impact
/// performance.
class MPInt {
private:
  union {
    int64_t val64;
    detail::SlowMPInt valSlow;
  };
  unsigned holdsSlow;

  __attribute__((always_inline)) void init64(int64_t o) {
    if (LLVM_UNLIKELY(isLarge()))
      valSlow.detail::SlowMPInt::~SlowMPInt();
    val64 = o;
    holdsSlow = false;
  }
  __attribute__((always_inline)) void initAP(const detail::SlowMPInt &o) {
    if (LLVM_LIKELY(isSmall())) {
      // The data in memory could be in an arbitrary state, not necessarily
      // corresponding to any valid state of valSlow; we cannot call any member
      // functions, e.g. the assignment operator on it, as they may access the
      // invalid internal state. We instead construct a new object using
      // placement new.
      new (&valSlow) detail::SlowMPInt(o);
    } else {
      // In this case, we need to use the assignment operator, because if we use
      // placement-new as above we would lose track of allocated memory
      // and leak it.
      valSlow = o;
    }
    holdsSlow = true;
  }

  __attribute__((always_inline)) explicit MPInt(const detail::SlowMPInt &val)
      : valSlow(val), holdsSlow(true) {}
  __attribute__((always_inline)) bool isSmall() const { return !holdsSlow; }
  __attribute__((always_inline)) bool isLarge() const { return holdsSlow; }
  __attribute__((always_inline)) int64_t get64() const {
    assert(isSmall());
    return val64;
  }
  __attribute__((always_inline)) int64_t &get64() {
    assert(isSmall());
    return val64;
  }
  __attribute__((always_inline)) const detail::SlowMPInt &getAP() const {
    assert(isLarge());
    return valSlow;
  }
  __attribute__((always_inline)) detail::SlowMPInt &getAP() {
    assert(isLarge());
    return valSlow;
  }
  explicit operator detail::SlowMPInt() const {
    if (isSmall())
      return detail::SlowMPInt(get64());
    return getAP();
  }
  __attribute__((always_inline)) detail::SlowMPInt getAsAP() const {
    return detail::SlowMPInt(*this);
  }

public:
  __attribute__((always_inline)) explicit MPInt(int64_t val)
      : val64(val), holdsSlow(false) {}
  __attribute__((always_inline)) MPInt() : MPInt(0) {}
  __attribute__((always_inline)) ~MPInt() {
    if (LLVM_UNLIKELY(isLarge()))
      valSlow.detail::SlowMPInt::~SlowMPInt();
  }
  __attribute__((always_inline)) MPInt(const MPInt &o)
      : val64(o.val64), holdsSlow(false) {
    if (LLVM_UNLIKELY(o.isLarge()))
      initAP(o.valSlow);
  }
  __attribute__((always_inline)) MPInt &operator=(const MPInt &o) {
    if (LLVM_LIKELY(o.isSmall())) {
      init64(o.val64);
      return *this;
    }
    initAP(o.valSlow);
    return *this;
  }
  __attribute__((always_inline)) MPInt &operator=(int x) {
    init64(x);
    return *this;
  }
  __attribute__((always_inline)) explicit operator int64_t() const {
    if (isSmall())
      return get64();
    return static_cast<int64_t>(getAP());
  }

  bool operator==(const MPInt &o) const;
  bool operator!=(const MPInt &o) const;
  bool operator>(const MPInt &o) const;
  bool operator<(const MPInt &o) const;
  bool operator<=(const MPInt &o) const;
  bool operator>=(const MPInt &o) const;
  MPInt operator+(const MPInt &o) const;
  MPInt operator-(const MPInt &o) const;
  MPInt operator*(const MPInt &o) const;
  MPInt operator/(const MPInt &o) const;
  MPInt operator%(const MPInt &o) const;
  MPInt &operator+=(const MPInt &o);
  MPInt &operator-=(const MPInt &o);
  MPInt &operator*=(const MPInt &o);
  MPInt &operator/=(const MPInt &o);
  MPInt &operator%=(const MPInt &o);
  MPInt operator-() const;
  MPInt &operator++();
  MPInt &operator--();

  // Divide by a number that is known to be positive.
  // This is slightly more efficient because it saves an overflow check.
  MPInt divByPositive(const MPInt &o) const;
  MPInt &divByPositiveInPlace(const MPInt &o);

  friend MPInt abs(const MPInt &x);
  friend MPInt gcdRange(ArrayRef<MPInt> range);
  friend MPInt ceilDiv(const MPInt &lhs, const MPInt &rhs);
  friend MPInt floorDiv(const MPInt &lhs, const MPInt &rhs);
  friend MPInt gcd(const MPInt &a, const MPInt &b);
  friend MPInt lcm(const MPInt &a, const MPInt &b);
  friend MPInt mod(const MPInt &lhs, const MPInt &rhs);

  llvm::raw_ostream &print(llvm::raw_ostream &os) const;
  void dump() const;

  /// ---------------------------------------------------------------------------
  /// Convenience operator overloads for int64_t.
  /// ---------------------------------------------------------------------------
  friend MPInt &operator+=(MPInt &a, int64_t b);
  friend MPInt &operator-=(MPInt &a, int64_t b);
  friend MPInt &operator*=(MPInt &a, int64_t b);
  friend MPInt &operator/=(MPInt &a, int64_t b);
  friend MPInt &operator%=(MPInt &a, int64_t b);

  friend bool operator==(const MPInt &a, int64_t b);
  friend bool operator!=(const MPInt &a, int64_t b);
  friend bool operator>(const MPInt &a, int64_t b);
  friend bool operator<(const MPInt &a, int64_t b);
  friend bool operator<=(const MPInt &a, int64_t b);
  friend bool operator>=(const MPInt &a, int64_t b);
  friend MPInt operator+(const MPInt &a, int64_t b);
  friend MPInt operator-(const MPInt &a, int64_t b);
  friend MPInt operator*(const MPInt &a, int64_t b);
  friend MPInt operator/(const MPInt &a, int64_t b);
  friend MPInt operator%(const MPInt &a, int64_t b);

  friend bool operator==(int64_t a, const MPInt &b);
  friend bool operator!=(int64_t a, const MPInt &b);
  friend bool operator>(int64_t a, const MPInt &b);
  friend bool operator<(int64_t a, const MPInt &b);
  friend bool operator<=(int64_t a, const MPInt &b);
  friend bool operator>=(int64_t a, const MPInt &b);
  friend MPInt operator+(int64_t a, const MPInt &b);
  friend MPInt operator-(int64_t a, const MPInt &b);
  friend MPInt operator*(int64_t a, const MPInt &b);
  friend MPInt operator/(int64_t a, const MPInt &b);
  friend MPInt operator%(int64_t a, const MPInt &b);

  friend llvm::hash_code hash_value(const MPInt &x); // NOLINT
};

/// This just calls through to the operator int64_t, but it's useful when a
/// function pointer is required. (Although this is marked inline, it is still
/// possible to obtain and use a function pointer to this.)
__attribute__((always_inline)) inline int64_t int64FromMPInt(const MPInt &x) {
  return int64_t(x);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const MPInt &x);

// The RHS is always expected to be positive, and the result
/// is always non-negative.
MPInt mod(const MPInt &lhs, const MPInt &rhs);

/// We define the operations here in the header to facilitate inlining.

/// ---------------------------------------------------------------------------
/// Comparison operators.
/// ---------------------------------------------------------------------------
__attribute__((always_inline)) inline bool
MPInt::operator==(const MPInt &o) const {
  if (LLVM_LIKELY(isSmall() && o.isSmall()))
    return get64() == o.get64();
  return getAsAP() == o.getAsAP();
}
__attribute__((always_inline)) inline bool
MPInt::operator!=(const MPInt &o) const {
  if (LLVM_LIKELY(isSmall() && o.isSmall()))
    return get64() != o.get64();
  return getAsAP() != o.getAsAP();
}
__attribute__((always_inline)) inline bool
MPInt::operator>(const MPInt &o) const {
  if (LLVM_LIKELY(isSmall() && o.isSmall()))
    return get64() > o.get64();
  return getAsAP() > o.getAsAP();
}
__attribute__((always_inline)) inline bool
MPInt::operator<(const MPInt &o) const {
  if (LLVM_LIKELY(isSmall() && o.isSmall()))
    return get64() < o.get64();
  return getAsAP() < o.getAsAP();
}
__attribute__((always_inline)) inline bool
MPInt::operator<=(const MPInt &o) const {
  if (LLVM_LIKELY(isSmall() && o.isSmall()))
    return get64() <= o.get64();
  return getAsAP() <= o.getAsAP();
}
__attribute__((always_inline)) inline bool
MPInt::operator>=(const MPInt &o) const {
  if (LLVM_LIKELY(isSmall() && o.isSmall()))
    return get64() >= o.get64();
  return getAsAP() >= o.getAsAP();
}

/// ---------------------------------------------------------------------------
/// Arithmetic operators.
/// ---------------------------------------------------------------------------
__attribute__((always_inline)) inline MPInt
MPInt::operator+(const MPInt &o) const {
  if (LLVM_LIKELY(isSmall() && o.isSmall())) {
    MPInt result;
    bool overflow = llvm::AddOverflow(get64(), o.get64(), result.get64());
    if (LLVM_LIKELY(!overflow))
      return result;
    return MPInt(getAsAP() + o.getAsAP());
  }
  return MPInt(getAsAP() + o.getAsAP());
}
__attribute__((always_inline)) inline MPInt
MPInt::operator-(const MPInt &o) const {
  if (LLVM_LIKELY(isSmall() && o.isSmall())) {
    MPInt result;
    bool overflow = llvm::SubOverflow(get64(), o.get64(), result.get64());
    if (LLVM_LIKELY(!overflow))
      return result;
    return MPInt(getAsAP() - o.getAsAP());
  }
  return MPInt(getAsAP() - o.getAsAP());
}
__attribute__((always_inline)) inline MPInt
MPInt::operator*(const MPInt &o) const {
  if (LLVM_LIKELY(isSmall() && o.isSmall())) {
    MPInt result;
    bool overflow = llvm::MulOverflow(get64(), o.get64(), result.get64());
    if (LLVM_LIKELY(!overflow))
      return result;
    return MPInt(getAsAP() * o.getAsAP());
  }
  return MPInt(getAsAP() * o.getAsAP());
}

__attribute__((always_inline)) inline MPInt
MPInt::divByPositive(const MPInt &o) const {
  assert(o > 0);
  if (LLVM_LIKELY(isSmall() && o.isSmall()))
    return MPInt(get64() / o.get64());
  return MPInt(getAsAP() / o.getAsAP());
}

__attribute__((always_inline)) inline MPInt
MPInt::operator/(const MPInt &o) const {
  if (LLVM_LIKELY(isSmall() && o.isSmall())) {
    if (LLVM_UNLIKELY(o.get64() == -1))
      return -*this;
    return MPInt(get64() / o.get64());
  }
  return MPInt(getAsAP() / o.getAsAP());
}

inline MPInt abs(const MPInt &x) { return MPInt(x >= 0 ? x : -x); }
inline MPInt ceilDiv(const MPInt &lhs, const MPInt &rhs) {
  if (LLVM_LIKELY(lhs.isSmall() && rhs.isSmall())) {
    if (rhs == -1)
      return -lhs;
    int64_t x = (rhs.get64() > 0) ? -1 : 1;
    return MPInt(((lhs.get64() != 0) && (lhs.get64() > 0) == (rhs.get64() > 0))
                     ? ((lhs.get64() + x) / rhs.get64()) + 1
                     : -(-lhs.get64() / rhs.get64()));
  }
  return MPInt(ceilDiv(lhs.getAsAP(), rhs.getAsAP()));
}
inline MPInt floorDiv(const MPInt &lhs, const MPInt &rhs) {
  if (LLVM_LIKELY(lhs.isSmall() && rhs.isSmall())) {
    if (rhs == -1)
      return -lhs;
    int64_t x = (rhs.get64() < 0) ? 1 : -1;
    return MPInt(
        ((lhs.get64() != 0) && ((lhs.get64() < 0) != (rhs.get64() < 0)))
            ? -((-lhs.get64() + x) / rhs.get64()) - 1
            : lhs.get64() / rhs.get64());
  }
  return MPInt(floorDiv(lhs.getAsAP(), rhs.getAsAP()));
}
// The RHS is always expected to be positive, and the result
/// is always non-negative.
inline MPInt mod(const MPInt &lhs, const MPInt &rhs) {
  if (LLVM_LIKELY(lhs.isSmall() && rhs.isSmall()))
    return MPInt(lhs.get64() % rhs.get64() < 0
                     ? lhs.get64() % rhs.get64() + rhs.get64()
                     : lhs.get64() % rhs.get64());
  return MPInt(mod(lhs.getAsAP(), rhs.getAsAP()));
}

__attribute__((always_inline)) inline MPInt gcd(const MPInt &a,
                                                const MPInt &b) {
  // TODO: fix unsigned/signed overflow issues
  if (LLVM_LIKELY(a.isSmall() && b.isSmall()))
    return MPInt(int64_t(llvm::GreatestCommonDivisor64(a.get64(), b.get64())));
  return MPInt(gcd(a.getAsAP(), b.getAsAP()));
}

/// Returns the least common multiple of 'a' and 'b'.
inline MPInt lcm(const MPInt &a, const MPInt &b) {
  MPInt x = abs(a);
  MPInt y = abs(b);
  return (x * y) / gcd(x, y);
}

/// This operation cannot overflow.
inline MPInt MPInt::operator%(const MPInt &o) const {
  if (LLVM_LIKELY(isSmall() && o.isSmall()))
    return MPInt(get64() % o.get64());
  return MPInt(getAsAP() % o.getAsAP());
}

inline MPInt MPInt::operator-() const {
  if (LLVM_LIKELY(isSmall())) {
    if (LLVM_LIKELY(get64() != std::numeric_limits<int64_t>::min()))
      return MPInt(-get64());
    return MPInt(-getAsAP());
  }
  return MPInt(-getAsAP());
}

/// ---------------------------------------------------------------------------
/// Assignment operators, preincrement, predecrement.
/// ---------------------------------------------------------------------------
__attribute__((always_inline)) inline MPInt &MPInt::operator+=(const MPInt &o) {
  if (LLVM_LIKELY(isSmall() && o.isSmall())) {
    int64_t result = get64();
    bool overflow = llvm::AddOverflow(get64(), o.get64(), result);
    if (LLVM_LIKELY(!overflow)) {
      get64() = result;
      return *this;
    }
    return *this = MPInt(getAsAP() + o.getAsAP());
  }
  return *this = MPInt(getAsAP() + o.getAsAP());
}
__attribute__((always_inline)) inline MPInt &MPInt::operator-=(const MPInt &o) {
  if (LLVM_LIKELY(isSmall() && o.isSmall())) {
    int64_t result = get64();
    bool overflow = llvm::SubOverflow(get64(), o.get64(), result);
    if (LLVM_LIKELY(!overflow)) {
      get64() = result;
      return *this;
    }
    return *this = MPInt(getAsAP() - o.getAsAP());
  }
  return *this = MPInt(getAsAP() - o.getAsAP());
}
__attribute__((always_inline)) inline MPInt &MPInt::operator*=(const MPInt &o) {
  if (LLVM_LIKELY(isSmall() && o.isSmall())) {
    int64_t result = get64();
    bool overflow = llvm::MulOverflow(get64(), o.get64(), result);
    if (LLVM_LIKELY(!overflow)) {
      get64() = result;
      return *this;
    }
    return *this = MPInt(getAsAP() * o.getAsAP());
  }
  return *this = MPInt(getAsAP() * o.getAsAP());
}
__attribute__((always_inline)) inline MPInt &MPInt::operator/=(const MPInt &o) {
  if (LLVM_LIKELY(isSmall() && o.isSmall())) {
    if (LLVM_UNLIKELY(o.get64() == -1))
      return *this = -*this;
    get64() /= o.get64();
    return *this;
  }
  return *this = MPInt(getAsAP() / o.getAsAP());
}

__attribute__((always_inline)) inline MPInt &
MPInt::divByPositiveInPlace(const MPInt &o) {
  assert(o > 0);
  if (LLVM_LIKELY(isSmall() && o.isSmall())) {
    get64() /= o.get64();
    return *this;
  }
  return *this = MPInt(getAsAP() / o.getAsAP());
}

__attribute__((always_inline)) inline MPInt &MPInt::operator%=(const MPInt &o) {
  *this = *this % o;
  return *this;
}
__attribute__((always_inline)) inline MPInt &MPInt::operator++() {
  *this += 1;
  return *this;
}
__attribute__((always_inline)) inline MPInt &MPInt::operator--() {
  *this -= 1;
  return *this;
}

/// ----------------------------------------------------------------------------
/// Convenience operator overloads for int64_t.
/// ----------------------------------------------------------------------------
__attribute__((always_inline)) inline MPInt &operator+=(MPInt &a, int64_t b) {
  return a = a + b;
}
__attribute__((always_inline)) inline MPInt &operator-=(MPInt &a, int64_t b) {
  return a = a - b;
}
__attribute__((always_inline)) inline MPInt &operator*=(MPInt &a, int64_t b) {
  return a = a * b;
}
__attribute__((always_inline)) inline MPInt &operator/=(MPInt &a, int64_t b) {
  return a = a / b;
}
__attribute__((always_inline)) inline MPInt &operator%=(MPInt &a, int64_t b) {
  return a = a % b;
}
inline MPInt operator+(const MPInt &a, int64_t b) { return a + MPInt(b); }
inline MPInt operator-(const MPInt &a, int64_t b) { return a - MPInt(b); }
inline MPInt operator*(const MPInt &a, int64_t b) { return a * MPInt(b); }
inline MPInt operator/(const MPInt &a, int64_t b) { return a / MPInt(b); }
inline MPInt operator%(const MPInt &a, int64_t b) { return a % MPInt(b); }
inline MPInt operator+(int64_t a, const MPInt &b) { return MPInt(a) + b; }
inline MPInt operator-(int64_t a, const MPInt &b) { return MPInt(a) - b; }
inline MPInt operator*(int64_t a, const MPInt &b) { return MPInt(a) * b; }
inline MPInt operator/(int64_t a, const MPInt &b) { return MPInt(a) / b; }
inline MPInt operator%(int64_t a, const MPInt &b) { return MPInt(a) % b; }

/// We provide special implementations of the comparison operators rather than
/// calling through as above, as this would result in a 1.2x slowdown.
inline bool operator==(const MPInt &a, int64_t b) {
  if (LLVM_LIKELY(a.isSmall()))
    return a.get64() == b;
  return a.getAP() == b;
}
inline bool operator!=(const MPInt &a, int64_t b) {
  if (LLVM_LIKELY(a.isSmall()))
    return a.get64() != b;
  return a.getAP() != b;
}
inline bool operator>(const MPInt &a, int64_t b) {
  if (LLVM_LIKELY(a.isSmall()))
    return a.get64() > b;
  return a.getAP() > b;
}
inline bool operator<(const MPInt &a, int64_t b) {
  if (LLVM_LIKELY(a.isSmall()))
    return a.get64() < b;
  return a.getAP() < b;
}
inline bool operator<=(const MPInt &a, int64_t b) {
  if (LLVM_LIKELY(a.isSmall()))
    return a.get64() <= b;
  return a.getAP() <= b;
}
inline bool operator>=(const MPInt &a, int64_t b) {
  if (LLVM_LIKELY(a.isSmall()))
    return a.get64() >= b;
  return a.getAP() >= b;
}
inline bool operator==(int64_t a, const MPInt &b) {
  if (LLVM_LIKELY(b.isSmall()))
    return a == b.get64();
  return a == b.getAP();
}
inline bool operator!=(int64_t a, const MPInt &b) {
  if (LLVM_LIKELY(b.isSmall()))
    return a != b.get64();
  return a != b.getAP();
}
inline bool operator>(int64_t a, const MPInt &b) {
  if (LLVM_LIKELY(b.isSmall()))
    return a > b.get64();
  return a > b.getAP();
}
inline bool operator<(int64_t a, const MPInt &b) {
  if (LLVM_LIKELY(b.isSmall()))
    return a < b.get64();
  return a < b.getAP();
}
inline bool operator<=(int64_t a, const MPInt &b) {
  if (LLVM_LIKELY(b.isSmall()))
    return a <= b.get64();
  return a <= b.getAP();
}
inline bool operator>=(int64_t a, const MPInt &b) {
  if (LLVM_LIKELY(b.isSmall()))
    return a >= b.get64();
  return a >= b.getAP();
}

} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_MPINT_H
