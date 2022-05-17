//===- MPInt.h - MLIR MPInt Class -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a simple class to represent arbitrary precision signed integers.
// Unlike APInt2, one does not have to specify a fixed maximum size, and the
// integer can take on any aribtrary values.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_MPINT_H
#define MLIR_ANALYSIS_PRESBURGER_MPINT_H

#include "mlir/Analysis/Presburger/APInt2.h"
#include "mlir/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <unistd.h>
#include <variant>

namespace mlir {
namespace presburger {

/// This class provides support for multi-precision arithmetic.
///
/// Unlike APInt2, this extends the precision as necessary to prevent overflows
/// and supports operations between objects with differing internal precisions.
///
/// Since it uses APInt2 internally, MPInt (MultiPrecision Integer) stores
/// values in a 64-bit machine integer for small values and uses slower
/// arbitrary-precision arithmetic only for larger values.
class MPInt {
public:
  __attribute__((always_inline)) explicit MPInt(int64_t val)
      : val64(val), holdsAP(false) {}
  __attribute__((always_inline)) MPInt() : MPInt(0) {}
  __attribute__((always_inline)) ~MPInt() {
    if (isLarge()) [[unlikely]]
      valAP.~APInt();
  }
  __attribute__((always_inline)) MPInt(const MPInt &o)
      : val64(o.val64), holdsAP(false) {
    if (o.isLarge()) [[unlikely]]
      initAP(o.valAP);
  }
  __attribute__((always_inline)) MPInt &operator=(const MPInt &o) {
    if (o.isSmall()) [[likely]] {
      init64(o.val64);
      return *this;
    }
    initAP(o.valAP);
    return *this;
  }
  __attribute__((always_inline)) MPInt &operator=(int x) {
    init64(x);
    return *this;
  }
  __attribute__((always_inline))
  explicit operator int64_t() const {
    if (isSmall())
      return get64();
    abort();
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

  friend MPInt abs(const MPInt &x);
  friend MPInt ceilDiv(const MPInt &lhs, const MPInt &rhs);
  friend MPInt floorDiv(const MPInt &lhs, const MPInt &rhs);
  friend MPInt gcd(const MPInt &a, const MPInt &b);
  friend MPInt mod(const MPInt &lhs, const MPInt &rhs);

  llvm::raw_ostream &print(llvm::raw_ostream &os) const;
  void dump() const;

  /// ---------------------------------------------------------------------------
  /// Convenience operator overloads for int64_t.
  /// ---------------------------------------------------------------------------
  friend MPInt &operator+=(MPInt &a, int64_t b) {
    if (a.isSmall()) {
      a.get64() += b;
      return a;
    }
    abort();
  }
  friend MPInt &operator-=(MPInt &a, int64_t b) {
    if (a.isSmall()) {
      a.get64() -= b;
      return a;
    }
    abort();
  }
  friend MPInt &operator*=(MPInt &a, int64_t b) {
    if (a.isSmall()) {
      a.get64() *= b;
      return a;
    }
    abort();
  }
  friend MPInt &operator/=(MPInt &a, int64_t b) {
    if (a.isSmall()) {
      a.get64() /= b;
      return a;
    }
    abort();
  }
  friend MPInt &operator%=(MPInt &a, int64_t b) {
    if (a.isSmall()) {
      a.get64() %= b;
      return a;
    }
    abort();
  }

  friend bool operator==(const MPInt &a, int64_t b) {
    if (a.isSmall())
      return a.get64() == b;
    abort();
  }
  friend bool operator!=(const MPInt &a, int64_t b) {
    if (a.isSmall())
      return a.get64() != b;
    abort();
  }
  friend bool operator>(const MPInt &a, int64_t b) {
    if (a.isSmall())
      return a.get64() > b;
    abort();
  }
  friend bool operator<(const MPInt &a, int64_t b) {
    if (a.isSmall())
      return a.get64() < b;
    abort();
  }
  friend bool operator<=(const MPInt &a, int64_t b) {
    if (a.isSmall())
      return a.get64() <= b;
    abort();
  }
  friend bool operator>=(const MPInt &a, int64_t b) {
    if (a.isSmall())
      return a.get64() >= b;
    abort();
  }
  friend MPInt operator+(const MPInt &a, int64_t b) {
    if (a.isSmall())
      return MPInt(a.get64() + b);
    abort();
  }
  friend MPInt operator-(const MPInt &a, int64_t b) {
    if (a.isSmall())
      return MPInt(a.get64() - b);
    abort();
  }
  friend MPInt operator*(const MPInt &a, int64_t b) {
    if (a.isSmall())
      return MPInt(a.get64() * b);
    abort();
  }
  friend MPInt operator/(const MPInt &a, int64_t b) {
    if (a.isSmall())
      return MPInt(a.get64() / b);
    abort();
  }
  friend MPInt operator%(const MPInt &a, int64_t b) {
    if (a.isSmall())
      return MPInt(a.get64() % b);
    abort();
  }

  friend bool operator==(int64_t a, const MPInt &b) {
    if (b.isSmall())
      return a == b.get64();
    abort();
  }
  friend bool operator!=(int64_t a, const MPInt &b) {
    if (b.isSmall())
      return a != b.get64();
    abort();
  }
  friend bool operator>(int64_t a, const MPInt &b) {
    if (b.isSmall())
      return a > b.get64();
    abort();
  }
  friend bool operator<(int64_t a, const MPInt &b) {
    if (b.isSmall())
      return a < b.get64();
    abort();
  }
  friend bool operator<=(int64_t a, const MPInt &b) {
    if (b.isSmall())
      return a <= b.get64();
    abort();
  }
  friend bool operator>=(int64_t a, const MPInt &b) {
    if (b.isSmall())
      return a >= b.get64();
    abort();
  }
  friend MPInt operator+(int64_t a, const MPInt &b) {
    if (b.isSmall())
      return MPInt(a + b.get64());
    abort();
  }
  friend MPInt operator-(int64_t a, const MPInt &b) {
    if (b.isSmall())
      return MPInt(a - b.get64());
    abort();
  }
  friend MPInt operator*(int64_t a, const MPInt &b) {
    if (b.isSmall())
      return MPInt(a * b.get64());
    abort();
  }
  friend MPInt operator/(int64_t a, const MPInt &b) {
    if (b.isSmall())
      return MPInt(a / b.get64());
    abort();
  }
  friend MPInt operator%(int64_t a, const MPInt &b) {
    if (b.isSmall())
      return MPInt(a % b.get64());
    abort();
  }

  friend llvm::hash_code hash_value(const MPInt &x); // NOLINT

private:
  __attribute__((always_inline)) bool isSmall() const { return !holdsAP; }
  __attribute__((always_inline)) bool isLarge() const { return holdsAP; }
  __attribute__((always_inline)) int64_t get64() const {
    assert(isSmall());
    return val64;
  }
  __attribute__((always_inline)) int64_t &get64() {
    assert(isSmall());
    return val64;
  }
  __attribute__((always_inline)) const APInt &getAP() const {
    assert(isLarge());
    return valAP;
  }
  __attribute__((always_inline)) APInt getAP() {
    assert(isLarge());
    return valAP;
  }

  union {
    int64_t val64;
    APInt valAP;
  };
  bool holdsAP;

  __attribute__((always_inline)) void init64(int64_t o) {
    val64 = o;
    holdsAP = false;
  }
  __attribute__((always_inline)) void initAP(const APInt &o) {
    valAP = o;
    holdsAP = true;
  }
};

/// This just calls through to the operator int64_t, but it's useful when a
/// function pointer is required.
__attribute__((always_inline)) inline int64_t int64FromMPInt(const MPInt &x) {
  return int64_t(x);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const MPInt &x);

// The RHS is always expected to be positive, and the result
/// is always non-negative.
MPInt mod(const MPInt &lhs, const MPInt &rhs);

/// Returns the least common multiple of 'a' and 'b'.
MPInt lcm(const MPInt &a, const MPInt &b);

/// We define the operations here in the header to facilitate inlining.

/// ---------------------------------------------------------------------------
/// Comparison operators.
/// ---------------------------------------------------------------------------
inline bool MPInt::operator==(const MPInt &o) const {
  if (isSmall() && o.isSmall())
    return get64() == o.get64();
  abort();
}
inline bool MPInt::operator!=(const MPInt &o) const {
  if (isSmall() && o.isSmall())
    return get64() != o.get64();
  abort();
}
inline bool MPInt::operator>(const MPInt &o) const {
  if (isSmall() && o.isSmall())
    return get64() > o.get64();
  abort();
}
inline bool MPInt::operator<(const MPInt &o) const {
  if (isSmall() && o.isSmall())
    return get64() < o.get64();
  abort();
}
inline bool MPInt::operator<=(const MPInt &o) const {
  if (isSmall() && o.isSmall())
    return get64() <= o.get64();
  abort();
}
inline bool MPInt::operator>=(const MPInt &o) const {
  if (isSmall() && o.isSmall())
    return get64() >= o.get64();
  abort();
}

/// ---------------------------------------------------------------------------
/// Arithmetic operators.
/// ---------------------------------------------------------------------------
inline MPInt MPInt::operator+(const MPInt &o) const {
  if (isSmall() && o.isSmall()) {
    MPInt result;
    bool overflow = __builtin_add_overflow(get64(), o.get64(), &result.get64());
    if (overflow)
      std::abort();
    return result;
  }
  abort();
}
inline MPInt MPInt::operator-(const MPInt &o) const {
  if (isSmall() && o.isSmall()) {
    MPInt result;
    bool overflow = __builtin_sub_overflow(get64(), o.get64(), &result.get64());
    if (overflow)
      std::abort();
    return result;
  }
  abort();
}
inline MPInt MPInt::operator*(const MPInt &o) const {
  if (isSmall() && o.isSmall()) {
    MPInt result;
    bool overflow = __builtin_mul_overflow(get64(), o.get64(), &result.get64());
    if (overflow)
      std::abort();
    return result;
  }
  abort();
}
inline MPInt MPInt::operator/(const MPInt &o) const {
  if (isSmall() && o.isSmall()) {
    if (o.get64() == -1)
      return -*this;
    return MPInt(get64() / o.get64());
  }
  abort();
}
inline MPInt abs(const MPInt &x) { return MPInt(x >= 0 ? x : -x); }
inline MPInt ceilDiv(const MPInt &lhs, const MPInt &rhs) {
  if (lhs.isSmall() && rhs.isSmall()) {
    if (rhs == -1)
      return -lhs;
    int64_t x = (rhs.get64() > 0) ? -1 : 1;
    return MPInt(((lhs.get64() != 0) && (lhs.get64() > 0) == (rhs.get64() > 0))
                     ? ((lhs.get64() + x) / rhs.get64()) + 1
                     : -(-lhs.get64() / rhs.get64()));
  }
  abort();
}
inline MPInt floorDiv(const MPInt &lhs, const MPInt &rhs) {
  if (lhs.isSmall() && rhs.isSmall()) {
    if (rhs == -1)
      return -lhs;
    int64_t x = (rhs.get64() < 0) ? 1 : -1;
    return MPInt(
        ((lhs.get64() != 0) && ((lhs.get64() < 0) != (rhs.get64() < 0)))
            ? -((-lhs.get64() + x) / rhs.get64()) - 1
            : lhs.get64() / rhs.get64());
  }
  abort();
}
// The RHS is always expected to be positive, and the result
/// is always non-negative.
inline MPInt mod(const MPInt &lhs, const MPInt &rhs) {
  if (lhs.isSmall() && rhs.isSmall())
    return MPInt(lhs.get64() % rhs.get64() < 0
                     ? lhs.get64() % rhs.get64() + rhs.get64()
                     : lhs.get64() % rhs.get64());
  abort();
}

inline MPInt gcd(const MPInt &a, const MPInt &b) {
  if (a.isSmall() && b.isSmall())
    return MPInt(llvm::GreatestCommonDivisor64(a.get64(), b.get64()));
  abort();
}

/// Returns the least common multiple of 'a' and 'b'.
inline MPInt lcm(const MPInt &a, const MPInt &b) {
  MPInt x = abs(a);
  MPInt y = abs(b);
  return (x * y) / gcd(x, y);
}

/// This operation cannot overflow.
inline MPInt MPInt::operator%(const MPInt &o) const {
  if (isSmall() && o.isSmall())
    return MPInt(get64() % o.get64());
  abort();
  // unsigned widthThis = get64().getBitWidth();
  // unsigned widthOther = o.get64().getBitWidth();
  // if (widthThis == widthOther)
  //   return MPInt(get64().srem(o.get64()));
  // if (widthThis < widthOther)
  //   return MPInt(get64().sext(widthOther).srem(o.get64()));
  // return MPInt(get64().srem(o.get64().sext(widthThis)));
}

inline MPInt MPInt::operator-() const {
  if (isSmall()) {
    if (get64() != std::numeric_limits<int64_t>::min())
      return MPInt(-get64());
    abort();
  }
  abort();
}

/// ---------------------------------------------------------------------------
/// Assignment operators, preincrement, predecrement.
/// ---------------------------------------------------------------------------
inline MPInt &MPInt::operator+=(const MPInt &o) {
  *this = *this + o;
  return *this;
}
inline MPInt &MPInt::operator-=(const MPInt &o) {
  *this = *this - o;
  return *this;
}
inline MPInt &MPInt::operator*=(const MPInt &o) {
  *this = *this * o;
  return *this;
}
inline MPInt &MPInt::operator/=(const MPInt &o) {
  *this = *this / o;
  return *this;
}
inline MPInt &MPInt::operator%=(const MPInt &o) {
  *this = *this % o;
  return *this;
}
inline MPInt &MPInt::operator++() {
  *this += 1;
  return *this;
}

inline MPInt &MPInt::operator--() {
  *this -= 1;
  return *this;
}

} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_MPINT_H
