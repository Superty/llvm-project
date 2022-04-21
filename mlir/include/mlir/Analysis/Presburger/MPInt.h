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

#include "mlir/Support/MathExtras.h"
#include "mlir/Analysis/Presburger/APInt2.h"
#include "llvm/Support/raw_ostream.h"
#include <unistd.h>

namespace mlir {
namespace presburger {

/// This class provides support for multi-precision arithmetic.
///
/// Unlike APInt2, this extends the precision as necessary to prevent overflows
/// and supports operations between objects with differing internal precisions.
///
/// Since it uses APInt2 internally, MPInt (MultiPrecision Integer) stores values
/// in a 64-bit machine integer for small values and uses slower
/// arbitrary-precision arithmetic only for larger values.
class MPInt {
public:
  explicit MPInt(int64_t val) : val(val), isLarge(false) {}
  MPInt() : MPInt(0) {}
  MPInt operator-() const;
  MPInt &operator=(int x) {
    val = x;
    return *this;
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

  MPInt &operator++();
  MPInt &operator--();

  explicit operator int64_t() const { return val; }
  friend MPInt abs(const MPInt &x);
  friend MPInt ceilDiv(const MPInt &lhs, const MPInt &rhs);
  friend MPInt floorDiv(const MPInt &lhs, const MPInt &rhs);
  friend MPInt greatestCommonDivisor(const MPInt &a, const MPInt &b);
  friend MPInt mod(const MPInt &lhs, const MPInt &rhs);

  llvm::raw_ostream &print(llvm::raw_ostream &os) const;
  void dump() const;

  /// ---------------------------------------------------------------------------
  /// Convenience operator overloads for int64_t.
  /// ---------------------------------------------------------------------------
  friend MPInt &operator+=(MPInt &a, int64_t b) { if (a.isLarge) { abort(); } a.val += b; return a; }
  friend MPInt &operator-=(MPInt &a, int64_t b) { if (a.isLarge) { abort(); } a.val -= b; return a; }
  friend MPInt &operator*=(MPInt &a, int64_t b) { if (a.isLarge) { abort(); } a.val *= b; return a; }
  friend MPInt &operator/=(MPInt &a, int64_t b) { if (a.isLarge) { abort(); } a.val /= b; return a; }
  friend MPInt &operator%=(MPInt &a, int64_t b) { if (a.isLarge) { abort(); } a.val %= b; return a; }

  friend bool operator==(const MPInt &a, int64_t b) { if (a.isLarge) { abort(); } return a.val == b; }
  friend bool operator!=(const MPInt &a, int64_t b) { if (a.isLarge) { abort(); } return a.val != b; }
  friend bool operator>(const MPInt &a, int64_t b) { if (a.isLarge) { abort(); } return a.val > b; }
  friend bool operator<(const MPInt &a, int64_t b) { if (a.isLarge) { abort(); } return a.val < b; }
  friend bool operator<=(const MPInt &a, int64_t b) { if (a.isLarge) { abort(); } return a.val <= b; }
  friend bool operator>=(const MPInt &a, int64_t b) { if (a.isLarge) { abort(); } return a.val >= b; }
  friend MPInt operator+(const MPInt &a, int64_t b) { if (a.isLarge) { abort(); } return MPInt(a.val + b); }
  friend MPInt operator-(const MPInt &a, int64_t b) { if (a.isLarge) { abort(); } return MPInt(a.val - b); }
  friend MPInt operator*(const MPInt &a, int64_t b) { if (a.isLarge) { abort(); } return MPInt(a.val * b); }
  friend MPInt operator/(const MPInt &a, int64_t b) { if (a.isLarge) { abort(); } return MPInt(a.val / b); }
  friend MPInt operator%(const MPInt &a, int64_t b) { if (a.isLarge) { abort(); } return MPInt(a.val % b); }

  friend bool operator==(int64_t a, const MPInt &b) { if (b.isLarge) { abort(); } return a == b.val; }
  friend bool operator!=(int64_t a, const MPInt &b) { if (b.isLarge) { abort(); } return a != b.val; }
  friend bool operator>(int64_t a, const MPInt &b) { if (b.isLarge) { abort(); } return a > b.val; }
  friend bool operator<(int64_t a, const MPInt &b) { if (b.isLarge) { abort(); } return a < b.val; }
  friend bool operator<=(int64_t a, const MPInt &b) { if (b.isLarge) { abort(); } return a <= b.val; }
  friend bool operator>=(int64_t a, const MPInt &b) { if (b.isLarge) { abort(); } return a >= b.val; }
  friend MPInt operator+(int64_t a, const MPInt &b) { if (b.isLarge) { abort(); } return MPInt(a + b.val); }
  friend MPInt operator-(int64_t a, const MPInt &b) { if (b.isLarge) { abort(); } return MPInt(a - b.val); }
  friend MPInt operator*(int64_t a, const MPInt &b) { if (b.isLarge) { abort(); } return MPInt(a * b.val); }
  friend MPInt operator/(int64_t a, const MPInt &b) { if (b.isLarge) { abort(); } return MPInt(a / b.val); }
  friend MPInt operator%(int64_t a, const MPInt &b) { if (b.isLarge) { abort(); } return MPInt(a % b.val); }

  friend llvm::hash_code hash_value(const MPInt &x); // NOLINT

private:
  int64_t val;
  bool isLarge;
};

/// This just calls through to the operator int64_t, but it's useful when a
/// function pointer is required.
inline int64_t int64FromMPInt(const MPInt &x) { return int64_t(x); }

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
  if (!isLarge && !o.isLarge)
    return val == o.val;
  abort();
}
inline bool MPInt::operator!=(const MPInt &o) const {
  if (!isLarge && !o.isLarge)
    return val != o.val;
  abort();
}
inline bool MPInt::operator>(const MPInt &o) const {
  if (!isLarge && !o.isLarge)
    return val > o.val;
  abort();
}
inline bool MPInt::operator<(const MPInt &o) const {
  if (!isLarge && !o.isLarge)
    return val < o.val;
  abort();
}
inline bool MPInt::operator<=(const MPInt &o) const {
  if (!isLarge && !o.isLarge)
    return val <= o.val;
  abort();
}
inline bool MPInt::operator>=(const MPInt &o) const {
  if (!isLarge && !o.isLarge)
    return val >= o.val;
  abort();
}

/// ---------------------------------------------------------------------------
/// Arithmetic operators.
/// ---------------------------------------------------------------------------
inline MPInt MPInt::operator+(const MPInt &o) const {
  if (!isLarge && !o.isLarge) {
    MPInt result;
    bool overflow = __builtin_add_overflow(val, o.val, &result.val);
    if (overflow)
      std::abort();
    return result;
  }
  abort();
}
inline MPInt MPInt::operator-(const MPInt &o) const {
  if (!isLarge && !o.isLarge) {
    MPInt result;
    bool overflow = __builtin_sub_overflow(val, o.val, &result.val);
    if (overflow)
      std::abort();
    return result;
  }
  abort();
}
inline MPInt MPInt::operator*(const MPInt &o) const {
  if (!isLarge && !o.isLarge) {
    MPInt result;
    bool overflow = __builtin_mul_overflow(val, o.val, &result.val);
    if (overflow)
      std::abort();
    return result;
  }
  abort();
}
inline MPInt MPInt::operator/(const MPInt &o) const {
  if (!isLarge && !o.isLarge) {
    if (o.val == -1)
      return -*this;
    return MPInt(val / o.val);
  }
  abort();
}
inline MPInt abs(const MPInt &x) { return MPInt(x >= 0 ? x : -x); }
inline MPInt ceilDiv(const MPInt &lhs, const MPInt &rhs) {
  if (!lhs.isLarge && !rhs.isLarge) {
    if (rhs == -1)
      return -lhs;
    int64_t x = (rhs.val > 0) ? -1 : 1;
    return MPInt(((lhs.val != 0) && (lhs.val > 0) == (rhs.val > 0)) ? ((lhs.val + x) / rhs.val) + 1
                                                  : -(-lhs.val / rhs.val));
  }
  abort();
}
inline MPInt floorDiv(const MPInt &lhs, const MPInt &rhs) {
  if (!lhs.isLarge && !rhs.isLarge) {
    if (rhs == -1)
      return -lhs;
    int64_t x = (rhs.val < 0) ? 1 : -1;
    return MPInt(((lhs.val != 0) && ((lhs.val < 0) != (rhs.val < 0))) ? -((-lhs.val + x) / rhs.val) - 1
                                                    : lhs.val / rhs.val);
  }
  abort();
}
// The RHS is always expected to be positive, and the result
/// is always non-negative.
inline MPInt mod(const MPInt &lhs, const MPInt &rhs) {
  if (!lhs.isLarge && !rhs.isLarge)
    return MPInt(lhs.val % rhs.val < 0 ? lhs.val % rhs.val + rhs.val : lhs.val % rhs.val);
  abort();
}

inline MPInt greatestCommonDivisor(const MPInt &a, const MPInt &b) {
  if (!a.isLarge && !b.isLarge)
    return MPInt(llvm::GreatestCommonDivisor64(a.val, b.val));
  abort();
}

/// Returns the least common multiple of 'a' and 'b'.
inline MPInt lcm(const MPInt &a, const MPInt &b) {
  MPInt x = abs(a);
  MPInt y = abs(b);
  return (x * y) / greatestCommonDivisor(x, y);
}

/// This operation cannot overflow.
inline MPInt MPInt::operator%(const MPInt &o) const {
  if (!isLarge && !o.isLarge)
    return MPInt(val % o.val);
  abort();
  // unsigned widthThis = val.getBitWidth();
  // unsigned widthOther = o.val.getBitWidth();
  // if (widthThis == widthOther)
  //   return MPInt(val.srem(o.val));
  // if (widthThis < widthOther)
  //   return MPInt(val.sext(widthOther).srem(o.val));
  // return MPInt(val.srem(o.val.sext(widthThis)));
}

inline MPInt MPInt::operator-() const {
  if (!isLarge) {
    if (val == std::numeric_limits<int64_t>::min()) {
      abort();
    }
    return MPInt(-val);
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
