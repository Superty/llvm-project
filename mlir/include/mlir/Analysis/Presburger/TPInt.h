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
// integer can take on any aribtrary values.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_MPINT_H
#define MLIR_ANALYSIS_PRESBURGER_MPINT_H

#include "mlir/Support/MathExtras.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace presburger {

/// This class provides support for multi-precision arithmetic.
///
/// Unlike APInt, this extends the precision as necessary to prevent overflows
/// and supports operations between objects with differing internal precisions.
///
/// Since it uses APInt internally, MPInt (MultiPrecision Integer) stores values
/// in a 64-bit machine integer for small values and uses slower
/// arbitrary-precision arithmetic only for larger values.
class MPInt {
public:
  explicit MPInt(int64_t val) : val(APSInt::get(val)) {}
  MPInt() : MPInt(0) {}
  explicit MPInt(const APSInt &val) : val(val) {}
  MPInt &operator=(int64_t val) { return *this = MPInt(val); }
  explicit operator int64_t() const { return val.getSExtValue(); }
  MPInt operator-() const;
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

  friend MPInt abs(const MPInt &x);
  friend MPInt ceilDiv(const MPInt &lhs, const MPInt &rhs);
  friend MPInt floorDiv(const MPInt &lhs, const MPInt &rhs);
  friend MPInt greatestCommonDivisor(const MPInt &a, const MPInt &b);
  /// Overload to compute a hash_code for a MPInt value.
  friend llvm::hash_code hash_value(const MPInt &x); // NOLINT

  llvm::raw_ostream &print(llvm::raw_ostream &os) const;
  void dump() const;

private:
  unsigned getBitWidth() const { return val.getBitWidth(); }

  // The held integer value.
  //
  // TODO: consider using APInt directly to avoid unnecessary repeated internal
  // signedness checks. This requires refactoring, exposing, or duplicating
  // APSInt::compareValues.
  APSInt val;
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

/// ---------------------------------------------------------------------------
/// Convenience operator overloads for int64_t.
/// ---------------------------------------------------------------------------
inline MPInt &operator+=(MPInt &a, int64_t b) { return a += MPInt(b); }
inline MPInt &operator-=(MPInt &a, int64_t b) { return a -= MPInt(b); }
inline MPInt &operator*=(MPInt &a, int64_t b) { return a *= MPInt(b); }
inline MPInt &operator/=(MPInt &a, int64_t b) { return a /= MPInt(b); }
inline MPInt &operator%=(MPInt &a, int64_t b) { return a %= MPInt(b); }

inline bool operator==(const MPInt &a, int64_t b) { return a == MPInt(b); }
inline bool operator!=(const MPInt &a, int64_t b) { return a != MPInt(b); }
inline bool operator>(const MPInt &a, int64_t b) { return a > MPInt(b); }
inline bool operator<(const MPInt &a, int64_t b) { return a < MPInt(b); }
inline bool operator<=(const MPInt &a, int64_t b) { return a <= MPInt(b); }
inline bool operator>=(const MPInt &a, int64_t b) { return a >= MPInt(b); }
inline MPInt operator+(const MPInt &a, int64_t b) { return a + MPInt(b); }
inline MPInt operator-(const MPInt &a, int64_t b) { return a - MPInt(b); }
inline MPInt operator*(const MPInt &a, int64_t b) { return a * MPInt(b); }
inline MPInt operator/(const MPInt &a, int64_t b) { return a / MPInt(b); }
inline MPInt operator%(const MPInt &a, int64_t b) { return a % MPInt(b); }

inline bool operator==(int64_t a, const MPInt &b) { return MPInt(a) == b; }
inline bool operator!=(int64_t a, const MPInt &b) { return MPInt(a) != b; }
inline bool operator>(int64_t a, const MPInt &b) { return MPInt(a) > b; }
inline bool operator<(int64_t a, const MPInt &b) { return MPInt(a) < b; }
inline bool operator<=(int64_t a, const MPInt &b) { return MPInt(a) <= b; }
inline bool operator>=(int64_t a, const MPInt &b) { return MPInt(a) >= b; }
inline MPInt operator+(int64_t a, const MPInt &b) { return MPInt(a) + b; }
inline MPInt operator-(int64_t a, const MPInt &b) { return MPInt(a) - b; }
inline MPInt operator*(int64_t a, const MPInt &b) { return MPInt(a) * b; }
inline MPInt operator/(int64_t a, const MPInt &b) { return MPInt(a) / b; }
inline MPInt operator%(int64_t a, const MPInt &b) { return MPInt(a) % b; }

/// We define the operators here in the header to facilitate inlining.

/// ---------------------------------------------------------------------------
/// Comparison operators.
/// ---------------------------------------------------------------------------
inline bool MPInt::operator==(const MPInt &o) const {
  return APSInt::compareValues(val, o.val) == 0;
}
inline bool MPInt::operator!=(const MPInt &o) const {
  return APSInt::compareValues(val, o.val) != 0;
}
inline bool MPInt::operator>(const MPInt &o) const {
  return APSInt::compareValues(val, o.val) > 0;
}
inline bool MPInt::operator<(const MPInt &o) const {
  return APSInt::compareValues(val, o.val) < 0;
}
inline bool MPInt::operator<=(const MPInt &o) const {
  return APSInt::compareValues(val, o.val) <= 0;
}
inline bool MPInt::operator>=(const MPInt &o) const {
  return APSInt::compareValues(val, o.val) >= 0;
}

/// ---------------------------------------------------------------------------
/// Arithmetic operators.
/// ---------------------------------------------------------------------------
namespace detail {
using APIntOvOp = APInt (APInt::*)(const APInt &b, bool &overflow) const;

/// Bring a and b to have the same width and then call a.op(b, overflow).
/// If the overflow bit becomes set, resize a and b to double the width and
/// call a.op(b, overflow), returning its result. The operation with double
/// widths should not also overflow.
inline APSInt doOpExpandIfOverflow(const APInt &a, const APInt &b, APIntOvOp op) {
  bool overflow;
  unsigned width = std::max(a.getBitWidth(), b.getBitWidth());
  // This calls a.sextOrSelf(width).op(b.sextOrSelf(width), overflow).
  // TODO: in C++17 we can use the simpler syntax with std::invoke.
  APInt ret = ((a.sextOrSelf(width)).*(op))(b.sextOrSelf(width), overflow);
  if (!overflow)
    return APSInt(ret, /*isUnsigned=*/false);

  width *= 2;
  // This calls a.sextOrSelf(width).op(b.sextOrSelf(width), overflow).
  ret = ((a.sextOrSelf(width)).*(op))(b.sextOrSelf(width), overflow);
  assert(!overflow && "double width should be sufficient to avoid overflow!");
  return APSInt(ret, /*isUnsigned=*/false);
}
} // namespace detail

inline MPInt MPInt::operator+(const MPInt &o) const {
  return MPInt(detail::doOpExpandIfOverflow(val, o.val, &APInt::sadd_ov));
}
inline MPInt MPInt::operator-(const MPInt &o) const {
  return MPInt(detail::doOpExpandIfOverflow(val, o.val, &APInt::ssub_ov));
}
inline MPInt MPInt::operator*(const MPInt &o) const {
  return MPInt(detail::doOpExpandIfOverflow(val, o.val, &APInt::smul_ov));
}
inline MPInt MPInt::operator/(const MPInt &o) const {
  return MPInt(detail::doOpExpandIfOverflow(val, o.val, &APInt::sdiv_ov));
}
inline MPInt abs(const MPInt &x) { return x >= 0 ? x : -x; }
inline MPInt ceilDiv(const MPInt &lhs, const MPInt &rhs) {
  if (rhs == -1)
    return -lhs;
  return MPInt(APSInt(llvm::APIntOps::RoundingSDiv(lhs.val, rhs.val, APInt::Rounding::UP),
                      /*isUnsigned=*/false));
}
inline MPInt floorDiv(const MPInt &lhs, const MPInt &rhs) {
  if (rhs == -1)
    return -lhs;
  return MPInt(APSInt(llvm::APIntOps::RoundingSDiv(lhs.val, rhs.val, APInt::Rounding::DOWN),
                      /*isUnsigned=*/false));
}
// The RHS is always expected to be positive, and the result
/// is always non-negative.
inline MPInt mod(const MPInt &lhs, const MPInt &rhs) {
  assert(rhs >= 1);
  return lhs % rhs < 0 ? lhs % rhs + rhs : lhs % rhs;
}

inline MPInt greatestCommonDivisor(const MPInt &a, const MPInt &b) {
  return MPInt(APSInt(llvm::APIntOps::GreatestCommonDivisor(a.val.abs(), b.val.abs()),
                      /*isUnsigned=*/false));
}

/// Returns the least common multiple of 'a' and 'b'.
inline MPInt lcm(const MPInt &a, const MPInt &b) {
  MPInt x = abs(a);
  MPInt y = abs(b);
  return (x * y) / greatestCommonDivisor(x, y);
}

/// This operation cannot overflow.
inline MPInt MPInt::operator%(const MPInt &o) const {
  unsigned width = std::max(val.getBitWidth(), o.val.getBitWidth());
  return MPInt(APSInt(val.sextOrSelf(width).srem(o.val.sextOrSelf(width)),
                      /*isUnsigned=*/false));
}

inline MPInt MPInt::operator-() const {
  if (val.isMinSignedValue()) {
    /// Overflow only occurs when the values is the minimum possible value.
    APSInt ret = val.extend(2 * val.getBitWidth());
    return MPInt(-ret);
  }
  return MPInt(-val);
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
