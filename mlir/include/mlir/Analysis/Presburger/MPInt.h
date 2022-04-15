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

namespace mlir {
namespace presburger {
using llvm::APInt2;

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
  explicit MPInt(int64_t val) : val(/*numBits=*/64, val, /*isSigned=*/true) {}
  MPInt() : MPInt(0) {}
  explicit MPInt(const APInt2 &val) : val(val) {}
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
  // TODO: consider using APInt2 directly to avoid unnecessary repeated internal
  // signedness checks. This requires refactoring, exposing, or duplicating
  // APInt2::compareValues.
  APInt2 val;
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

/// We define the operations here in the header to facilitate inlining.

/// ---------------------------------------------------------------------------
/// Comparison operators.
/// ---------------------------------------------------------------------------
namespace detail {
static int compareValues(const APInt2 &lhs, const APInt2 &rhs) {
  unsigned width = std::max(lhs.getBitWidth(), rhs.getBitWidth());
  return lhs.sextOrSelf(width).compareSigned(rhs.sextOrSelf(width));
}
} // namespace detail
inline bool MPInt::operator==(const MPInt &o) const {
  return detail::compareValues(val, o.val) == 0;
}
inline bool MPInt::operator!=(const MPInt &o) const {
  return detail::compareValues(val, o.val) != 0;
}
inline bool MPInt::operator>(const MPInt &o) const {
  return detail::compareValues(val, o.val) > 0;
}
inline bool MPInt::operator<(const MPInt &o) const {
  return detail::compareValues(val, o.val) < 0;
}
inline bool MPInt::operator<=(const MPInt &o) const {
  return detail::compareValues(val, o.val) <= 0;
}
inline bool MPInt::operator>=(const MPInt &o) const {
  return detail::compareValues(val, o.val) >= 0;
}

/// ---------------------------------------------------------------------------
/// Arithmetic operators.
/// ---------------------------------------------------------------------------
namespace detail {
/// Bring a and b to have the same width and then call a.op(b, overflow).
/// If the overflow bit becomes set, resize a and b to double the width and
/// call a.op(b, overflow), returning its result. The operation with double
/// widths should not also overflow.
template <typename Function>
inline APInt2 runOpWithExpandOnOverflow(const APInt2 &a, const APInt2 &b,
                                        const Function &op) {
  bool overflow;
  unsigned widthA = a.getBitWidth(), widthB = b.getBitWidth();
  APInt2 ret;
  if (widthA == widthB)
    ret = op(a, b, overflow);
  else if (widthA < widthB)
    ret = op(a.sext(widthB), b, overflow);
  else
    ret = op(a, b.sext(widthA), overflow);
  if (!overflow)
    return ret;

  unsigned newWidth = 2*std::max(widthA, widthB);
  ret = op(a.sext(newWidth), b.sext(newWidth), overflow);
  assert(!overflow && "double width should be sufficient to avoid overflow!");
  return ret;
}
} // namespace detail

inline MPInt MPInt::operator+(const MPInt &o) const {
  return MPInt(detail::runOpWithExpandOnOverflow(val, o.val,
                                                 std::mem_fn(&APInt2::sadd_ov)));
}
inline MPInt MPInt::operator-(const MPInt &o) const {
  return MPInt(detail::runOpWithExpandOnOverflow(val, o.val,
                                                 std::mem_fn(&APInt2::ssub_ov)));
}
inline MPInt MPInt::operator*(const MPInt &o) const {
  return MPInt(detail::runOpWithExpandOnOverflow(val, o.val,
                                                 std::mem_fn(&APInt2::smul_ov)));
}
inline MPInt MPInt::operator/(const MPInt &o) const {
  return MPInt(detail::runOpWithExpandOnOverflow(val, o.val,
                                                 std::mem_fn(&APInt2::sdiv_ov)));
}
inline MPInt abs(const MPInt &x) { return x >= 0 ? x : -x; }
inline MPInt ceilDiv(const MPInt &lhs, const MPInt &rhs) {
  if (rhs == -1)
    return -lhs;
  return MPInt(llvm::APInt2Ops::RoundingSDiv(lhs.val, rhs.val, APInt2::Rounding::UP));
}
inline MPInt floorDiv(const MPInt &lhs, const MPInt &rhs) {
  if (rhs == -1)
    return -lhs;
  return MPInt(llvm::APInt2Ops::RoundingSDiv(lhs.val, rhs.val, APInt2::Rounding::DOWN));
}
// The RHS is always expected to be positive, and the result
/// is always non-negative.
inline MPInt mod(const MPInt &lhs, const MPInt &rhs) {
  assert(rhs >= 1);
  return lhs % rhs < 0 ? lhs % rhs + rhs : lhs % rhs;
}

inline MPInt greatestCommonDivisor(const MPInt &a, const MPInt &b) {
  return MPInt(llvm::APInt2Ops::GreatestCommonDivisor(a.val.abs(), b.val.abs()));
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
  return MPInt(val.sextOrSelf(width).srem(o.val.sextOrSelf(width)));
}

inline MPInt MPInt::operator-() const {
  if (val.isMinSignedValue()) {
    /// Overflow only occurs when the value is the minimum possible value.
    return MPInt(-val.sext(2 * val.getBitWidth()));
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
