//===- MPInt.cpp - MLIR MPInt Class ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/MPAPInt.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace presburger;
using detail::MPAPInt;

llvm::hash_code mlir::presburger::detail::hash_value(const MPAPInt &x) {
  return hash_value(x.val);
}

/// ---------------------------------------------------------------------------
/// Printing.
/// ---------------------------------------------------------------------------
llvm::raw_ostream &MPAPInt::print(llvm::raw_ostream &os) const {
  return os << val;
}

void MPAPInt::dump() const { print(llvm::errs()); }

llvm::raw_ostream &mlir::presburger::detail::operator<<(llvm::raw_ostream &os,
                                                const MPAPInt &x) {
  x.print(os);
  return os;
}

/// ---------------------------------------------------------------------------
/// Convenience operator overloads for int64_t.
/// ---------------------------------------------------------------------------
MPAPInt &operator+=(MPAPInt &a, int64_t b) { return a += MPAPInt(b); }
MPAPInt &operator-=(MPAPInt &a, int64_t b) { return a -= MPAPInt(b); }
MPAPInt &operator*=(MPAPInt &a, int64_t b) { return a *= MPAPInt(b); }
MPAPInt &operator/=(MPAPInt &a, int64_t b) { return a /= MPAPInt(b); }
MPAPInt &operator%=(MPAPInt &a, int64_t b) { return a %= MPAPInt(b); }

bool operator==(const MPAPInt &a, int64_t b) { return a == MPAPInt(b); }
bool operator!=(const MPAPInt &a, int64_t b) { return a != MPAPInt(b); }
bool operator>(const MPAPInt &a, int64_t b) { return a > MPAPInt(b); }
bool operator<(const MPAPInt &a, int64_t b) { return a < MPAPInt(b); }
bool operator<=(const MPAPInt &a, int64_t b) { return a <= MPAPInt(b); }
bool operator>=(const MPAPInt &a, int64_t b) { return a >= MPAPInt(b); }
MPAPInt operator+(const MPAPInt &a, int64_t b) { return a + MPAPInt(b); }
MPAPInt operator-(const MPAPInt &a, int64_t b) { return a - MPAPInt(b); }
MPAPInt operator*(const MPAPInt &a, int64_t b) { return a * MPAPInt(b); }
MPAPInt operator/(const MPAPInt &a, int64_t b) { return a / MPAPInt(b); }
MPAPInt operator%(const MPAPInt &a, int64_t b) { return a % MPAPInt(b); }

bool operator==(int64_t a, const MPAPInt &b) { return MPAPInt(a) == b; }
bool operator!=(int64_t a, const MPAPInt &b) { return MPAPInt(a) != b; }
bool operator>(int64_t a, const MPAPInt &b) { return MPAPInt(a) > b; }
bool operator<(int64_t a, const MPAPInt &b) { return MPAPInt(a) < b; }
bool operator<=(int64_t a, const MPAPInt &b) { return MPAPInt(a) <= b; }
bool operator>=(int64_t a, const MPAPInt &b) { return MPAPInt(a) >= b; }
MPAPInt operator+(int64_t a, const MPAPInt &b) { return MPAPInt(a) + b; }
MPAPInt operator-(int64_t a, const MPAPInt &b) { return MPAPInt(a) - b; }
MPAPInt operator*(int64_t a, const MPAPInt &b) { return MPAPInt(a) * b; }
MPAPInt operator/(int64_t a, const MPAPInt &b) { return MPAPInt(a) / b; }
MPAPInt operator%(int64_t a, const MPAPInt &b) { return MPAPInt(a) % b; }

/// We define the operations here in the header to facilitate inlining.

/// ---------------------------------------------------------------------------
/// Comparison operators.
/// ---------------------------------------------------------------------------
bool MPAPInt::operator==(const MPAPInt &o) const {
  return APSInt::compareValues(val, o.val) == 0;
}
bool MPAPInt::operator!=(const MPAPInt &o) const {
  return APSInt::compareValues(val, o.val) != 0;
}
bool MPAPInt::operator>(const MPAPInt &o) const {
  return APSInt::compareValues(val, o.val) > 0;
}
bool MPAPInt::operator<(const MPAPInt &o) const {
  return APSInt::compareValues(val, o.val) < 0;
}
bool MPAPInt::operator<=(const MPAPInt &o) const {
  return APSInt::compareValues(val, o.val) <= 0;
}
bool MPAPInt::operator>=(const MPAPInt &o) const {
  return APSInt::compareValues(val, o.val) >= 0;
}

/// ---------------------------------------------------------------------------
/// Arithmetic operators.
/// ---------------------------------------------------------------------------

/// Bring a and b to have the same width and then call a.op(b, overflow).
/// If the overflow bit becomes set, resize a and b to double the width and
/// call a.op(b, overflow), returning its result. The operation with double
/// widths should not also overflow.
template <typename Function>
APSInt runOpWithExpandOnOverflow(const APInt &a, const APInt &b,
                                        const Function &op) {
  bool overflow;
  unsigned width = std::max(a.getBitWidth(), b.getBitWidth());
  APInt ret = op(a.sextOrSelf(width), b.sextOrSelf(width), overflow);
  if (!overflow)
    return APSInt(ret, /*isUnsigned=*/false);

  width *= 2;
  ret = op(a.sextOrSelf(width), b.sextOrSelf(width), overflow);
  assert(!overflow && "double width should be sufficient to avoid overflow!");
  return APSInt(ret, /*isUnsigned=*/false);
}

MPAPInt MPAPInt::operator+(const MPAPInt &o) const {
  return MPAPInt(runOpWithExpandOnOverflow(val, o.val,
                                                 std::mem_fn(&APInt::sadd_ov)));
}
MPAPInt MPAPInt::operator-(const MPAPInt &o) const {
  return MPAPInt(runOpWithExpandOnOverflow(val, o.val,
                                                 std::mem_fn(&APInt::ssub_ov)));
}
MPAPInt MPAPInt::operator*(const MPAPInt &o) const {
  return MPAPInt(runOpWithExpandOnOverflow(val, o.val,
                                                 std::mem_fn(&APInt::smul_ov)));
}
MPAPInt MPAPInt::operator/(const MPAPInt &o) const {
  return MPAPInt(runOpWithExpandOnOverflow(val, o.val,
                                                 std::mem_fn(&APInt::sdiv_ov)));
}
MPAPInt detail::abs(const MPAPInt &x) { return x >= 0 ? x : -x; }
MPAPInt detail::ceilDiv(const MPAPInt &lhs, const MPAPInt &rhs) {
  if (rhs == -1)
    return -lhs;
  return MPAPInt(APSInt(
      llvm::APIntOps::RoundingSDiv(lhs.val, rhs.val, APInt::Rounding::UP),
      /*isUnsigned=*/false));
}
MPAPInt detail::floorDiv(const MPAPInt &lhs, const MPAPInt &rhs) {
  if (rhs == -1)
    return -lhs;
  return MPAPInt(APSInt(
      llvm::APIntOps::RoundingSDiv(lhs.val, rhs.val, APInt::Rounding::DOWN),
      /*isUnsigned=*/false));
}
// The RHS is always expected to be positive, and the result
/// is always non-negative.
MPAPInt detail::mod(const MPAPInt &lhs, const MPAPInt &rhs) {
  assert(rhs >= 1);
  return lhs % rhs < 0 ? lhs % rhs + rhs : lhs % rhs;
}

MPAPInt detail::greatestCommonDivisor(const MPAPInt &a, const MPAPInt &b) {
  return MPAPInt(
      APSInt(llvm::APIntOps::GreatestCommonDivisor(a.val.abs(), b.val.abs()),
             /*isUnsigned=*/false));
}

/// Returns the least common multiple of 'a' and 'b'.
MPAPInt detail::lcm(const MPAPInt &a, const MPAPInt &b) {
  MPAPInt x = abs(a);
  MPAPInt y = abs(b);
  return (x * y) / greatestCommonDivisor(x, y);
}

/// This operation cannot overflow.
MPAPInt MPAPInt::operator%(const MPAPInt &o) const {
  unsigned width = std::max(val.getBitWidth(), o.val.getBitWidth());
  return MPAPInt(APSInt(val.sextOrSelf(width).srem(o.val.sextOrSelf(width)),
                      /*isUnsigned=*/false));
}

MPAPInt MPAPInt::operator-() const {
  if (val.isMinSignedValue()) {
    /// Overflow only occurs when the value is the minimum possible value.
    APSInt ret = val.extend(2 * val.getBitWidth());
    return MPAPInt(-ret);
  }
  return MPAPInt(-val);
}

/// ---------------------------------------------------------------------------
/// Assignment operators, preincrement, predecrement.
/// ---------------------------------------------------------------------------
MPAPInt &MPAPInt::operator+=(const MPAPInt &o) {
  *this = *this + o;
  return *this;
}
MPAPInt &MPAPInt::operator-=(const MPAPInt &o) {
  *this = *this - o;
  return *this;
}
MPAPInt &MPAPInt::operator*=(const MPAPInt &o) {
  *this = *this * o;
  return *this;
}
MPAPInt &MPAPInt::operator/=(const MPAPInt &o) {
  *this = *this / o;
  return *this;
}
MPAPInt &MPAPInt::operator%=(const MPAPInt &o) {
  *this = *this % o;
  return *this;
}
MPAPInt &MPAPInt::operator++() {
  *this += 1;
  return *this;
}

MPAPInt &MPAPInt::operator--() {
  *this -= 1;
  return *this;
}
