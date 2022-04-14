//===- TPInt.cpp - MLIR TPInt Class ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/TPInt.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace presburger;

llvm::hash_code mlir::presburger::hash_value(const TPInt &x) {
  return hash_value(x.val);
}

/// ---------------------------------------------------------------------------
/// Printing.
/// ---------------------------------------------------------------------------
llvm::raw_ostream &TPInt::print(llvm::raw_ostream &os) const {
  return os << val;
}

void TPInt::dump() const { print(llvm::errs()); }

llvm::raw_ostream &mlir::presburger::operator<<(llvm::raw_ostream &os,
                                                const TPInt &x) {
  x.print(os);
  return os;
}

/// ---------------------------------------------------------------------------
/// Comparison operators.
/// ---------------------------------------------------------------------------
bool TPInt::operator==(const TPInt &o) const {
  return APSInt::compareValues(val, o.val) == 0;
}
bool TPInt::operator!=(const TPInt &o) const {
  return APSInt::compareValues(val, o.val) != 0;
}
bool TPInt::operator>(const TPInt &o) const {
  return APSInt::compareValues(val, o.val) > 0;
}
bool TPInt::operator<(const TPInt &o) const {
  return APSInt::compareValues(val, o.val) < 0;
}
bool TPInt::operator<=(const TPInt &o) const {
  return APSInt::compareValues(val, o.val) <= 0;
}
bool TPInt::operator>=(const TPInt &o) const {
  return APSInt::compareValues(val, o.val) >= 0;
}

/// ---------------------------------------------------------------------------
/// Arithmetic operators.
/// ---------------------------------------------------------------------------
using APIntOvOp = APInt (APInt::*)(const APInt &b, bool &overflow) const;

/// Bring a and b to have the same width and then call a.op(b, overflow).
/// If the overflow bit becomes set, resize a and b to double the width and
/// call a.op(b, overflow), returning its result. The operation with double
/// widths should not also overflow.
APSInt doOpExpandIfOverflow(const APInt &a, const APInt &b, APIntOvOp op) {
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

TPInt TPInt::operator+(const TPInt &o) const {
  return TPInt(doOpExpandIfOverflow(val, o.val, &APInt::sadd_ov));
}
TPInt TPInt::operator-(const TPInt &o) const {
  return TPInt(doOpExpandIfOverflow(val, o.val, &APInt::ssub_ov));
}
TPInt TPInt::operator*(const TPInt &o) const {
  return TPInt(doOpExpandIfOverflow(val, o.val, &APInt::smul_ov));
}
TPInt TPInt::operator/(const TPInt &o) const {
  return TPInt(doOpExpandIfOverflow(val, o.val, &APInt::sdiv_ov));
}
namespace mlir {
namespace presburger {
using llvm::APIntOps::GreatestCommonDivisor;
using llvm::APIntOps::RoundingSDiv;
TPInt abs(const TPInt &x) { return x >= 0 ? x : -x; }
TPInt ceilDiv(const TPInt &lhs, const TPInt &rhs) {
  if (rhs == -1)
    return -lhs;
  return TPInt(APSInt(RoundingSDiv(lhs.val, rhs.val, APInt::Rounding::UP),
                      /*isUnsigned=*/false));
}
TPInt floorDiv(const TPInt &lhs, const TPInt &rhs) {
  if (rhs == -1)
    return -lhs;
  return TPInt(APSInt(RoundingSDiv(lhs.val, rhs.val, APInt::Rounding::DOWN),
                      /*isUnsigned=*/false));
}
// The RHS is always expected to be positive, and the result
/// is always non-negative.
TPInt mod(const TPInt &lhs, const TPInt &rhs) {
  assert(rhs >= 1);
  return lhs % rhs < 0 ? lhs % rhs + rhs : lhs % rhs;
}

TPInt greatestCommonDivisor(const TPInt &a, const TPInt &b) {
  return TPInt(APSInt(GreatestCommonDivisor(a.val.abs(), b.val.abs()),
                      /*isUnsigned=*/false));
}

/// Returns the least common multiple of 'a' and 'b'.
TPInt lcm(const TPInt &a, const TPInt &b) {
  TPInt x = abs(a);
  TPInt y = abs(b);
  TPInt lcm = (x * y) / greatestCommonDivisor(x, y);
  assert((lcm >= a && lcm >= b) && "LCM overflow");
  return lcm;
}
} // namespace presburger
} // namespace mlir

TPInt TPInt::operator%(const TPInt &o) const {
  unsigned width = std::max(val.getBitWidth(), o.val.getBitWidth());
  return TPInt(APSInt(val.sextOrSelf(width).srem(o.val.sextOrSelf(width)),
                      /*isUnsigned=*/false));
}

TPInt TPInt::operator-() const {
  if (val.isMinSignedValue()) {
    APSInt ret = val.extend(2 * val.getBitWidth());
    return TPInt(-ret);
  }
  return TPInt(-val);
}

/// ---------------------------------------------------------------------------
/// Assignment operators, preincrement, predecrement.
/// ---------------------------------------------------------------------------
TPInt &TPInt::operator+=(const TPInt &o) {
  *this = *this + o;
  return *this;
}
TPInt &TPInt::operator-=(const TPInt &o) {
  *this = *this - o;
  return *this;
}
TPInt &TPInt::operator*=(const TPInt &o) {
  *this = *this * o;
  return *this;
}
TPInt &TPInt::operator/=(const TPInt &o) {
  *this = *this / o;
  return *this;
}
TPInt &TPInt::operator%=(const TPInt &o) {
  *this = *this % o;
  return *this;
}
TPInt &TPInt::operator++() {
  *this += 1;
  return *this;
}

TPInt &TPInt::operator--() {
  *this -= 1;
  return *this;
}

/// ---------------------------------------------------------------------------
/// Convenience operator overloads for int64_t.
/// ---------------------------------------------------------------------------
namespace mlir {
namespace presburger {
TPInt &operator+=(TPInt &a, int64_t b) { return a += TPInt(b); }
TPInt &operator-=(TPInt &a, int64_t b) { return a -= TPInt(b); }
TPInt &operator*=(TPInt &a, int64_t b) { return a *= TPInt(b); }
TPInt &operator/=(TPInt &a, int64_t b) { return a /= TPInt(b); }
TPInt &operator%=(TPInt &a, int64_t b) { return a %= TPInt(b); }
bool operator==(const TPInt &a, int64_t b) { return a == TPInt(b); }
bool operator!=(const TPInt &a, int64_t b) { return a != TPInt(b); }
bool operator>(const TPInt &a, int64_t b) { return a > TPInt(b); }
bool operator<(const TPInt &a, int64_t b) { return a < TPInt(b); }
bool operator<=(const TPInt &a, int64_t b) { return a <= TPInt(b); }
bool operator>=(const TPInt &a, int64_t b) { return a >= TPInt(b); }
TPInt operator+(const TPInt &a, int64_t b) { return a + TPInt(b); }
TPInt operator-(const TPInt &a, int64_t b) { return a - TPInt(b); }
TPInt operator*(const TPInt &a, int64_t b) { return a * TPInt(b); }
TPInt operator/(const TPInt &a, int64_t b) { return a / TPInt(b); }
TPInt operator%(const TPInt &a, int64_t b) { return a % TPInt(b); }
bool operator==(int64_t a, const TPInt &b) { return TPInt(a) == b; }
bool operator!=(int64_t a, const TPInt &b) { return TPInt(a) != b; }
bool operator>(int64_t a, const TPInt &b) { return TPInt(a) > b; }
bool operator<(int64_t a, const TPInt &b) { return TPInt(a) < b; }
bool operator<=(int64_t a, const TPInt &b) { return TPInt(a) <= b; }
bool operator>=(int64_t a, const TPInt &b) { return TPInt(a) >= b; }
TPInt operator+(int64_t a, const TPInt &b) { return TPInt(a) + b; }
TPInt operator-(int64_t a, const TPInt &b) { return TPInt(a) - b; }
TPInt operator*(int64_t a, const TPInt &b) { return TPInt(a) * b; }
TPInt operator/(int64_t a, const TPInt &b) { return TPInt(a) / b; }
TPInt operator%(int64_t a, const TPInt &b) { return TPInt(a) % b; }
} // namespace presburger
} // namespace mlir
