//===- TPInt.h - MLIR TPInt Class -------------------------------*- C++ -*-===//
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

#ifndef MLIR_ANALYSIS_PRESBURGER_TPINT_H
#define MLIR_ANALYSIS_PRESBURGER_TPINT_H

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
/// Since it uses APInt internally, TPInt (TransPrecision Int) stores values in
/// a 64-bit machine integer internally for small values and uses slower
/// arbitrary-precision arithmetic only for larger values.
class TPInt {
public:
  explicit TPInt(int64_t val) : val(APSInt::get(val)) {}
  TPInt() : TPInt(0) {}
  explicit TPInt(const APSInt &val) : val(val) {}
  TPInt &operator=(int64_t val) { return *this = TPInt(val); }
  explicit operator int64_t() const { return val.getSExtValue(); }
  TPInt operator-() const;
  bool operator==(const TPInt &o) const;
  bool operator!=(const TPInt &o) const;
  bool operator>(const TPInt &o) const;
  bool operator<(const TPInt &o) const;
  bool operator<=(const TPInt &o) const;
  bool operator>=(const TPInt &o) const;
  TPInt operator+(const TPInt &o) const;
  TPInt operator-(const TPInt &o) const;
  TPInt operator*(const TPInt &o) const;
  TPInt operator/(const TPInt &o) const;
  TPInt operator%(const TPInt &o) const;
  TPInt &operator+=(const TPInt &o);
  TPInt &operator-=(const TPInt &o);
  TPInt &operator*=(const TPInt &o);
  TPInt &operator/=(const TPInt &o);
  TPInt &operator%=(const TPInt &o);

  TPInt &operator++();
  TPInt &operator--();

  friend TPInt abs(const TPInt &x);
  friend TPInt ceilDiv(const TPInt &lhs, const TPInt &rhs);
  friend TPInt floorDiv(const TPInt &lhs, const TPInt &rhs);
  friend TPInt greatestCommonDivisor(const TPInt &a, const TPInt &b);
  /// Overload to compute a hash_code for a TPInt value.
  friend llvm::hash_code hash_value(const TPInt &x); // NOLINT

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
inline int64_t int64FromTPInt(const TPInt &x) { return int64_t(x); }

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const TPInt &x);

// The RHS is always expected to be positive, and the result
/// is always non-negative.
TPInt mod(const TPInt &lhs, const TPInt &rhs);

/// Returns the least common multiple of 'a' and 'b'.
TPInt lcm(const TPInt &a, const TPInt &b);

/// Convenience overloads for 64-bit integers.
TPInt &operator+=(TPInt &a, int64_t b);
TPInt &operator-=(TPInt &a, int64_t b);
TPInt &operator*=(TPInt &a, int64_t b);
TPInt &operator/=(TPInt &a, int64_t b);
TPInt &operator%=(TPInt &a, int64_t b);

bool operator==(const TPInt &a, int64_t b);
bool operator!=(const TPInt &a, int64_t b);
bool operator>(const TPInt &a, int64_t b);
bool operator<(const TPInt &a, int64_t b);
bool operator<=(const TPInt &a, int64_t b);
bool operator>=(const TPInt &a, int64_t b);
TPInt operator+(const TPInt &a, int64_t b);
TPInt operator-(const TPInt &a, int64_t b);
TPInt operator*(const TPInt &a, int64_t b);
TPInt operator/(const TPInt &a, int64_t b);
TPInt operator%(const TPInt &a, int64_t b);

bool operator==(int64_t a, const TPInt &b);
bool operator!=(int64_t a, const TPInt &b);
bool operator>(int64_t a, const TPInt &b);
bool operator<(int64_t a, const TPInt &b);
bool operator<=(int64_t a, const TPInt &b);
bool operator>=(int64_t a, const TPInt &b);
TPInt operator+(int64_t a, const TPInt &b);
TPInt operator-(int64_t a, const TPInt &b);
TPInt operator*(int64_t a, const TPInt &b);
TPInt operator/(int64_t a, const TPInt &b);
TPInt operator%(int64_t a, const TPInt &b);

} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_TPINT_H
