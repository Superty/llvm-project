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

#ifndef MLIR_ANALYSIS_PRESBURGER_MPAPINT_H
#define MLIR_ANALYSIS_PRESBURGER_MPAPINT_H

#include "mlir/Support/MathExtras.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace presburger {

namespace detail {
class MPAPInt {
public:
  explicit MPAPInt(int64_t val);
  MPAPInt();
  explicit MPAPInt(const llvm::APInt &val);
  MPAPInt &operator=(int64_t val);
  explicit operator int64_t() const;
  MPAPInt operator-() const;
  bool operator==(const MPAPInt &o) const;
  bool operator!=(const MPAPInt &o) const;
  bool operator>(const MPAPInt &o) const;
  bool operator<(const MPAPInt &o) const;
  bool operator<=(const MPAPInt &o) const;
  bool operator>=(const MPAPInt &o) const;
  MPAPInt operator+(const MPAPInt &o) const;
  MPAPInt operator-(const MPAPInt &o) const;
  MPAPInt operator*(const MPAPInt &o) const;
  MPAPInt operator/(const MPAPInt &o) const;
  MPAPInt operator%(const MPAPInt &o) const;
  MPAPInt &operator+=(const MPAPInt &o);
  MPAPInt &operator-=(const MPAPInt &o);
  MPAPInt &operator*=(const MPAPInt &o);
  MPAPInt &operator/=(const MPAPInt &o);
  MPAPInt &operator%=(const MPAPInt &o);

  MPAPInt &operator++();
  MPAPInt &operator--();

  friend MPAPInt abs(const MPAPInt &x);
  friend MPAPInt ceilDiv(const MPAPInt &lhs, const MPAPInt &rhs);
  friend MPAPInt floorDiv(const MPAPInt &lhs, const MPAPInt &rhs);
  friend MPAPInt gcd(const MPAPInt &a, const MPAPInt &b);
  /// Overload to compute a hash_code for a MPAPInt value.
  friend llvm::hash_code hash_value(const MPAPInt &x); // NOLINT

  llvm::raw_ostream &print(llvm::raw_ostream &os) const;
  void dump() const;

  unsigned getBitWidth() const { return val.getBitWidth(); }
private:
  int compare(const MPAPInt &o) const;
  llvm::APInt val;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const MPAPInt &x);
// The RHS is always expected to be positive, and the result
/// is always non-negative.
MPAPInt mod(const MPAPInt &lhs, const MPAPInt &rhs);
/// Returns the least common multiple of 'a' and 'b'.
MPAPInt lcm(const MPAPInt &a, const MPAPInt &b);

/// Redeclarations of friend declarations above to
/// make it discoverable by lookups.
MPAPInt abs(const MPAPInt &x);
MPAPInt ceilDiv(const MPAPInt &lhs, const MPAPInt &rhs);
MPAPInt floorDiv(const MPAPInt &lhs, const MPAPInt &rhs);
MPAPInt gcd(const MPAPInt &a, const MPAPInt &b);
/// Overload to compute a hash_code for a MPAPInt value.
llvm::hash_code hash_value(const MPAPInt &x); // NOLINT

/// ---------------------------------------------------------------------------
/// Convenience operator overloads for int64_t.
/// ---------------------------------------------------------------------------
inline MPAPInt &operator+=(MPAPInt &a, int64_t b) { return a += MPAPInt(b); }
inline MPAPInt &operator-=(MPAPInt &a, int64_t b) { return a -= MPAPInt(b); }
inline MPAPInt &operator*=(MPAPInt &a, int64_t b) { return a *= MPAPInt(b); }
inline MPAPInt &operator/=(MPAPInt &a, int64_t b) { return a /= MPAPInt(b); }
inline MPAPInt &operator%=(MPAPInt &a, int64_t b) { return a %= MPAPInt(b); }

inline bool operator==(const MPAPInt &a, int64_t b) { return a == MPAPInt(b); }
inline bool operator!=(const MPAPInt &a, int64_t b) { return a != MPAPInt(b); }
inline bool operator>(const MPAPInt &a, int64_t b) { return a > MPAPInt(b); }
inline bool operator<(const MPAPInt &a, int64_t b) { return a < MPAPInt(b); }
inline bool operator<=(const MPAPInt &a, int64_t b) { return a <= MPAPInt(b); }
inline bool operator>=(const MPAPInt &a, int64_t b) { return a >= MPAPInt(b); }
inline MPAPInt operator+(const MPAPInt &a, int64_t b) { return a + MPAPInt(b); }
inline MPAPInt operator-(const MPAPInt &a, int64_t b) { return a - MPAPInt(b); }
inline MPAPInt operator*(const MPAPInt &a, int64_t b) { return a * MPAPInt(b); }
inline MPAPInt operator/(const MPAPInt &a, int64_t b) { return a / MPAPInt(b); }
inline MPAPInt operator%(const MPAPInt &a, int64_t b) { return a % MPAPInt(b); }

inline bool operator==(int64_t a, const MPAPInt &b) { return MPAPInt(a) == b; }
inline bool operator!=(int64_t a, const MPAPInt &b) { return MPAPInt(a) != b; }
inline bool operator>(int64_t a, const MPAPInt &b) { return MPAPInt(a) > b; }
inline bool operator<(int64_t a, const MPAPInt &b) { return MPAPInt(a) < b; }
inline bool operator<=(int64_t a, const MPAPInt &b) { return MPAPInt(a) <= b; }
inline bool operator>=(int64_t a, const MPAPInt &b) { return MPAPInt(a) >= b; }
inline MPAPInt operator+(int64_t a, const MPAPInt &b) { return MPAPInt(a) + b; }
inline MPAPInt operator-(int64_t a, const MPAPInt &b) { return MPAPInt(a) - b; }
inline MPAPInt operator*(int64_t a, const MPAPInt &b) { return MPAPInt(a) * b; }
inline MPAPInt operator/(int64_t a, const MPAPInt &b) { return MPAPInt(a) / b; }
inline MPAPInt operator%(int64_t a, const MPAPInt &b) { return MPAPInt(a) % b; }
} // namespace detail
} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_MPAPINT_H
