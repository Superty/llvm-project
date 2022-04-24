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

#include "mlir/Analysis/Presburger/MPAPInt.h"
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
  explicit MPInt(int64_t val) : val64(val), holdsAP(false) {}
  MPInt() : MPInt(0) {}
  ~MPInt() {
    if (isLarge())
      valAP.detail::MPAPInt::~MPAPInt();
  }
  MPInt(const MPInt &o) : val64(o.val64), holdsAP(false) {
    if (o.isLarge())
      initAP(o.valAP);
  }
  MPInt &operator=(const MPInt &o) {
    if (o.isLarge())
      initAP(o.valAP);
    else
      init64(o.val64);
    return *this;
  }
  MPInt &operator=(int x) {
    init64(x);
    return *this;
  }
  explicit operator int64_t() const {
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

private:
  explicit MPInt(detail::MPAPInt val) : valAP(val), holdsAP(true) {}
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
  __attribute__((always_inline)) const detail::MPAPInt &getAP() const {
    assert(isLarge());
    return valAP;
  }
  __attribute__((always_inline)) detail::MPAPInt &getAP() {
    assert(isLarge());
    return valAP;
  }
  explicit operator detail::MPAPInt() const {
    if (isSmall())
      return detail::MPAPInt(get64());
    return getAP();
  }
  detail::MPAPInt getAsAP() const {
    return detail::MPAPInt(*this);
  }

  union {
    int64_t val64;
    detail::MPAPInt valAP;
  };
  void init64(int64_t o) {
    val64 = o;
    holdsAP = false;
  }
  void initAP(const detail::MPAPInt &o) {
    valAP = o;
    holdsAP = true;
  }
  bool holdsAP;
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
  if (isSmall() && o.isSmall())
    return get64() == o.get64();
  return getAsAP() == o.getAsAP();
}
inline bool MPInt::operator!=(const MPInt &o) const {
  if (isSmall() && o.isSmall())
    return get64() != o.get64();
  return getAsAP() != o.getAsAP();
}
inline bool MPInt::operator>(const MPInt &o) const {
  if (isSmall() && o.isSmall())
    return get64() > o.get64();
  return getAsAP() > o.getAsAP();
}
inline bool MPInt::operator<(const MPInt &o) const {
  if (isSmall() && o.isSmall())
    return get64() < o.get64();
  return getAsAP() < o.getAsAP();
}
inline bool MPInt::operator<=(const MPInt &o) const {
  if (isSmall() && o.isSmall())
    return get64() <= o.get64();
  return getAsAP() <= o.getAsAP();
}
inline bool MPInt::operator>=(const MPInt &o) const {
  if (isSmall() && o.isSmall())
    return get64() >= o.get64();
  return getAsAP() >= o.getAsAP();
}

inline MPInt MPInt::operator*(const MPInt &o) const {
  if (isSmall() && o.isSmall()) {
    MPInt result;
    bool overflow = __builtin_mul_overflow(get64(), o.get64(), &result.get64());
    if (!overflow)
      return result;
    return MPInt(getAsAP() * o.getAsAP());
  }
  return MPInt(getAsAP() * o.getAsAP());
}

} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_MPINT_H
