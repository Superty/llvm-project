//===- DynamicAPInt.h - DynamicAPInt Class ----------------------*- C++ -*-===//
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

#ifndef LLVM_ADT_DYNAMICAPINT_H
#define LLVM_ADT_DYNAMICAPINT_H

#include "llvm/ADT/SlowDynamicAPInt.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <numeric>
#include <bit>

namespace llvm {
/// This class provides support for dynamic arbitrary-precision arithmetic.
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
/// When isLarge returns true, a SlowMPInt is held in the union. If isSmall
/// returns true, the int32_t is held. We don't have a separate field for
/// indicating this, and instead "steal" memory from ValLarge when it is not in
/// use because we know that the memory layout of APInt is such that BitWidth
/// doesn't overlap with ValSmall (see static_assert_layout). Using std::variant
/// instead would lead to significantly worse performance.
const int32_t HoldsSmallVal = 1 << 31;
class DynamicAPInt {
  int32_t HoldsSmall;
  int32_t ValSmall;

  LLVM_ATTRIBUTE_ALWAYS_INLINE void initSmall(int32_t O) {
    if (LLVM_UNLIKELY(isLarge()))
      delete &getLarge();
    ValSmall = O;
    HoldsSmall = HoldsSmallVal;
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE void
  setLarge(const detail::SlowDynamicAPInt &O) {
    detail::SlowDynamicAPInt *Ptr = new detail::SlowDynamicAPInt(O);
    int64_t PtrInt = bit_cast<int64_t>(Ptr);
    HoldsSmall = static_cast<int32_t>(PtrInt >> 32);
    ValSmall = static_cast<int32_t>(PtrInt);
  }

  LLVM_ATTRIBUTE_ALWAYS_INLINE void
  initLarge(const detail::SlowDynamicAPInt &O) {
    if (LLVM_UNLIKELY(isLarge()))
      delete &getLarge();
    setLarge(O);
}

  LLVM_ATTRIBUTE_ALWAYS_INLINE explicit DynamicAPInt(
      const detail::SlowDynamicAPInt &Val) {
    setLarge(Val);
}
  LLVM_ATTRIBUTE_ALWAYS_INLINE constexpr bool isSmall() const {
    return HoldsSmall != 0;
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE constexpr bool isLarge() const {
    return !isSmall();
  }
  /// Get the stored value. For getSmall/Large,
  /// the stored value should be small/large.
  LLVM_ATTRIBUTE_ALWAYS_INLINE int32_t getSmall() const {
    assert(isSmall() &&
           "getSmall should only be called when the value stored is small!");
    return ValSmall;
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE int32_t &getSmall() {
    assert(isSmall() &&
           "getSmall should only be called when the value stored is small!");
    return ValSmall;
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE const detail::SlowDynamicAPInt &
  getLarge() const {
    assert(isLarge() &&
           "getLarge should only be called when the value stored is large!");
    int64_t PtrInt = (static_cast<int64_t>(HoldsSmall) << 32) | static_cast<int64_t>(ValSmall);
    auto *Ptr = reinterpret_cast<detail::SlowDynamicAPInt*>(PtrInt);
    return *Ptr;
    // exit(3);
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE detail::SlowDynamicAPInt &getLarge() {
    assert(isLarge() &&
           "getLarge should only be called when the value stored is large!");
    int64_t PtrInt = (static_cast<int64_t>(HoldsSmall) << 32) | static_cast<int64_t>(ValSmall);
    auto *Ptr = reinterpret_cast<detail::SlowDynamicAPInt*>(PtrInt);
    return *Ptr;
    // exit(3);
  }
  explicit operator detail::SlowDynamicAPInt() const {
    if (isSmall())
      return detail::SlowDynamicAPInt(getSmall());
    return getLarge();
  }

public:
  LLVM_ATTRIBUTE_ALWAYS_INLINE explicit DynamicAPInt(int32_t Val)
      : ValSmall(Val) {
    HoldsSmall = HoldsSmallVal;
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt() : DynamicAPInt(0) {}
  LLVM_ATTRIBUTE_ALWAYS_INLINE ~DynamicAPInt() {
    if (LLVM_UNLIKELY(isLarge()))
      delete &getLarge();
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt(const DynamicAPInt &O)
      : ValSmall(O.ValSmall) {
    HoldsSmall = HoldsSmallVal;
    if (LLVM_UNLIKELY(O.isLarge()))
      setLarge(O.getLarge());
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt &operator=(const DynamicAPInt &O) {
    if (LLVM_LIKELY(O.isSmall())) {
      initSmall(O.ValSmall);
      return *this;
    }
    initLarge(O.getLarge());
    return *this;
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt &operator=(int X) {
    initSmall(X);
    return *this;
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE explicit operator int32_t() const {
    if (isSmall())
      return getSmall();
    exit(3);
    // return static_cast<int32_t>(getLarge());
  }

  bool operator==(const DynamicAPInt &O) const;
  bool operator!=(const DynamicAPInt &O) const;
  bool operator>(const DynamicAPInt &O) const;
  bool operator<(const DynamicAPInt &O) const;
  bool operator<=(const DynamicAPInt &O) const;
  bool operator>=(const DynamicAPInt &O) const;
  DynamicAPInt operator+(const DynamicAPInt &O) const;
  DynamicAPInt operator-(const DynamicAPInt &O) const;
  DynamicAPInt operator*(const DynamicAPInt &O) const;
  DynamicAPInt operator/(const DynamicAPInt &O) const;
  DynamicAPInt operator%(const DynamicAPInt &O) const;
  DynamicAPInt &operator+=(const DynamicAPInt &O);
  DynamicAPInt &operator-=(const DynamicAPInt &O);
  DynamicAPInt &operator*=(const DynamicAPInt &O);
  DynamicAPInt &operator/=(const DynamicAPInt &O);
  DynamicAPInt &operator%=(const DynamicAPInt &O);
  DynamicAPInt operator-() const;
  DynamicAPInt &operator++();
  DynamicAPInt &operator--();

  // Divide by a number that is known to be positive.
  // This is slightly more efficient because it saves an overflow check.
  DynamicAPInt divByPositive(const DynamicAPInt &O) const;
  DynamicAPInt &divByPositiveInPlace(const DynamicAPInt &O);

  friend DynamicAPInt abs(const DynamicAPInt &X);
  friend DynamicAPInt ceilDiv(const DynamicAPInt &LHS, const DynamicAPInt &RHS);
  friend DynamicAPInt floorDiv(const DynamicAPInt &LHS,
                               const DynamicAPInt &RHS);
  // The operands must be non-negative for gcd.
  friend DynamicAPInt gcd(const DynamicAPInt &A, const DynamicAPInt &B);
  friend DynamicAPInt lcm(const DynamicAPInt &A, const DynamicAPInt &B);
  friend DynamicAPInt mod(const DynamicAPInt &LHS, const DynamicAPInt &RHS);

  /// ---------------------------------------------------------------------------
  /// Convenience operator overloads for int32_t.
  /// ---------------------------------------------------------------------------
  friend DynamicAPInt &operator+=(DynamicAPInt &A, int32_t B);
  friend DynamicAPInt &operator-=(DynamicAPInt &A, int32_t B);
  friend DynamicAPInt &operator*=(DynamicAPInt &A, int32_t B);
  friend DynamicAPInt &operator/=(DynamicAPInt &A, int32_t B);
  friend DynamicAPInt &operator%=(DynamicAPInt &A, int32_t B);

  friend bool operator==(const DynamicAPInt &A, int32_t B);
  friend bool operator!=(const DynamicAPInt &A, int32_t B);
  friend bool operator>(const DynamicAPInt &A, int32_t B);
  friend bool operator<(const DynamicAPInt &A, int32_t B);
  friend bool operator<=(const DynamicAPInt &A, int32_t B);
  friend bool operator>=(const DynamicAPInt &A, int32_t B);
  friend DynamicAPInt operator+(DynamicAPInt &A, int32_t B);
  friend DynamicAPInt operator-(const DynamicAPInt &A, int32_t B);
  friend DynamicAPInt operator*(const DynamicAPInt &A, int32_t B);
  friend DynamicAPInt operator/(const DynamicAPInt &A, int32_t B);
  friend DynamicAPInt operator%(const DynamicAPInt &A, int32_t B);

  friend bool operator==(int32_t A, const DynamicAPInt &B);
  friend bool operator!=(int32_t A, const DynamicAPInt &B);
  friend bool operator>(int32_t A, const DynamicAPInt &B);
  friend bool operator<(int32_t A, const DynamicAPInt &B);
  friend bool operator<=(int32_t A, const DynamicAPInt &B);
  friend bool operator>=(int32_t A, const DynamicAPInt &B);
  friend DynamicAPInt operator+(int32_t A, const DynamicAPInt &B);
  friend DynamicAPInt operator-(int32_t A, const DynamicAPInt &B);
  friend DynamicAPInt operator*(int32_t A, const DynamicAPInt &B);
  friend DynamicAPInt operator/(int32_t A, const DynamicAPInt &B);
  friend DynamicAPInt operator%(int32_t A, const DynamicAPInt &B);

  friend hash_code hash_value(const DynamicAPInt &x); // NOLINT

  void static_assert_layout(); // NOLINT

  raw_ostream &print(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;


  friend DynamicAPInt &fallbackOpMulEqual(DynamicAPInt &A, int32_t B);
};

inline raw_ostream &operator<<(raw_ostream &OS, const DynamicAPInt &X) {
  X.print(OS);
  return OS;
}

/// Redeclarations of friend declaration above to
/// make it discoverable by lookups.
hash_code hash_value(const DynamicAPInt &X); // NOLINT

/// This just calls through to the operator int32_t, but it's useful when a
/// function pointer is required. (Although this is marked inline, it is still
/// possible to obtain and use a function pointer to this.)
static inline int32_t int64fromDynamicAPInt(const DynamicAPInt &X) {
  return int32_t(X);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt dynamicAPIntFromInt64(int32_t X) {
  return DynamicAPInt(X);
}

// The RHS is always expected to be positive, and the result
/// is always non-negative.
LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt mod(const DynamicAPInt &LHS,
                                              const DynamicAPInt &RHS);

/// We define the operations here in the header to facilitate inlining.

/// ---------------------------------------------------------------------------
/// Comparison operators.
/// ---------------------------------------------------------------------------
LLVM_ATTRIBUTE_ALWAYS_INLINE bool
DynamicAPInt::operator==(const DynamicAPInt &O) const {
  if (LLVM_LIKELY(isSmall() && O.isSmall()))
    return getSmall() == O.getSmall();
  return detail::SlowDynamicAPInt(*this) == detail::SlowDynamicAPInt(O);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE bool
DynamicAPInt::operator!=(const DynamicAPInt &O) const {
  if (LLVM_LIKELY(isSmall() && O.isSmall()))
    return getSmall() != O.getSmall();
  return detail::SlowDynamicAPInt(*this) != detail::SlowDynamicAPInt(O);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE bool
DynamicAPInt::operator>(const DynamicAPInt &O) const {
  if (LLVM_LIKELY(isSmall() && O.isSmall()))
    return getSmall() > O.getSmall();
  return detail::SlowDynamicAPInt(*this) > detail::SlowDynamicAPInt(O);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE bool
DynamicAPInt::operator<(const DynamicAPInt &O) const {
  if (LLVM_LIKELY(isSmall() && O.isSmall()))
    return getSmall() < O.getSmall();
  return detail::SlowDynamicAPInt(*this) < detail::SlowDynamicAPInt(O);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE bool
DynamicAPInt::operator<=(const DynamicAPInt &O) const {
  if (LLVM_LIKELY(isSmall() && O.isSmall()))
    return getSmall() <= O.getSmall();
  return detail::SlowDynamicAPInt(*this) <= detail::SlowDynamicAPInt(O);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE bool
DynamicAPInt::operator>=(const DynamicAPInt &O) const {
  if (LLVM_LIKELY(isSmall() && O.isSmall()))
    return getSmall() >= O.getSmall();
  return detail::SlowDynamicAPInt(*this) >= detail::SlowDynamicAPInt(O);
}

/// ---------------------------------------------------------------------------
/// Arithmetic operators.
/// ---------------------------------------------------------------------------

LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt
DynamicAPInt::operator+(const DynamicAPInt &O) const {
  if (LLVM_LIKELY(isSmall() && O.isSmall())) {
    DynamicAPInt Result;
    bool Overflow = AddOverflow(getSmall(), O.getSmall(), Result.getSmall());
    if (LLVM_LIKELY(!Overflow))
      return Result;
    return DynamicAPInt(detail::SlowDynamicAPInt(*this) +
                        detail::SlowDynamicAPInt(O));
  }
  return DynamicAPInt(detail::SlowDynamicAPInt(*this) +
                      detail::SlowDynamicAPInt(O));
}
LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt
DynamicAPInt::operator-(const DynamicAPInt &O) const {
  if (LLVM_LIKELY(isSmall() && O.isSmall())) {
    DynamicAPInt Result;
    bool Overflow = SubOverflow(getSmall(), O.getSmall(), Result.getSmall());
    if (LLVM_LIKELY(!Overflow))
      return Result;
    return DynamicAPInt(detail::SlowDynamicAPInt(*this) -
                        detail::SlowDynamicAPInt(O));
  }
  return DynamicAPInt(detail::SlowDynamicAPInt(*this) -
                      detail::SlowDynamicAPInt(O));
}
LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt
DynamicAPInt::operator*(const DynamicAPInt &O) const {
  if (LLVM_LIKELY(isSmall() && O.isSmall())) {
    DynamicAPInt Result;
    bool Overflow = MulOverflow(getSmall(), O.getSmall(), Result.getSmall());
    if (LLVM_LIKELY(!Overflow))
      return Result;
    return DynamicAPInt(detail::SlowDynamicAPInt(*this) *
                        detail::SlowDynamicAPInt(O));
  }
  return DynamicAPInt(detail::SlowDynamicAPInt(*this) *
                      detail::SlowDynamicAPInt(O));
}

// Division overflows only occur when negating the minimal possible value.
LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt
DynamicAPInt::divByPositive(const DynamicAPInt &O) const {
  assert(O > 0);
  if (LLVM_LIKELY(isSmall() && O.isSmall()))
    return DynamicAPInt(getSmall() / O.getSmall());
  return DynamicAPInt(detail::SlowDynamicAPInt(*this) /
                      detail::SlowDynamicAPInt(O));
}

LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt
DynamicAPInt::operator/(const DynamicAPInt &O) const {
  if (LLVM_LIKELY(isSmall() && O.isSmall())) {
    // Division overflows only occur when negating the minimal possible value.
    if (LLVM_UNLIKELY(divideSignedWouldOverflow(getSmall(), O.getSmall())))
      return -*this;
    return DynamicAPInt(getSmall() / O.getSmall());
  }
  return DynamicAPInt(detail::SlowDynamicAPInt(*this) /
                      detail::SlowDynamicAPInt(O));
}

LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt abs(const DynamicAPInt &X) {
  return DynamicAPInt(X >= 0 ? X : -X);
}
// Division overflows only occur when negating the minimal possible value.
LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt ceilDiv(const DynamicAPInt &LHS,
                                                  const DynamicAPInt &RHS) {
  if (LLVM_LIKELY(LHS.isSmall() && RHS.isSmall())) {
    if (LLVM_UNLIKELY(
            divideSignedWouldOverflow(LHS.getSmall(), RHS.getSmall())))
      return -LHS;
    return DynamicAPInt(divideCeilSigned(LHS.getSmall(), RHS.getSmall()));
  }
  return DynamicAPInt(
      ceilDiv(detail::SlowDynamicAPInt(LHS), detail::SlowDynamicAPInt(RHS)));
}
LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt floorDiv(const DynamicAPInt &LHS,
                                                   const DynamicAPInt &RHS) {
  if (LLVM_LIKELY(LHS.isSmall() && RHS.isSmall())) {
    if (LLVM_UNLIKELY(
            divideSignedWouldOverflow(LHS.getSmall(), RHS.getSmall())))
      return -LHS;
    return DynamicAPInt(divideFloorSigned(LHS.getSmall(), RHS.getSmall()));
  }
  return DynamicAPInt(
      floorDiv(detail::SlowDynamicAPInt(LHS), detail::SlowDynamicAPInt(RHS)));
}
// The RHS is always expected to be positive, and the result
/// is always non-negative.
LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt mod(const DynamicAPInt &LHS,
                                              const DynamicAPInt &RHS) {
  if (LLVM_LIKELY(LHS.isSmall() && RHS.isSmall()))
    return DynamicAPInt(mod(LHS.getSmall(), RHS.getSmall()));
  return DynamicAPInt(
      mod(detail::SlowDynamicAPInt(LHS), detail::SlowDynamicAPInt(RHS)));
}

LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt gcd(const DynamicAPInt &A,
                                              const DynamicAPInt &B) {
  assert(A >= 0 && B >= 0 && "operands must be non-negative!");
  if (LLVM_LIKELY(A.isSmall() && B.isSmall()))
    return DynamicAPInt(std::gcd(A.getSmall(), B.getSmall()));
  return DynamicAPInt(
      gcd(detail::SlowDynamicAPInt(A), detail::SlowDynamicAPInt(B)));
}

/// Returns the least common multiple of A and B.
LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt lcm(const DynamicAPInt &A,
                                              const DynamicAPInt &B) {
  DynamicAPInt X = abs(A);
  DynamicAPInt Y = abs(B);
  return (X * Y) / gcd(X, Y);
}

/// This operation cannot overflow.
LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt
DynamicAPInt::operator%(const DynamicAPInt &O) const {
  if (LLVM_LIKELY(isSmall() && O.isSmall()))
    return DynamicAPInt(getSmall() % O.getSmall());
  return DynamicAPInt(detail::SlowDynamicAPInt(*this) %
                      detail::SlowDynamicAPInt(O));
}

LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt DynamicAPInt::operator-() const {
  if (LLVM_LIKELY(isSmall())) {
    if (LLVM_LIKELY(getSmall() != std::numeric_limits<int32_t>::min()))
      return DynamicAPInt(-getSmall());
    return DynamicAPInt(-detail::SlowDynamicAPInt(*this));
  }
  return DynamicAPInt(-detail::SlowDynamicAPInt(*this));
}

/// ---------------------------------------------------------------------------
/// Assignment operators, preincrement, predecrement.
/// ---------------------------------------------------------------------------
LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt &
DynamicAPInt::operator+=(const DynamicAPInt &O) {
  if (LLVM_LIKELY(isSmall() && O.isSmall())) {
    int32_t Result = getSmall();
    bool Overflow = AddOverflow(getSmall(), O.getSmall(), Result);
    if (LLVM_LIKELY(!Overflow)) {
      getSmall() = Result;
      return *this;
    }
    // Note: this return is not strictly required but
    // removing it leads to a performance regression.
    return *this = DynamicAPInt(detail::SlowDynamicAPInt(*this) +
                                detail::SlowDynamicAPInt(O));
  }
  return *this = DynamicAPInt(detail::SlowDynamicAPInt(*this) +
                              detail::SlowDynamicAPInt(O));
}
LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt &
DynamicAPInt::operator-=(const DynamicAPInt &O) {
  if (LLVM_LIKELY(isSmall() && O.isSmall())) {
    int32_t Result = getSmall();
    bool Overflow = SubOverflow(getSmall(), O.getSmall(), Result);
    if (LLVM_LIKELY(!Overflow)) {
      getSmall() = Result;
      return *this;
    }
    // Note: this return is not strictly required but
    // removing it leads to a performance regression.
    return *this = DynamicAPInt(detail::SlowDynamicAPInt(*this) -
                                detail::SlowDynamicAPInt(O));
  }
  return *this = DynamicAPInt(detail::SlowDynamicAPInt(*this) -
                              detail::SlowDynamicAPInt(O));
}
LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt &
DynamicAPInt::operator*=(const DynamicAPInt &O) {
  if (LLVM_LIKELY(isSmall() && O.isSmall())) {
    int32_t Result = ValSmall;
    bool Overflow = MulOverflow(ValSmall, O.ValSmall, Result);
    if (LLVM_LIKELY(!Overflow)) {
      ValSmall = Result;
      return *this;
    }
    // Note: this return is not strictly required but
    // removing it leads to a performance regression.
    // return *this = DynamicAPInt(detail::SlowDynamicAPInt(*this) *
    //                             detail::SlowDynamicAPInt(O));
  }
  exit(2);
  // return *this = DynamicAPInt(detail::SlowDynamicAPInt(*this) *
  //                             detail::SlowDynamicAPInt(O));
}
LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt &
DynamicAPInt::operator/=(const DynamicAPInt &O) {
  if (LLVM_LIKELY(isSmall() && O.isSmall())) {
    // Division overflows only occur when negating the minimal possible value.
    if (LLVM_UNLIKELY(divideSignedWouldOverflow(getSmall(), O.getSmall())))
      return *this = -*this;
    getSmall() /= O.getSmall();
    return *this;
  }
  return *this = DynamicAPInt(detail::SlowDynamicAPInt(*this) /
                              detail::SlowDynamicAPInt(O));
}

// Division overflows only occur when the divisor is -1.
LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt &
DynamicAPInt::divByPositiveInPlace(const DynamicAPInt &O) {
  assert(O > 0);
  if (LLVM_LIKELY(isSmall() && O.isSmall())) {
    getSmall() /= O.getSmall();
    return *this;
  }
  return *this = DynamicAPInt(detail::SlowDynamicAPInt(*this) /
                              detail::SlowDynamicAPInt(O));
}

LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt &
DynamicAPInt::operator%=(const DynamicAPInt &O) {
  return *this = *this % O;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt &DynamicAPInt::operator++() {
  return *this += 1;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt &DynamicAPInt::operator--() {
  return *this -= 1;
}

/// ----------------------------------------------------------------------------
/// Convenience operator overloads for int32_t.
/// ----------------------------------------------------------------------------
LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt &operator+=(DynamicAPInt &A,
                                                      int32_t B) {
  // int32_t backup = A.getSmall();
  if (LLVM_LIKELY(A.isSmall())) {
    // int32_t result;
    bool Overflow = AddOverflow(A.ValSmall, B, A.ValSmall);
    if (LLVM_LIKELY(!Overflow)) {
      // A.ValSmall = result;
      return A;
    }
    // A.getSmall() = backup;
    // exit(2);
    // Note: this return is not strictly required but
    // removing it leads to a performance regression.
    // return A = DynamicAPInt(detail::SlowDynamicAPInt(backup) *
    //                             detail::SlowDynamicAPInt(B));
  }
  // llvm_unreachable("");
  // exit(2);
  // {
  // __sync_synchronize();
  // assert(false);
  A.ValSmall = bit_cast<int32_t>(bit_cast<uint32_t>(A.ValSmall) - bit_cast<uint32_t>(B));
  return A = DynamicAPInt(detail::SlowDynamicAPInt(A) + detail::SlowDynamicAPInt(B));
  // }
}
LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt &operator-=(DynamicAPInt &A,
                                                      int32_t B) {
  return A = A - B;
}

inline LLVM_ATTRIBUTE_NOINLINE DynamicAPInt &fallbackOpMulEqual(DynamicAPInt &A,
                                                      int32_t B) {
  return A = DynamicAPInt(detail::SlowDynamicAPInt(A) * detail::SlowDynamicAPInt(B));
}

LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt &operator*=(DynamicAPInt &A,
                                                      int32_t B) {
  // int32_t backup = A.getSmall();
  if (LLVM_LIKELY(A.isSmall())) {
    int32_t result;
    bool Overflow = MulOverflow(A.ValSmall, B, result);
    if (LLVM_LIKELY(!Overflow)) {
      A.ValSmall = result;
      return A;
    }
    // A.getSmall() = backup;
    // exit(2);
    // Note: this return is not strictly required but
    // removing it leads to a performance regression.
    // return A = DynamicAPInt(detail::SlowDynamicAPInt(backup) *
    //                             detail::SlowDynamicAPInt(B));
  }
  // llvm_unreachable("");
  exit(2);
  // {
  // return A = DynamicAPInt(detail::SlowDynamicAPInt(A) *
  //                         detail::SlowDynamicAPInt(B));
  // }
}

LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt &operator/=(DynamicAPInt &A,
                                                      int32_t B) {
  return A = A / B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt &operator%=(DynamicAPInt &A,
                                                      int32_t B) {
  return A = A % B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt operator+(DynamicAPInt &A,
                                                    int32_t B) {
  if (LLVM_LIKELY(A.isSmall())) {
    bool Overflow = AddOverflow(A.ValSmall, B, A.ValSmall);
    if (LLVM_LIKELY(!Overflow)) {
      DynamicAPInt Result(A.ValSmall);
      A.ValSmall -= B;
      return Result;
    }
    // return DynamicAPInt(detail::SlowDynamicAPInt(A) +
    //                     detail::SlowDynamicAPInt(B));
  }
  A.ValSmall -= B;
  return DynamicAPInt(detail::SlowDynamicAPInt(A) +
                      detail::SlowDynamicAPInt(B));
}
LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt operator-(const DynamicAPInt &A,
                                                    int32_t B) {
  return A - DynamicAPInt(B);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt operator*(const DynamicAPInt &A,
                                                    int32_t B) {
  return A * DynamicAPInt(B);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt operator/(const DynamicAPInt &A,
                                                    int32_t B) {
  return A / DynamicAPInt(B);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt operator%(const DynamicAPInt &A,
                                                    int32_t B) {
  return A % DynamicAPInt(B);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt operator+(int32_t A,
                                                    const DynamicAPInt &B) {
  return DynamicAPInt(A) + B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt operator-(int32_t A,
                                                    const DynamicAPInt &B) {
  return DynamicAPInt(A) - B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt operator*(int32_t A,
                                                    const DynamicAPInt &B) {
  return DynamicAPInt(A) * B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt operator/(int32_t A,
                                                    const DynamicAPInt &B) {
  return DynamicAPInt(A) / B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE DynamicAPInt operator%(int32_t A,
                                                    const DynamicAPInt &B) {
  return DynamicAPInt(A) % B;
}

/// We provide special implementations of the comparison operators rather than
/// calling through as above, as this would result in a 1.2x slowdown.
LLVM_ATTRIBUTE_ALWAYS_INLINE bool operator==(const DynamicAPInt &A, int32_t B) {
  if (LLVM_LIKELY(A.isSmall()))
    return A.getSmall() == B;
  return A.getLarge() == B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE bool operator!=(const DynamicAPInt &A, int32_t B) {
  if (LLVM_LIKELY(A.isSmall()))
    return A.getSmall() != B;
  return A.getLarge() != B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE bool operator>(const DynamicAPInt &A, int32_t B) {
  if (LLVM_LIKELY(A.isSmall()))
    return A.getSmall() > B;
  return A.getLarge() > B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE bool operator<(const DynamicAPInt &A, int32_t B) {
  if (LLVM_LIKELY(A.isSmall()))
    return A.getSmall() < B;
  return A.getLarge() < B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE bool operator<=(const DynamicAPInt &A, int32_t B) {
  if (LLVM_LIKELY(A.isSmall()))
    return A.getSmall() <= B;
  return A.getLarge() <= B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE bool operator>=(const DynamicAPInt &A, int32_t B) {
  if (LLVM_LIKELY(A.isSmall()))
    return A.getSmall() >= B;
  return A.getLarge() >= B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE bool operator==(int32_t A, const DynamicAPInt &B) {
  if (LLVM_LIKELY(B.isSmall()))
    return A == B.getSmall();
  return A == B.getLarge();
}
LLVM_ATTRIBUTE_ALWAYS_INLINE bool operator!=(int32_t A, const DynamicAPInt &B) {
  if (LLVM_LIKELY(B.isSmall()))
    return A != B.getSmall();
  return A != B.getLarge();
}
LLVM_ATTRIBUTE_ALWAYS_INLINE bool operator>(int32_t A, const DynamicAPInt &B) {
  if (LLVM_LIKELY(B.isSmall()))
    return A > B.getSmall();
  return A > B.getLarge();
}
LLVM_ATTRIBUTE_ALWAYS_INLINE bool operator<(int32_t A, const DynamicAPInt &B) {
  if (LLVM_LIKELY(B.isSmall()))
    return A < B.getSmall();
  return A < B.getLarge();
}
LLVM_ATTRIBUTE_ALWAYS_INLINE bool operator<=(int32_t A, const DynamicAPInt &B) {
  if (LLVM_LIKELY(B.isSmall()))
    return A <= B.getSmall();
  return A <= B.getLarge();
}
LLVM_ATTRIBUTE_ALWAYS_INLINE bool operator>=(int32_t A, const DynamicAPInt &B) {
  if (LLVM_LIKELY(B.isSmall()))
    return A >= B.getSmall();
  return A >= B.getLarge();
}
} // namespace llvm

#endif // LLVM_ADT_DYNAMICAPINT_H
