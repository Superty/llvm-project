//===- SlowDynamicAPInt.h - SlowDynamicAPInt Class --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a simple class to represent arbitrary precision signed integers.
// Unlike APInt, one does not have to specify a fixed maximum size, and the
// integer can take on any arbitrary values.
//
// This class is to be used as a fallback slow path for the DynamicAPInt class,
// and is not intended to be used directly.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_SLOWDYNAMICAPINT_H
#define LLVM_ADT_SLOWDYNAMICAPINT_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
class LLVMDynamicAPInt;
// class DynamicAPInt;
} // namespace llvm

namespace llvm::detail {
/// A simple class providing dynamic arbitrary-precision arithmetic. Internally,
/// it stores an APInt, whose width is doubled whenever an overflow occurs at a
/// certain width. The default constructor sets the initial width to 64.
/// SlowDynamicAPInt is primarily intended to be used as a slow fallback path
/// for the upcoming DynamicAPInt class.
class SlowDynamicAPInt {
  APInt Val;

public:
  explicit SlowDynamicAPInt(int64_t Val);
  SlowDynamicAPInt();
  explicit SlowDynamicAPInt(const APInt &Val);
  SlowDynamicAPInt &operator=(int64_t Val);
  explicit operator int64_t() const;
  SlowDynamicAPInt operator-() const;
  bool operator==(const SlowDynamicAPInt &O) const;
  bool operator!=(const SlowDynamicAPInt &O) const;
  bool operator>(const SlowDynamicAPInt &O) const;
  bool operator<(const SlowDynamicAPInt &O) const;
  bool operator<=(const SlowDynamicAPInt &O) const;
  bool operator>=(const SlowDynamicAPInt &O) const;
  SlowDynamicAPInt operator+(const SlowDynamicAPInt &O) const;
  SlowDynamicAPInt operator-(const SlowDynamicAPInt &O) const;
  SlowDynamicAPInt operator*(const SlowDynamicAPInt &O) const;
  SlowDynamicAPInt operator/(const SlowDynamicAPInt &O) const;
  SlowDynamicAPInt operator%(const SlowDynamicAPInt &O) const;
  SlowDynamicAPInt &operator+=(const SlowDynamicAPInt &O);
  SlowDynamicAPInt &operator-=(const SlowDynamicAPInt &O);
  SlowDynamicAPInt &operator*=(const SlowDynamicAPInt &O);
  SlowDynamicAPInt &operator/=(const SlowDynamicAPInt &O);
  SlowDynamicAPInt &operator%=(const SlowDynamicAPInt &O);

  SlowDynamicAPInt &operator++();
  SlowDynamicAPInt &operator--();

  friend SlowDynamicAPInt abs(const SlowDynamicAPInt &X);
  friend SlowDynamicAPInt ceilDiv(const SlowDynamicAPInt &LHS,
                                  const SlowDynamicAPInt &RHS);
  friend SlowDynamicAPInt floorDiv(const SlowDynamicAPInt &LHS,
                                   const SlowDynamicAPInt &RHS);
  /// The operands must be non-negative for gcd.
  friend SlowDynamicAPInt gcd(const SlowDynamicAPInt &A,
                              const SlowDynamicAPInt &B);

  /// Overload to compute a hash_code for a SlowDynamicAPInt value.
  friend hash_code hash_value(const SlowDynamicAPInt &X); // NOLINT

  // Make DynamicAPInt a friend so it can access Val directly.
  friend LLVMDynamicAPInt;
  // friend DynamicAPInt;

  unsigned getBitWidth() const { return Val.getBitWidth(); }

  void print(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
};

inline raw_ostream &operator<<(raw_ostream &OS, const SlowDynamicAPInt &X) {
  X.print(OS);
  return OS;
}

/// Returns the remainder of dividing LHS by RHS.
///
/// The RHS is always expected to be positive, and the result
/// is always non-negative.
SlowDynamicAPInt mod(const SlowDynamicAPInt &LHS, const SlowDynamicAPInt &RHS);

/// Returns the least common multiple of A and B.
SlowDynamicAPInt lcm(const SlowDynamicAPInt &A, const SlowDynamicAPInt &B);

/// Redeclarations of friend declarations above to
/// make it discoverable by lookups.
SlowDynamicAPInt abs(const SlowDynamicAPInt &X);
SlowDynamicAPInt ceilDiv(const SlowDynamicAPInt &LHS,
                         const SlowDynamicAPInt &RHS);
SlowDynamicAPInt floorDiv(const SlowDynamicAPInt &LHS,
                          const SlowDynamicAPInt &RHS);
SlowDynamicAPInt gcd(const SlowDynamicAPInt &A, const SlowDynamicAPInt &B);
hash_code hash_value(const SlowDynamicAPInt &X); // NOLINT

/// ---------------------------------------------------------------------------
/// Convenience operator overloads for int64_t.
/// ---------------------------------------------------------------------------
SlowDynamicAPInt &operator+=(SlowDynamicAPInt &A, int64_t B);
SlowDynamicAPInt &operator-=(SlowDynamicAPInt &A, int64_t B);
SlowDynamicAPInt &operator*=(SlowDynamicAPInt &A, int64_t B);
SlowDynamicAPInt &operator/=(SlowDynamicAPInt &A, int64_t B);
SlowDynamicAPInt &operator%=(SlowDynamicAPInt &A, int64_t B);

bool operator==(const SlowDynamicAPInt &A, int64_t B);
bool operator!=(const SlowDynamicAPInt &A, int64_t B);
bool operator>(const SlowDynamicAPInt &A, int64_t B);
bool operator<(const SlowDynamicAPInt &A, int64_t B);
bool operator<=(const SlowDynamicAPInt &A, int64_t B);
bool operator>=(const SlowDynamicAPInt &A, int64_t B);
SlowDynamicAPInt operator+(const SlowDynamicAPInt &A, int64_t B);
SlowDynamicAPInt operator-(const SlowDynamicAPInt &A, int64_t B);
SlowDynamicAPInt operator*(const SlowDynamicAPInt &A, int64_t B);
SlowDynamicAPInt operator/(const SlowDynamicAPInt &A, int64_t B);
SlowDynamicAPInt operator%(const SlowDynamicAPInt &A, int64_t B);

bool operator==(int64_t A, const SlowDynamicAPInt &B);
bool operator!=(int64_t A, const SlowDynamicAPInt &B);
bool operator>(int64_t A, const SlowDynamicAPInt &B);
bool operator<(int64_t A, const SlowDynamicAPInt &B);
bool operator<=(int64_t A, const SlowDynamicAPInt &B);
bool operator>=(int64_t A, const SlowDynamicAPInt &B);
SlowDynamicAPInt operator+(int64_t A, const SlowDynamicAPInt &B);
SlowDynamicAPInt operator-(int64_t A, const SlowDynamicAPInt &B);
SlowDynamicAPInt operator*(int64_t A, const SlowDynamicAPInt &B);
SlowDynamicAPInt operator/(int64_t A, const SlowDynamicAPInt &B);
SlowDynamicAPInt operator%(int64_t A, const SlowDynamicAPInt &B);
} // namespace llvm::detail

namespace llvm {
LLVM_ATTRIBUTE_ALWAYS_INLINE detail::SlowDynamicAPInt dynamicAPIntFromInt64(int64_t X) {
  return detail::SlowDynamicAPInt(X);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE int64_t int64fromDynamicAPInt(const detail::SlowDynamicAPInt &X) {
  return int64_t(X);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE detail::SlowDynamicAPInt lcm(const detail::SlowDynamicAPInt &A,
                                              const detail::SlowDynamicAPInt &B) {
  return detail::lcm(A, B);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE detail::SlowDynamicAPInt gcd(const detail::SlowDynamicAPInt &A,
                                              const detail::SlowDynamicAPInt &B) {
  return detail::gcd(A, B);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE detail::SlowDynamicAPInt mod(const detail::SlowDynamicAPInt &A,
                                              const detail::SlowDynamicAPInt &B) {
  return detail::mod(A, B);
}

namespace detail {
LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt::SlowDynamicAPInt(int64_t Val)
    : Val(64, Val, /*isSigned=*/true) {}
    
LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt::SlowDynamicAPInt() : SlowDynamicAPInt(0) {}

LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt::SlowDynamicAPInt(const APInt &Val) : Val(Val) {}

LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt &SlowDynamicAPInt::operator=(int64_t Val) {
  return *this = SlowDynamicAPInt(Val);
}

LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt::operator int64_t() const { return Val.getSExtValue(); }

LLVM_ATTRIBUTE_ALWAYS_INLINE
hash_code hash_value(const SlowDynamicAPInt &X) {
  return hash_value(X.Val);
}

/// ---------------------------------------------------------------------------
/// Convenience operator overloads for int64_t.
/// ---------------------------------------------------------------------------
LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt &operator+=(SlowDynamicAPInt &A, int64_t B) {
  return A += SlowDynamicAPInt(B);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt &operator-=(SlowDynamicAPInt &A, int64_t B) {
  return A -= SlowDynamicAPInt(B);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt &operator*=(SlowDynamicAPInt &A, int64_t B) {
  return A *= SlowDynamicAPInt(B);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt &operator/=(SlowDynamicAPInt &A, int64_t B) {
  return A /= SlowDynamicAPInt(B);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt &operator%=(SlowDynamicAPInt &A, int64_t B) {
  return A %= SlowDynamicAPInt(B);
}

LLVM_ATTRIBUTE_ALWAYS_INLINE
bool operator==(const SlowDynamicAPInt &A, int64_t B) {
  return A == SlowDynamicAPInt(B);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE
bool operator!=(const SlowDynamicAPInt &A, int64_t B) {
  return A != SlowDynamicAPInt(B);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE
bool operator>(const SlowDynamicAPInt &A, int64_t B) {
  return A > SlowDynamicAPInt(B);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE
bool operator<(const SlowDynamicAPInt &A, int64_t B) {
  return A < SlowDynamicAPInt(B);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE
bool operator<=(const SlowDynamicAPInt &A, int64_t B) {
  return A <= SlowDynamicAPInt(B);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE
bool operator>=(const SlowDynamicAPInt &A, int64_t B) {
  return A >= SlowDynamicAPInt(B);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt operator+(const SlowDynamicAPInt &A, int64_t B) {
  return A + SlowDynamicAPInt(B);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt operator-(const SlowDynamicAPInt &A, int64_t B) {
  return A - SlowDynamicAPInt(B);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt operator*(const SlowDynamicAPInt &A, int64_t B) {
  return A * SlowDynamicAPInt(B);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt operator/(const SlowDynamicAPInt &A, int64_t B) {
  return A / SlowDynamicAPInt(B);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt operator%(const SlowDynamicAPInt &A, int64_t B) {
  return A % SlowDynamicAPInt(B);
}

LLVM_ATTRIBUTE_ALWAYS_INLINE
bool operator==(int64_t A, const SlowDynamicAPInt &B) {
  return SlowDynamicAPInt(A) == B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE
bool operator!=(int64_t A, const SlowDynamicAPInt &B) {
  return SlowDynamicAPInt(A) != B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE
bool operator>(int64_t A, const SlowDynamicAPInt &B) {
  return SlowDynamicAPInt(A) > B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE
bool operator<(int64_t A, const SlowDynamicAPInt &B) {
  return SlowDynamicAPInt(A) < B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE
bool operator<=(int64_t A, const SlowDynamicAPInt &B) {
  return SlowDynamicAPInt(A) <= B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE
bool operator>=(int64_t A, const SlowDynamicAPInt &B) {
  return SlowDynamicAPInt(A) >= B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt operator+(int64_t A, const SlowDynamicAPInt &B) {
  return SlowDynamicAPInt(A) + B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt operator-(int64_t A, const SlowDynamicAPInt &B) {
  return SlowDynamicAPInt(A) - B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt operator*(int64_t A, const SlowDynamicAPInt &B) {
  return SlowDynamicAPInt(A) * B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt operator/(int64_t A, const SlowDynamicAPInt &B) {
  return SlowDynamicAPInt(A) / B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt operator%(int64_t A, const SlowDynamicAPInt &B) {
  return SlowDynamicAPInt(A) % B;
}

static unsigned getMaxWidth(const APInt &A, const APInt &B) {
  return std::max(A.getBitWidth(), B.getBitWidth());
}

/// ---------------------------------------------------------------------------
/// Comparison operators.
/// ---------------------------------------------------------------------------


LLVM_ATTRIBUTE_ALWAYS_INLINE
// TODO: consider instead making APInt::compare available and using that.
bool SlowDynamicAPInt::operator==(const SlowDynamicAPInt &O) const {
  unsigned Width = getMaxWidth(Val, O.Val);
  return Val.sext(Width) == O.Val.sext(Width);
}

LLVM_ATTRIBUTE_ALWAYS_INLINE
bool SlowDynamicAPInt::operator!=(const SlowDynamicAPInt &O) const {
  unsigned Width = getMaxWidth(Val, O.Val);
  return Val.sext(Width) != O.Val.sext(Width);
}

LLVM_ATTRIBUTE_ALWAYS_INLINE
bool SlowDynamicAPInt::operator>(const SlowDynamicAPInt &O) const {
  unsigned Width = getMaxWidth(Val, O.Val);
  return Val.sext(Width).sgt(O.Val.sext(Width));
}

LLVM_ATTRIBUTE_ALWAYS_INLINE
bool SlowDynamicAPInt::operator<(const SlowDynamicAPInt &O) const {
  unsigned Width = getMaxWidth(Val, O.Val);
  return Val.sext(Width).slt(O.Val.sext(Width));
}

LLVM_ATTRIBUTE_ALWAYS_INLINE
bool SlowDynamicAPInt::operator<=(const SlowDynamicAPInt &O) const {
  unsigned Width = getMaxWidth(Val, O.Val);
  return Val.sext(Width).sle(O.Val.sext(Width));
}

LLVM_ATTRIBUTE_ALWAYS_INLINE
bool SlowDynamicAPInt::operator>=(const SlowDynamicAPInt &O) const {
  unsigned Width = getMaxWidth(Val, O.Val);
  return Val.sext(Width).sge(O.Val.sext(Width));
}

/// ---------------------------------------------------------------------------
/// Arithmetic operators.
/// ---------------------------------------------------------------------------

/// Bring a and b to have the same width and then call op(a, b, overflow).
/// If the overflow bit becomes set, resize a and b to double the width and
/// call op(a, b, overflow), returning its result. The operation with double
/// widths should not also overflow.
LLVM_ATTRIBUTE_ALWAYS_INLINE
APInt runOpWithExpandOnOverflow(
    const APInt &A, const APInt &B,
    function_ref<APInt(const APInt &, const APInt &, bool &Overflow)> Op) {
  bool Overflow;
  unsigned Width = getMaxWidth(A, B);
  APInt Ret = Op(A.sext(Width), B.sext(Width), Overflow);
  if (!Overflow)
    return Ret;

  Width *= 2;
  Ret = Op(A.sext(Width), B.sext(Width), Overflow);
  assert(!Overflow && "double width should be sufficient to avoid overflow!");
  return Ret;
}


LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt SlowDynamicAPInt::operator+(const SlowDynamicAPInt &O) const {
  return SlowDynamicAPInt(
      runOpWithExpandOnOverflow(Val, O.Val, std::mem_fn(&APInt::sadd_ov)));
}

LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt SlowDynamicAPInt::operator-(const SlowDynamicAPInt &O) const {
  return SlowDynamicAPInt(
      runOpWithExpandOnOverflow(Val, O.Val, std::mem_fn(&APInt::ssub_ov)));
}

LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt SlowDynamicAPInt::operator*(const SlowDynamicAPInt &O) const {
  return SlowDynamicAPInt(
      runOpWithExpandOnOverflow(Val, O.Val, std::mem_fn(&APInt::smul_ov)));
}

LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt SlowDynamicAPInt::operator/(const SlowDynamicAPInt &O) const {
  return SlowDynamicAPInt(
      runOpWithExpandOnOverflow(Val, O.Val, std::mem_fn(&APInt::sdiv_ov)));
}

LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt abs(const SlowDynamicAPInt &X) {
  return X >= 0 ? X : -X;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt ceilDiv(const SlowDynamicAPInt &LHS,
                                 const SlowDynamicAPInt &RHS) {
  if (RHS == -1)
    return -LHS;
  unsigned Width = getMaxWidth(LHS.Val, RHS.Val);
  return SlowDynamicAPInt(APIntOps::RoundingSDiv(
      LHS.Val.sext(Width), RHS.Val.sext(Width), APInt::Rounding::UP));
}

LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt floorDiv(const SlowDynamicAPInt &LHS,
                                  const SlowDynamicAPInt &RHS) {
  if (RHS == -1)
    return -LHS;
  unsigned Width = getMaxWidth(LHS.Val, RHS.Val);
  return SlowDynamicAPInt(APIntOps::RoundingSDiv(
      LHS.Val.sext(Width), RHS.Val.sext(Width), APInt::Rounding::DOWN));
}
// The RHS is always expected to be positive, and the result
/// is always non-negative.
LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt mod(const SlowDynamicAPInt &LHS,
                             const SlowDynamicAPInt &RHS) {
  assert(RHS >= 1 && "mod is only supported for positive divisors!");
  return LHS % RHS < 0 ? LHS % RHS + RHS : LHS % RHS;
}

LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt gcd(const SlowDynamicAPInt &A,
                             const SlowDynamicAPInt &B) {
  assert(A >= 0 && B >= 0 && "operands must be non-negative!");
  unsigned Width = getMaxWidth(A.Val, B.Val);
  return SlowDynamicAPInt(
      APIntOps::GreatestCommonDivisor(A.Val.sext(Width), B.Val.sext(Width)));
}

/// Returns the least common multiple of A and B.
LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt lcm(const SlowDynamicAPInt &A,
                             const SlowDynamicAPInt &B) {
  SlowDynamicAPInt X = abs(A);
  SlowDynamicAPInt Y = abs(B);
  return (X * Y) / gcd(X, Y);
}

/// This operation cannot overflow.

LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt SlowDynamicAPInt::operator%(const SlowDynamicAPInt &O) const {
  unsigned Width = std::max(Val.getBitWidth(), O.Val.getBitWidth());
  return SlowDynamicAPInt(Val.sext(Width).srem(O.Val.sext(Width)));
}


LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt SlowDynamicAPInt::operator-() const {
  if (Val.isMinSignedValue()) {
    /// Overflow only occurs when the value is the minimum possible value.
    APInt Ret = Val.sext(2 * Val.getBitWidth());
    return SlowDynamicAPInt(-Ret);
  }
  return SlowDynamicAPInt(-Val);
}

/// ---------------------------------------------------------------------------
/// Assignment operators, preincrement, predecrement.
/// ---------------------------------------------------------------------------

LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt &SlowDynamicAPInt::operator+=(const SlowDynamicAPInt &O) {
  *this = *this + O;
  return *this;
}

LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt &SlowDynamicAPInt::operator-=(const SlowDynamicAPInt &O) {
  *this = *this - O;
  return *this;
}

LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt &SlowDynamicAPInt::operator*=(const SlowDynamicAPInt &O) {
  *this = *this * O;
  return *this;
}

LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt &SlowDynamicAPInt::operator/=(const SlowDynamicAPInt &O) {
  *this = *this / O;
  return *this;
}

LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt &SlowDynamicAPInt::operator%=(const SlowDynamicAPInt &O) {
  *this = *this % O;
  return *this;
}

LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt &SlowDynamicAPInt::operator++() {
  *this += 1;
  return *this;
}


LLVM_ATTRIBUTE_ALWAYS_INLINE
SlowDynamicAPInt &SlowDynamicAPInt::operator--() {
  *this -= 1;
  return *this;
}

/// ---------------------------------------------------------------------------
/// Printing.
/// ---------------------------------------------------------------------------

LLVM_ATTRIBUTE_ALWAYS_INLINE
void SlowDynamicAPInt::print(raw_ostream &OS) const { OS << Val; }


LLVM_ATTRIBUTE_ALWAYS_INLINE
void SlowDynamicAPInt::dump() const { print(dbgs()); }
} // namespace detail

 
} // namespace llvm 

#endif // LLVM_ADT_SLOWDYNAMICAPINT_H
