//===- SafeInteger.h - MLIR SafeInteger Class -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a simple class to represent fractions. It supports multiplication,
// comparison, floor, and ceiling operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_SAFE_INTEGER_H
#define MLIR_ANALYSIS_PRESBURGER_SAFE_INTEGER_H

#include "mlir/Support/MathExtras.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <iostream>
#include <limits>

namespace mlir {
namespace analysis {
namespace presburger {
using Int = __int128_t;

inline void overflowErrorIf(bool overflow) {
  if (overflow) {
    llvm::errs() << "Overflow!\n";
    abort();
  }
}

/// A class to overflow-aware 64-bit integers.
///
/// Overflows are asserted to not occur.
struct SafeInteger {
  /// Construct a SafeInteger from a numerator and denominator.
  SafeInteger(__int128_t oVal) {
    // overflowErrorIf(!(std::numeric_limits<Int>::min() <= oVal &&
    //                   oVal <= std::numeric_limits<Int>::max()));
    val = oVal;
  }

  /// Default constructor initializes the number to zero.
  SafeInteger() : SafeInteger(0) {}

  inline explicit operator bool();

  /// The stored value. This is always 64-bit.
  Int val;
};

inline bool operator<(const SafeInteger &x, const SafeInteger &y) {
  return x.val < y.val;
}
inline bool operator<=(const SafeInteger &x, const SafeInteger &y) {
  return x.val <= y.val;
}
inline bool operator==(const SafeInteger &x, const SafeInteger &y) {
  return x.val == y.val;
}
inline bool operator!=(const SafeInteger &x, const SafeInteger &y) {
  return x.val != y.val;
}
inline bool operator>(const SafeInteger &x, const SafeInteger &y) {
  return x.val > y.val;
}
inline bool operator>=(const SafeInteger &x, const SafeInteger &y) {
  return x.val >= y.val;
}

inline SafeInteger operator+(const SafeInteger &x, const SafeInteger &y) {
  Int result;
  bool overflow = __builtin_add_overflow(x.val, y.val, &result);
  overflowErrorIf(overflow);
  return SafeInteger(result);
}

inline SafeInteger operator-(const SafeInteger &x, const SafeInteger &y) {
  Int result;
  bool overflow = __builtin_sub_overflow(x.val, y.val, &result);
  overflowErrorIf(overflow);
  return SafeInteger(result);
}

inline SafeInteger operator-(const SafeInteger &x) {
  return SafeInteger(0) - x;
}

inline SafeInteger operator*(const SafeInteger &x, const SafeInteger &y) {
  Int result;
  bool overflow = __builtin_mul_overflow(x.val, y.val, &result);
  overflowErrorIf(overflow);
  return SafeInteger(result);
}

inline SafeInteger operator/(const SafeInteger &x, const SafeInteger &y) {
  // overflow only possible if y == -1
  if (y == SafeInteger(-1))
    return -x;
  return x.val / y.val;
}

inline SafeInteger operator%(const SafeInteger &x, const SafeInteger &y) {
  return x.val % y.val;
}

inline void operator+=(SafeInteger &x, const SafeInteger &y) { x = x + y; }

inline void operator-=(SafeInteger &x, const SafeInteger &y) { x = x - y; }

inline void operator*=(SafeInteger &x, const SafeInteger &y) { x = x * y; }

inline void operator/=(SafeInteger &x, const SafeInteger &y) { x = x / y; }

inline void operator%=(SafeInteger &x, const SafeInteger &y) { x = x % y; }

/// Returns MLIR's mod operation on constants. MLIR's mod operation yields the
/// remainder of the Euclidean division of 'lhs' by 'rhs', and is therefore not
/// C's % operator.  The RHS is always expected to be positive, and the result
/// is always non-negative.
inline SafeInteger mod(SafeInteger lhs, SafeInteger rhs) {
  assert(rhs >= 1);
  return lhs % rhs < 0 ? lhs % rhs + rhs : lhs % rhs;
}

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const SafeInteger &x) {
  if (x == 0) {
    os << "0";
    return os;
  }
  std::string out;
  auto copy = x;
  if (copy < 0) {
    os << '-';
    copy = -copy;
  }
  while (copy > 0) {
    out.push_back('0' + int(copy.val % 10));
    copy /= 10;
  }
  std::reverse(out.begin(), out.end());

  os << out;
  return os;
}

inline std::ostream &operator<<(std::ostream &os, const SafeInteger &x) {
  os << "[SafeInteger::operator<<(std::ostream) NYI";
  // os << x.val;
  return os;
}

inline SafeInteger ceilDiv(SafeInteger lhs, SafeInteger rhs) {
  assert(rhs >= 1);
  return lhs % rhs > 0 ? lhs / rhs + 1 : lhs / rhs;
}

inline SafeInteger floorDiv(SafeInteger lhs, SafeInteger rhs) {
  assert(rhs >= 1);
  return lhs % rhs < 0 ? lhs / rhs - 1 : lhs / rhs;
}

inline SafeInteger::operator bool() { return *this != 0; }

} // namespace presburger
} // namespace analysis
} // namespace mlir

namespace std {
using SafeInteger = mlir::analysis::presburger::SafeInteger;
inline SafeInteger abs(SafeInteger x) { return x < 0 ? -x : x; }

/// Returns the least common multiple of 'a' and 'b'.
inline SafeInteger lcm(SafeInteger a, SafeInteger b) {
  SafeInteger x = abs(a);
  SafeInteger y = abs(b);
  SafeInteger lcm = (x * y) / llvm::greatestCommonDivisor(x, y);
  return lcm;
}
}

#endif // MLIR_ANALYSIS_PRESBURGER_SAFE_INTEGER_H
