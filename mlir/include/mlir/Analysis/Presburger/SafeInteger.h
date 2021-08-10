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
#include <stdexcept>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <gmpxx.h>

namespace mlir {
namespace analysis {
namespace presburger {

template <typename T, typename U, unsigned S>
SmallVector<T, S> convert(const SmallVector<U, S> &v) {
  SmallVector<T, S> res;
  for (const auto &elem : v)
    res.push_back(elem);
  return res;
}

// template <typename Vector>
// struct SafeVector {
//   SafeVector(const Vector &x) : vec(x) {};
//   SafeVector(Vector &&x) : vec(x) {};
//   Vector vec;
// };

using DefaultInt = mpz_class;

/// A class to overflow-aware 64-bit integers.
template <typename Int>
struct SafeInteger {
  using UnderlyingInt = Int;
  /// Construct a SafeInteger<Int> from an Int.
  /// Note that if this was the constructor for Int = int16_t, then SafeInteger<int16_t>(123) causes problems as 123 is an int and convering it to int16_t narrows it.
  /// Therefore we only use this when Int i snot int16_t (a hack; it should also be disabled for int8_t but we never use that anyway). For int16_t we provide a different constructor
  /// instead that takes an int32_t as argument.
  /// Because integer constants (without any suffixes) are ints by default
  template <typename IntCopy = Int, std::enable_if_t<(std::is_integral<IntCopy>::value || std::is_same<IntCopy, __int128_t>::value || std::is_same<IntCopy, mpz_class>::value) && !std::is_same<IntCopy, int16_t>::value, bool> = true>
  SafeInteger(IntCopy oVal) : val(oVal) {}
  template <typename IntCopy = Int, std::enable_if_t<std::is_same<IntCopy, int16_t>::value, bool> = true>
  SafeInteger(IntCopy oVal) {
    Int min = std::numeric_limits<Int>::min();
    Int max = std::numeric_limits<Int>::max();
    throwOverflowIf(!(min <= oVal && oVal <= max));
    val = oVal;
  }

  explicit operator Int() {
    return val;
  }

  template <typename OInt>
  SafeInteger(const SafeInteger<OInt> &o) : val(o.val) {
    static_assert(sizeof(Int) >= sizeof(OInt));
  }

  /// Default constructor initializes the number to zero.
  SafeInteger() : SafeInteger(0) {}

  inline explicit operator bool();

  /// The stored value.
  Int val;

  static void throwOverflowIf(bool cond) {
    if (cond)
      throw std::overflow_error("Overflow!");
  }
};

template <typename Int>
using SafeInt = typename std::conditional<std::is_same<Int, mpz_class>::value,
mpz_class,
SafeInteger<Int>>::type;

template <typename Int>
inline bool operator<(const SafeInteger<Int> &x, const SafeInteger<Int> &y) {
  return x.val < y.val;
}
template <typename Int>
inline bool operator<=(const SafeInteger<Int> &x, const SafeInteger<Int> &y) {
  return x.val <= y.val;
}
template <typename Int>
inline bool operator==(const SafeInteger<Int> &x, const SafeInteger<Int> &y) {
  return x.val == y.val;
}
template <typename Int>
inline bool operator!=(const SafeInteger<Int> &x, const SafeInteger<Int> &y) {
  return x.val != y.val;
}
template <typename Int>
inline bool operator>(const SafeInteger<Int> &x, const SafeInteger<Int> &y) {
  return x.val > y.val;
}
template <typename Int>
inline bool operator>=(const SafeInteger<Int> &x, const SafeInteger<Int> &y) {
  return x.val >= y.val;
}

template <typename Int>
inline bool operator<(const SafeInteger<Int> &x, int y) {
  return x.val < y;
}
template <typename Int>
inline bool operator<=(const SafeInteger<Int> &x, int y) {
  return x.val <= y;
}
template <typename Int>
inline bool operator==(const SafeInteger<Int> &x, int y) {
  return x.val == y;
}
template <typename Int>
inline bool operator!=(const SafeInteger<Int> &x, int y) {
  return x.val != y;
}
template <typename Int>
inline bool operator>(const SafeInteger<Int> &x, int y) {
  return x.val > y;
}
template <typename Int>
inline bool operator>=(const SafeInteger<Int> &x, int y) {
  return x.val >= y;
}

template <typename Int>
inline SafeInteger<Int> operator+(const SafeInteger<Int> &x, const SafeInteger<Int> &y) {
  if constexpr (std::is_same<Int, mpz_class>::value) {
    Int result = x.val + y.val;
    return SafeInteger<Int>(result);
  } else {
    Int result;
    bool overflow = __builtin_add_overflow(x.val, y.val, &result);
    SafeInteger<Int>::throwOverflowIf(overflow);
    return SafeInteger<Int>(result);
  }
}

template <typename Int>
inline SafeInteger<Int> operator-(const SafeInteger<Int> &x, const SafeInteger<Int> &y) {
  if constexpr (std::is_same<Int, mpz_class>::value) {
    Int result = x.val - y.val;
    return SafeInteger<Int>(result);
  } else {
    Int result;
    bool overflow = __builtin_sub_overflow(x.val, y.val, &result);
    SafeInteger<Int>::throwOverflowIf(overflow);
    return SafeInteger<Int>(result);
  }
}

template <typename Int>
inline SafeInteger<Int> operator-(const SafeInteger<Int> &x) {
  if constexpr (std::is_same<Int, mpz_class>::value) {
    Int result = -x.val;
    return SafeInteger<Int>(result);
  } else {
    return SafeInteger<Int>(0) - x;
  }
}

template <typename Int>
inline SafeInteger<Int> operator*(const SafeInteger<Int> &x, const SafeInteger<Int> &y) {
  if constexpr (std::is_same<Int, mpz_class>::value) {
    Int result = x.val * y.val;
    return SafeInteger<Int>(result);
  } else {
    Int result;
    bool overflow = __builtin_mul_overflow(x.val, y.val, &result);
    SafeInteger<Int>::throwOverflowIf(overflow);
    return SafeInteger<Int>(result);
  }
}

template <typename Int>
inline SafeInteger<Int> operator/(const SafeInteger<Int> &x, const SafeInteger<Int> &y) {
  if constexpr (std::is_same<Int, mpz_class>::value) {
    Int result = x.val / y.val;
    return SafeInteger<Int>(result);
  } else {
    if (y.val == -1)
      return -x;
    Int result = x.val / y.val;
    return SafeInteger<Int>(result);
  }
}

template <typename Int>
inline SafeInteger<Int> operator%(const SafeInteger<Int> &x, const SafeInteger<Int> &y) {
  Int result = x.val % y.val;
  return SafeInteger<Int>(result);
}

template <typename Int>
inline void operator+=(SafeInteger<Int> &x, const SafeInteger<Int> &y) { x = x + y; }

template <typename Int>
inline void operator-=(SafeInteger<Int> &x, const SafeInteger<Int> &y) { x = x - y; }

template <typename Int>
inline void operator*=(SafeInteger<Int> &x, const SafeInteger<Int> &y) { x = x * y; }

template <typename Int>
inline void operator/=(SafeInteger<Int> &x, const SafeInteger<Int> &y) { x = x / y; }

template <typename Int>
inline void operator%=(SafeInteger<Int> &x, const SafeInteger<Int> &y) { x = x % y; }

template <typename Int>
inline void operator+=(SafeInteger<Int> &x, int y) { x = x + SafeInteger<Int>(y); }

template <typename Int>
inline void operator-=(SafeInteger<Int> &x, int y) { x = x - SafeInteger<Int>(y); }

template <typename Int>
inline void operator*=(SafeInteger<Int> &x, int y) { x = x * SafeInteger<Int>(y); }

template <typename Int>
inline void operator/=(SafeInteger<Int> &x, int y) { x = x / SafeInteger<Int>(y); }

template <typename Int>
inline void operator%=(SafeInteger<Int> &x, int y) { x = x % SafeInteger<Int>(y); }

template <typename Int>
inline SafeInteger<Int> operator+(const SafeInteger<Int> &x, int y) { return x + SafeInteger<Int>(y); }

template <typename Int>
inline SafeInteger<Int> operator-(const SafeInteger<Int> &x, int y) { return x - SafeInteger<Int>(y); }

template <typename Int>
inline SafeInteger<Int> operator*(const SafeInteger<Int> &x, int y) { return x * SafeInteger<Int>(y); }

template <typename Int>
inline SafeInteger<Int> operator/(const SafeInteger<Int> &x, int y) { return x / SafeInteger<Int>(y); }

template <typename Int>
inline SafeInteger<Int> operator%(const SafeInteger<Int> &x, int y) { return x % SafeInteger<Int>(y); }
/// Returns MLIR's mod operation on constants. MLIR's mod operation yields the
/// remainder of the Euclidean division of 'lhs' by 'rhs', and is therefore not
/// C's % operator.  The RHS is always expected to be positive, and the result
/// is always non-negative.
template <typename Int>
inline Int mod(Int lhs, Int rhs) {
  assert(rhs >= 1);
  return lhs % rhs < 0 ? lhs % rhs + rhs : lhs % rhs;
}
inline mpz_class mod(mpz_class lhs, mpz_class rhs) {
  assert(rhs >= 1);
  return lhs % rhs < 0 ? mpz_class(lhs % rhs + rhs) : lhs % rhs;
}

template <typename Int>
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const SafeInteger<Int> &x) {
  if constexpr (std::is_same<Int, __int128_t>::value) {
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
      copy /= SafeInteger<Int>(10);
    }
    std::reverse(out.begin(), out.end());

    os << out;
  } else if constexpr (std::is_same<Int, mpz_class>::value) {
    std::stringstream ss;
    ss << x.val;
    os << ss.str();
  } else {
    os << x.val;
  }
  return os;
}

llvm::raw_ostream &operator<< (llvm::raw_ostream &os, const mpz_class &x) {
  std::stringstream ss;
  ss << x;
  os << ss.str();
  return os;
}

template <typename Int>
inline std::ostream &operator<<(std::ostream &os, const SafeInteger<Int> &x) {
  os << "[SafeInteger<Int>::operator<<(std::ostream) NYI";
  // os << x.val;
  return os;
}

template <typename Int>
inline Int ceilDiv(Int lhs, Int rhs) {
  assert(rhs >= 1);
  return lhs % rhs > 0 ? lhs / rhs + 1 : lhs / rhs;
}

template <typename Int>
inline Int floorDiv(Int lhs, Int rhs) {
  assert(rhs >= 1);
  return lhs % rhs < 0 ? lhs / rhs - 1 : lhs / rhs;
}

inline mpz_class ceilDiv(mpz_class lhs, mpz_class rhs) {
  assert(rhs >= 1);
  return lhs % rhs > 0 ? mpz_class(lhs / rhs + 1) : lhs / rhs;
}

inline mpz_class floorDiv(mpz_class lhs, mpz_class rhs) {
  assert(rhs >= 1);
  return lhs % rhs < 0 ? mpz_class(lhs / rhs - 1) : lhs / rhs;
}

template <typename Int>
inline SafeInteger<Int>::operator bool() { return *this != 0; }

} // namespace presburger
} // namespace analysis
} // namespace mlir

namespace std {

template <typename Int>
using SafeInteger = mlir::analysis::presburger::SafeInteger<Int>;

template <typename Int>
inline Int abs(Int x) { return x < 0 ? -x : x; }
inline mpz_class abs(mpz_class x) { return x < 0 ? -x : x; }

/// Returns the least common multiple of 'a' and 'b'.
template <typename Int>
inline Int lcm(Int a, Int b) {
  Int x = abs(a);
  Int y = abs(b);
  Int lcm = (x * y) / llvm::greatestCommonDivisor(x, y);
  return lcm;
}
inline mpz_class lcm(mpz_class a, mpz_class b) {
  mpz_class x = abs(a);
  mpz_class y = abs(b);
  mpz_class lcm = (x * y) / llvm::greatestCommonDivisor(x, y);
  return lcm;
}
} // namespace std

#endif // MLIR_ANALYSIS_PRESBURGER_SAFE_INTEGER_H
