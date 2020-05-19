//===- Simplex.h - MLIR Fraction Class --------------------------*- C++ -*-===//
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

#ifndef MLIR_ANALYSIS_PRESBURGER_FRACTION_H
#define MLIR_ANALYSIS_PRESBURGER_FRACTION_H

namespace mlir {

template <typename T>
struct Fraction {
  Fraction() : num(0), den(1) {}
  Fraction(T oNum, T oDen) : num(oNum), den(oDen) {
    if (den < 0) {
      num = -num;
      den = -den;
    }
  }
  T num, den;
};

template <typename T>
int compare(const Fraction<T> &x, const Fraction<T> &y) {
  auto diff = x.num * y.den - y.num * x.den;
  if (diff > 0)
    return +1;
  if (diff < 0)
    return -1;
  return 0;
}

// Division rounds towards zero, so if the result is already non-negative
// then num/den is the floor. If the result is negative and the division
// leaves a remainder then we need to subtract one to get the floor.
template <typename T>
T floor(Fraction<T> f) {
  return f.num / f.den - (f.num < 0 && f.num % f.den != 0);
}

// Division rounds towards zero, so if the result is already non-positive
// then num/den is the ceiling. If the result is positive and the division
// leaves a remainder then we need to add one to get the ceiling.
template <typename T>
T ceil(Fraction<T> f) {
  return f.num / f.den + (f.num > 0 && f.num % f.den != 0);
}

template <typename T>
int compare(const T &x, const Fraction<T> &y) {
  return sign(x * y.den - y.num);
}

template <typename T>
int compare(const Fraction<T> &x, const T &y) {
  return -compare(y, x);
}

template <typename T>
Fraction<T> operator-(const Fraction<T> &x) {
  return Fraction<T>(-x.num, x.den);
}

template <typename T, typename U>
bool operator<(const T &x, const Fraction<U> &y) {
  return compare(x, y) < 0;
}

template <typename T, typename U>
bool operator<=(const T &x, const Fraction<U> &y) {
  return compare(x, y) <= 0;
}

template <typename T, typename U>
bool operator==(const T &x, const Fraction<U> &y) {
  return compare(x, y) == 0;
}
template <typename T, typename U>
bool operator>(const T &x, const Fraction<U> &y) {
  return compare(x, y) > 0;
}

template <typename T, typename U>
bool operator>=(const T &x, const Fraction<U> &y) {
  return compare(x, y) >= 0;
}

template <typename T>
Fraction<T> operator*(const Fraction<T> &x, const Fraction<T> &y) {
  return Fraction<T>(x.num * y.num, x.den * y.den);
}
} // namespace mlir
#endif // MLIR_ANALYSIS_PRESBURGER_FRACTION_H
