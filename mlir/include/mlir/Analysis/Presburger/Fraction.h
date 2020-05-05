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

template <typename T>
T floor(Fraction<T> f) {
  if (f.num >= 0)
    return f.num / f.den;
  else
    return (f.num / f.den) - (f.num % f.den != 0);
}

template <typename T>
T ceil(Fraction<T> f) {
  if (f.num <= 0)
    return f.num / f.den;
  else
    return (f.num / f.den) + (f.num % f.den != 0);
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
