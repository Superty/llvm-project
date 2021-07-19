//===- Constraint.h - MLIR Constraint Class -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Constraint class. Supports inequality, equality, and division constraints.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_CONSTRAINT_H
#define MLIR_ANALYSIS_PRESBURGER_CONSTRAINT_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

class Constraint {
public:
  Constraint() = delete;

  unsigned getNumDims() const {
    // The last element of the coefficient vector is the constant term and does
    // not correspond to any dimension.
    return coeffs.size() - 1;
  }

  /// Insert `count` empty dimensions before the `pos`th dimension, or at the
  /// end if `pos` is equal to `getNumDims()`.
  void insertDimensions(unsigned pos, unsigned count) {
    assert(pos <= getNumDims());
    coeffs.insert(coeffs.begin() + pos, count, 0);
  }

  /// Erase `count` dimensions starting at the `pos`-th one.
  void eraseDimensions(unsigned pos, unsigned count) {
    assert(pos + count - 1 < getNumDims() &&
           "Dimension to be erased does not exist!");
    coeffs.erase(coeffs.begin() + pos, coeffs.begin() + pos + count);
  }

  ArrayRef<int64_t> getCoeffs() const {
    return coeffs;
  }

  void shiftToOrigin() {
    coeffs.back() = 0;
  }

  void substitute(ArrayRef<int64_t> values) {
    assert(values.size() <= getNumDims() && "Too many values to substitute!");
    for (size_t i = 0; i < values.size(); i++)
      coeffs.back() += values[i] * coeffs[i];

    coeffs = SmallVector<int64_t, 8>(coeffs.begin() + values.size(), coeffs.end());
  }

  static bool sameConstraint(const Constraint &a, const Constraint &b) {
    if (a.coeffs.size() != b.coeffs.size())
      return false;
    for (unsigned i = 0; i < a.coeffs.size(); ++i) {
      if (a.coeffs[i] != b.coeffs[i])
        return false;
    }

    return true;
  }

  int64_t getConstant() const { return coeffs.back(); }

  void shift(int64_t x) { coeffs.back() += x; }

  void setCoeff(unsigned var, int64_t constant) { coeffs[var] = constant; }

  void shiftCoeff(unsigned var, int64_t constant) { coeffs[var] += constant; }

  /// Swaps the elements in the range [first, last) in such a way that the
  /// element nFirst becomes the first element of the new range and nFirst - 1
  /// becomes the last element.
  void rotate(unsigned first, unsigned nFirst, unsigned last) {
    std::rotate(coeffs.begin() + first, coeffs.begin() + nFirst,
                coeffs.begin() + last);
  }

  // Shift each coefficent by coeffShifts[i] * constant
  void shiftCoeffs(const ArrayRef<int64_t> &coeffShifts, int64_t constant) {
    assert(coeffShifts.size() - 1 == getNumDims() &&
           "Incorrect number of dimensions");

    for (unsigned i = 0; i < coeffs.size(); ++i)
      coeffs[i] += coeffShifts[i] * constant;
  }

  /// Swap coefficients at position vari, varj
  void swapCoeffs(unsigned vari, unsigned varj) {
    std::swap(coeffs[vari], coeffs[varj]);
  }

  /// Factor out the greatest common divisor of coefficents
  void normalizeCoeffs() {
    int64_t currGcd = 0;
    for (int64_t &coeff : coeffs)
      currGcd = llvm::greatestCommonDivisor(currGcd, std::abs(coeff));

    if (currGcd > 1) {
      for (int64_t &coeff : coeffs) 
        coeff /= currGcd;
    }
  }

  bool isTrivial() const {
    for (unsigned i = 0; i < coeffs.size() - 1; ++i)
      if (coeffs[i] != 0)
        return false;
    return true;
  }

  void appendDimension() {
    insertDimensions(getNumDims(), 1);
  }

  void removeLastDimension() {
    eraseDimensions(getNumDims() - 1, 1);
  }

  void print(raw_ostream &os) const {
    bool first = true;
    if (coeffs.back() != 0) {
      os << coeffs.back();
      first = false;
    }

    bool printed = false;
    for (unsigned i = 0; i < coeffs.size() - 1; ++i) {
      if (coeffs[i] == 0)
        continue;
      printed = true;

      if (first) {
        if (coeffs[i] == -1)
          os << '-';
        else if (coeffs[i] != 1)
          os << coeffs[i];
        first = false;
      } else if (coeffs[i] > 0) {
        os << " + ";
        if (coeffs[i] != 1)
          os << coeffs[i];
      } else {
        os << " - " << -coeffs[i];
        if (-coeffs[i] != 1)
          os << -coeffs[i];
      }
      
      os << "x" << i;
    }

    if (!printed) 
      os << coeffs.back();
  }

  void dump() const { print(llvm::errs()); }

  void dumpCoeffs() const {
    for (auto coeff : coeffs) {
      llvm::errs() << coeff << ' ';
    }
    llvm::errs() << '\n';
  }

protected:
  Constraint(ArrayRef<int64_t> oCoeffs) : coeffs(oCoeffs.begin(), oCoeffs.end()) {}
  SmallVector<int64_t, 8> coeffs;
};

class InequalityConstraint : public Constraint {
public:
  InequalityConstraint(ArrayRef<int64_t> oCoeffs) : Constraint(oCoeffs) {}
  void print(raw_ostream &os) const {
    Constraint::print(os);
    os << " >= 0";
  }
  void dump() const { print(llvm::errs()); }
};

class EqualityConstraint : public Constraint {
public:
  EqualityConstraint(ArrayRef<int64_t> oCoeffs) : Constraint(oCoeffs) {}
  void print(raw_ostream &os) const {
    Constraint::print(os);
    os << " = 0";
  }
  void dump() const { print(llvm::errs()); }
};

class DivisionConstraint : public Constraint {
public:
  DivisionConstraint(ArrayRef<int64_t> oCoeffs, int64_t oDenom, unsigned oVariable)
    : Constraint(oCoeffs), denom(oDenom), variable(oVariable) {}
  void print(raw_ostream &os) const {
    os << "x" << variable << " = floor((";
    Constraint::print(os);
    os << ")/" << denom << ')';
  }

  int64_t getDenominator() const {
    return denom;
  }

  unsigned getVariable() const {
    return variable;
  }

  InequalityConstraint getInequalityLowerBound() const {
    SmallVector<int64_t, 8> ineqCoeffs = coeffs;
    ineqCoeffs[variable] -= denom;
    return InequalityConstraint(ineqCoeffs);
  }

  InequalityConstraint getInequalityUpperBound() const {
    SmallVector<int64_t, 8> ineqCoeffs;
    ineqCoeffs.reserve(coeffs.size());
    for (int64_t coeff : coeffs)
      ineqCoeffs.push_back(-coeff);
    ineqCoeffs[variable] += denom;
    ineqCoeffs.back() += denom - 1;
    return InequalityConstraint(ineqCoeffs);
  }

  void insertDimensions(unsigned pos, unsigned count) {
    if (pos <= variable)
      variable += count;
    Constraint::insertDimensions(pos, count);
  }

  void eraseDimensions(unsigned pos, unsigned count) {
    if (pos < variable)
      variable -= count;
    Constraint::eraseDimensions(pos, count);
  }

  void substitute(ArrayRef<int64_t> values) {
    assert(variable >= values.size() && "Not yet implemented");
  }
 
  /// Swaps the "variable" property of two division constraints
  static void swapVariables(DivisionConstraint &diva,
                            DivisionConstraint &divb) {
    std::swap(diva.variable, divb.variable);
  }

  /// Remove common factor in numerator and denominator not taking into account
  /// the constant term.
  /// 
  /// Given divisions: [m * (f(x) + c) / m * d]
  /// Replace it by: [f(x) + floor(c/m) / d]
  /// 
  /// The constant term need not have the same common factor since the
  /// difference is (c / m) / d which satisfies 0 <= (c / m) / d < 1 / d 
  /// and therefore cannot influence the result.
  void removeCommonFactor() {
    int64_t currGcd = std::abs(denom);
    for (unsigned i = 0; i < coeffs.size() - 1; ++i)
      currGcd = llvm::greatestCommonDivisor(currGcd, std::abs(coeffs[i]));

    if (currGcd > 1) {
      for (int64_t &coeff : coeffs)
        coeff /= currGcd;
      denom /= currGcd;
    }
  }

  /// Checks if coefficients of two divisions are same. Assumes that the
  /// divisions were normalized before.
  /// This function is only useful if the divisions are ordered.
  static bool sameDivision(DivisionConstraint &div1, DivisionConstraint &div2) {
    if (div1.getDenominator() != div2.getDenominator())
      return false;

    for (unsigned i = 0; i < div1.coeffs.size(); --i) {
      if (div1.coeffs[i] != div2.coeffs[i])
        return false;
    }

    return true;
  }

  void printVar() {
    llvm::errs() << variable << "\n";
  }

  void dump() const { print(llvm::errs()); }
private:
  int64_t denom;
  unsigned variable;
};
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_CONSTRAINT_H
