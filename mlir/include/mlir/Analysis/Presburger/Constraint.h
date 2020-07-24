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
#include "llvm/Support/raw_ostream.h"

namespace mlir {

class Constraint {
public:
  Constraint() = delete;

  void appendNewDimension() {
    coeffs.push_back(coeffs.back());
    coeffs[coeffs.size() - 2] = 0;
  }

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

  void shift(int64_t x) {
    coeffs.back() += x;
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

    for (unsigned i = 0; i < coeffs.size() - 1; ++i) {
      if (coeffs[i] == 0)
        continue;

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
  }

  void dump() const { print(llvm::errs()); }

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

  void insertDimensions(unsigned pos, unsigned count) {
    if (pos <= variable)
      variable += count;
    Constraint::insertDimensions(pos, count);
  }

  void eraseDimensions(unsigned pos, unsigned count) {
    assert(!(pos <= variable && variable < pos + count) &&
           "cannot erase division variable!");
    Constraint::eraseDimensions(pos, count);
  }

  void substitute(ArrayRef<int64_t> values) {
    assert(variable >= values.size() && "Not yet implemented");
  }

  void dump() const { print(llvm::errs()); }
private:
  int64_t denom;
  unsigned variable;
};
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_CONSTRAINT_H
