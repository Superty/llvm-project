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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {

class Constraint {
public:
  Constraint() = delete;

  void appendNewDimension() {
    coeffs.push_back(coeffs.back());
    coeffs[coeffs.size() - 2] = 0;
  }

  void removeLastDimension() {
    coeffs[coeffs.size() - 2] = coeffs.back();
    coeffs.pop_back();
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
        os << coeffs[i];
        first = false;
      } else if (coeffs[i] > 0) {
        os << " + " << coeffs[i];
      } else {
        os << " - " << -coeffs[i];
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
  void dump() const { print(llvm::errs()); }
private:
  int64_t denom;
  unsigned variable;
};
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_CONSTRAINT_H
