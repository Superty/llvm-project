//===- ParamLexSimplex.cpp - MLIR ParamLexSimplex Class -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/ParamLexSimplex.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Support/MathExtras.h"

namespace mlir {
using Direction = Simplex::Direction;

const int nullIndex = std::numeric_limits<int>::max();

/// Construct a Simplex object with `nVar` variables.
ParamLexSimplex::ParamLexSimplex(unsigned nVar, unsigned oNParam)
    : Simplex(nVar), nParam(oNParam) {
}

ParamLexSimplex::ParamLexSimplex(const FlatAffineConstraints &constraints)
    : ParamLexSimplex(constraints.getNumIds(), constraints.getNumIds()) {
  // TODO get symbol count from the FAC!
  for (unsigned i = 0, numIneqs = constraints.getNumInequalities();
       i < numIneqs; ++i)
    addInequality(constraints.getInequality(i));
  for (unsigned i = 0, numEqs = constraints.getNumEqualities(); i < numEqs; ++i)
    addEquality(constraints.getEquality(i));
}

/// Add an inequality to the tableau. If coeffs is c_0, c_1, ... c_n, where n
/// is the curent number of variables, then the corresponding inequality is
/// c_n + c_0*x_0 + c_1*x_1 + ... + c_{n-1}*x_{n-1} >= 0.
///
/// We add the inequality and mark it as restricted. We then try to make its
/// sample value non-negative. If this is not possible, the tableau has become
/// empty and we mark it as such.
void ParamLexSimplex::addInequality(ArrayRef<int64_t> coeffs) {
  unsigned conIndex = addRow(coeffs);
  Unknown &u = con[conIndex];
  u.restricted = true;
  restoreConsistency();
}

/// Add an equality to the tableau. If coeffs is c_0, c_1, ... c_n, where n
/// is the curent number of variables, then the corresponding equality is
/// c_n + c_0*x_0 + c_1*x_1 + ... + c_{n-1}*x_{n-1} == 0.
///
/// We simply add two opposing inequalities, which force the expression to
/// be zero.
void ParamLexSimplex::addEquality(ArrayRef<int64_t> coeffs) {
  addInequality(coeffs);
  SmallVector<int64_t, 8> negatedCoeffs;
  for (int64_t coeff : coeffs)
    negatedCoeffs.emplace_back(-coeff);
  addInequality(negatedCoeffs);
}

/// Add a division variable to the tableau. If coeffs is c_0, c_1, ... c_n, where n
/// is the curent number of variables, then the corresponding variable is
/// q = floor(c_n + c_0*x_0 + c_1*x_1 + ... + c_{n-1}*x_{n-1})/denom.
///
/// This is implemented by adding a new variable, q, and adding the two
/// inequalities:
///
/// 0 <= c_n + c_0*x_0 + c_1*x_1 + ... + c_{n-1}*x_{n-1} - denom*q <= denom - 1.
///
/// We simply add two opposing inequalities, which force the expression to
/// be zero.
void ParamLexSimplex::addDivisionVariable(ArrayRef<int64_t> coeffs, int64_t denom) {
  addVariable();
  nParam++;
  
  SmallVector<int64_t, 8> ineq1(coeffs.begin(), coeffs.end());
  ineq1.push_back(-denom);
  addInequality(ineq1);

  SmallVector<int64_t, 8> ineq2;
  for (int64_t coeff : coeffs)
    ineq2.push_back(-coeff);
  ineq2.push_back(denom - 1);
  addInequality(ineq2);
}

// Find the pivot column for the given pivot row which would result in the
// lexicographically smallest positive change in the sample value. The pivot
// row must be violated.
//
// A pivot causes the following change:
//            pivot col    other col                   pivot col    other col
// pivot row     a             b       ->   pivot row     1/a         -b/a    
// other row     c             d            other row     c/a        d - bc/a
//
// We just compute the change for every column and pick the lexicographically
// smallest positive one, ignoring columns which the row has 0 coefficient for
// (since the pivot is then impossible) and columns which correspond to
// parameters (since we always keep the parameters in the basis).
//
// A change is lexicographically positive if it is not all zeros, and the
// first non-zero value is positive.
Optional<unsigned> ParamLexSimplex::findPivot(unsigned row) const {
  assert(tableau(row, 1) < 0 && "Pivot row must be violated!");

  auto getLexChange = [this, row](unsigned col)
                -> SmallVector<Fraction, 8> {
    SmallVector<Fraction, 8> change;
    auto a = tableau(row, col);
    for (unsigned i = 0; i < var.size(); ++i) {
      if (var[i].orientation == Orientation::Column) {
        if (var[i].pos == col)
          change.emplace_back(1, a);
        else
          change.emplace_back(0, 1);
      } else {
        assert(var[i].pos != row &&
               "pivot row should be a violated constraint and so cannot be a variable");
        change.emplace_back(tableau(var[i].pos, col), a);
      }
    }
    return change;
  };

  Optional<unsigned> maybeColumn;
  SmallVector<Fraction, 8> change;
  for (unsigned col = 2; col < nCol - nParam; ++col) {
    if (tableau(row, col) <= 0)
      continue;
    // // Never remove parameters from the basis.
    // if (col_var[col] >= 0 && col_var[col] < int(n_param))
    //   continue;

    auto newChange = getLexChange(col);
    if (!maybeColumn || newChange < change) {
      maybeColumn = col;
      change = std::move(newChange);
    }
  }
  return maybeColumn;
}

LogicalResult ParamLexSimplex::restoreRow(unsigned row) {
  Optional<unsigned> maybeColumn = findPivot(row);
  if (!maybeColumn)
    return LogicalResult::Failure;
  pivot(row, *maybeColumn);
  return LogicalResult::Success;
}

void ParamLexSimplex::restoreConsistency() {
  auto maybeGetViolatedRow = [this]() -> Optional<unsigned> {
    for (unsigned row = 0; row < nRow; ++row) {
      if (tableau(row, 1) < 0)
        return row;
    }
    return {};
  };

  while (Optional<unsigned> maybeViolatedRow = maybeGetViolatedRow()) {
    LogicalResult status = restoreRow(*maybeViolatedRow);
    if (failed(status)) {
      markEmpty();
      return;
    }
  }
}

pwaFunction ParamLexSimplex::findParamLexmin() {
  pwaFunction result;
  PresburgerBasicSet domainSet(var.size(), nParam, 0);
  ParamLexSimplex domainSimplex(var.size(), nParam);
  findParamLexminRecursively(domainSimplex, domainSet, result);
  return result;
}

ArrayRef<int64_t> ParamLexSimplex::getRowParamSample(unsigned row) {
  return ArrayRef<int64_t>{&tableau(row, nCol - nParam), &tableau(row, nCol)};
}

void ParamLexSimplex::findParamLexminRecursively(ParamLexSimplex &domainSimplex, PresburgerBasicSet &domainSet, pwaFunction &result) {
  if (empty || domainSimplex.isEmpty())
    return;

  for (unsigned row = 0; row < nRow; ++row) {
    auto paramSample = getRowParamSample(row);
    auto maybeMin = domainSimplex.computeOptimum(Direction::Down, paramSample);
    bool nonNegative = maybeMin.hasValue() && *maybeMin >= Fraction(0, 0);
    if (nonNegative)
      continue;

    auto maybeMax = domainSimplex.computeOptimum(Direction::Up, paramSample);
    bool nonPositive = maybeMax.hasValue() && *maybeMax <= Fraction(0, 0);

    if (nonPositive) {
      restoreRow(row);
      continue;
    }

    unsigned snapshot = getSnapshotBasis();
    domainSimplex.addInequality(paramSample);
    domainSet.addInequality(paramSample);
    findParamLexminRecursively(domainSimplex, domainSet, result);
    domainSet.removeLastInequality();
    domainSimplex.rollback(snapshot);

    SmallVector<int64_t, 8> complementIneq;
    for (int64_t coeff : paramSample)
      complementIneq.push_back(-coeff);
    complementIneq.back()--;

    snapshot = getSnapshotBasis();
    domainSimplex.addInequality(complementIneq);
    domainSet.addInequality(complementIneq);
    findParamLexminRecursively(domainSimplex, domainSet, result);
    domainSet.removeLastInequality();
    domainSimplex.rollback(snapshot);
  }

  auto rowHasIntegerCoeffs = [this](unsigned row) {
    for (unsigned col = 1; col < nCol; col++) {
      if (tableau(row, col) % tableau(row, 0) != 0)
        return false;
    }
    return true;
  };

  for (unsigned row = 0; row < nRow; ++row) {
    if (rowHasIntegerCoeffs(row))
      continue;

    SmallVector<int64_t, 8> divCoeffs;
    int64_t denom = tableau(row, 0);
    for (unsigned col = nCol - nParam; col < nCol; ++col) {
      divCoeffs.push_back(mod(-tableau(row, col), denom));
    }
    unsigned snapshot = getSnapshotBasis();
    domainSimplex.addDivisionVariable(divCoeffs, denom);
    domainSet.appendDivisionVariable(divCoeffs, denom);

    findParamLexminRecursively(domainSimplex, domainSet, result);

    domainSimplex.rollback(snapshot);
    domainSet.removeLastDivision();

    return;
  }

  result.domain.push_back(domainSet);
  SmallVector<SmallVector<int64_t, 8>, 8> lexmin;
  for (unsigned i = 0; i < var.size() - nParam; ++i) {
    assert(var[i].orientation == Orientation::Row && "lexmin is unbounded!");
    unsigned row = var[i].pos;
    auto coeffs = getRowParamSample(var[i].pos);
    int64_t denom = tableau(row, 0);
    SmallVector<int64_t, 8> value;
    for (const int64_t &coeff : coeffs) {
      assert(coeff % denom == 0 && "coefficient is fractional!");
      value.push_back(coeff / denom);
    }
    lexmin.push_back(std::move(value));
  }
  result.value.push_back(lexmin);
}


} // namespace mlir
