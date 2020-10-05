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

/// Construct a Simplex object with `nVar` variables.
ParamLexSimplex::ParamLexSimplex(unsigned nVar, unsigned oNParam)
    : Simplex(nVar), nParam(oNParam), nDiv(0) {
  for (unsigned i = 0; i < nParam; ++i) {
    colUnknown[nCol - nParam + i] = i;
    var[i].pos = nCol - nParam + i;
  }
  for (unsigned i = nParam; i < var.size(); ++i) {
    colUnknown[i - nParam + 2] = i;
    var[i].pos = i - nParam + 2; 
  }
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
  originalCoeffs.emplace_back(llvm::to_vector<8>(coeffs));
  unsigned conIndex = addRow(coeffs);
  Unknown &u = con[conIndex];
  u.restricted = true;
  // restoreConsistency();
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
  nDiv++;
  
  SmallVector<int64_t, 8> ineq(coeffs.begin(), coeffs.end());
  int64_t constTerm = ineq.back();
  ineq.back() = -denom;
  ineq.push_back(constTerm);
  addInequality(ineq);

  for (int64_t &coeff : ineq)
    coeff = -coeff;
  ineq.back() += denom - 1;
  addInequality(ineq);
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
  // assert(tableau(row, 1) < 0 && "Pivot row must be violated!");

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

LogicalResult ParamLexSimplex::moveRowUnknownToColumn(unsigned row) {
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
    LogicalResult status = moveRowUnknownToColumn(*maybeViolatedRow);
    if (failed(status)) {
      markEmpty();
      return;
    }
  }
}

pwaFunction ParamLexSimplex::findParamLexmin() {
  pwaFunction result;
  PresburgerBasicSet domainSet(nParam, 0, 0);
  Simplex domainSimplex(nParam);
  findParamLexminRecursively(domainSimplex, domainSet, result);
  return result;
}

SmallVector<int64_t, 8> ParamLexSimplex::getRowParamSample(unsigned row) {
  SmallVector<int64_t, 8> sample;
  sample.reserve(nParam + 1);
  for (unsigned col = nCol - nParam; col < nCol; ++col)
    sample.push_back(tableau(row, col));
  sample.push_back(tableau(row, 1));
  return sample;
}

// SmallVector<int64_t, 8> ParamLexSimplex::varCoeffsFromRowCoeffs(ArrayRef<int64_t> rowCoeffs) const {
//   SmallVector<int64_t, 8> varCoeffs(var.size() + 1, 0);

//   // Copy the constant term.
//   varCoeffs.back() = rowCoeffs.back();
//   for (unsigned i = 0; i < rowCoeffs.size() - 1; ++i) {
//     if (colUnknown[i] >= 0) {
//       varCoeffs[colUnknown[i]] += rowCoeffs[i];
//       continue;
//     }

//     // Include the constant term of the row.
//     for (unsigned j = 0; j < rowCoeffs.size(); ++j)
//       varCoeffs[j] += rowCoeffs[i] * originalCoeffs[~colUnknown[i]][j];
//   }
//   return varCoeffs;
// }

unsigned ParamLexSimplex::getSnapshot() {
  return getSnapshotBasis();
}

void ParamLexSimplex::findParamLexminRecursively(Simplex &domainSimplex, PresburgerBasicSet &domainSet, pwaFunction &result) {
  // dump();
  // domainSet.dump();
  // llvm::errs() << "nParam = " << nParam << '\n';
  if (empty || domainSimplex.isEmpty())
    return;

  for (unsigned row = 0; row < nRow; ++row) {
    if (!unknownFromRow(row).restricted)
      continue;

    auto paramSample = getRowParamSample(row);
    auto maybeMin = domainSimplex.computeOptimum(Direction::Down, paramSample);
    bool nonNegative = maybeMin.hasValue() && *maybeMin >= Fraction(0, 1);
    if (nonNegative)
      continue;

    auto maybeMax = domainSimplex.computeOptimum(Direction::Up, paramSample);
    bool negative = maybeMax.hasValue() && *maybeMax < Fraction(0, 1);

    if (negative) {
      auto status = moveRowUnknownToColumn(row);
      if (failed(status))
        return;
      findParamLexminRecursively(domainSimplex, domainSet, result);
      return;
    }

    unsigned snapshot = getSnapshot();
    unsigned domainSnapshot = domainSimplex.getSnapshot();
    domainSimplex.addInequality(paramSample);
    domainSet.addInequality(paramSample);

    findParamLexminRecursively(domainSimplex, domainSet, result);

    domainSet.removeLastInequality();
    domainSimplex.rollback(domainSnapshot);
    rollback(snapshot);

    SmallVector<int64_t, 8> complementIneq;
    for (int64_t coeff : paramSample)
      complementIneq.push_back(-coeff);
    complementIneq.back()--;

    snapshot = getSnapshot();
    domainSnapshot = domainSimplex.getSnapshot();
    domainSimplex.addInequality(complementIneq);
    domainSet.addInequality(complementIneq);

    findParamLexminRecursively(domainSimplex, domainSet, result);

    domainSet.removeLastInequality();
    domainSimplex.rollback(domainSnapshot);
    rollback(snapshot);

    return;
  }

  auto rowHasIntegerCoeffs = [this](unsigned row) {
    int64_t denom = tableau(row, 0);
    if (tableau(row, 1) % denom != 0)
      return false;
    for (unsigned col = nCol - nParam; col < nCol; col++) {
      if (tableau(row, col) % denom != 0)
        return false;
    }
    return true;
  };

  for (const auto &u : var) {
    if (u.orientation == Orientation::Column)
      continue;

    unsigned row = u.pos;
    if (rowHasIntegerCoeffs(row))
      continue;

    llvm_unreachable("Not yet implemented");

    SmallVector<int64_t, 8> domainDivCoeffs;
    int64_t denom = tableau(row, 0);
    for (unsigned col = nCol - nParam; col < nCol; ++col)
      domainDivCoeffs.push_back(mod(-tableau(row, col), denom));
    domainDivCoeffs.push_back(mod(-tableau(row, 1), denom));

    unsigned snapshot = getSnapshot();
    unsigned domainSnapshot = domainSimplex.getSnapshot();
    domainSimplex.addDivisionVariable(domainDivCoeffs, denom);
    domainSet.appendDivisionVariable(domainDivCoeffs, denom);

    // SmallVector<int64_t, 8> divCoeffs = domainDivCoeffs;
    // divCoeffs.insert(divCoeffs.end(),
    //   domainDivCoeffs.begin(), domainDivCoeffs.end());
    domainDivCoeffs.insert(domainDivCoeffs.begin() + nParam - nDiv, var.size() - nParam, 0);
    addDivisionVariable(domainDivCoeffs, denom);

    addZeroConstraint();
    con.back().restricted = true;
    tableau(nRow - 1, 0) = denom;
    tableau(nRow - 1, 1) = -mod(-tableau(row, 1), denom);
    for (unsigned col = 2; col < nCol - nParam; ++col)
      tableau(nRow - 1, col) = mod(tableau(row, col), denom);
    for (unsigned col = nCol - nParam; col < nCol - 1; ++col)
      tableau(nRow - 1, col) = -mod(-tableau(row, col), denom);
    tableau(nRow - 1, nCol - 1) = denom;
    // restoreConsistency();

    findParamLexminRecursively(domainSimplex, domainSet, result);

    domainSimplex.rollback(domainSnapshot);
    domainSet.removeLastDivision();
    nParam--;
    nDiv--;
    rollback(snapshot);

    return;
  }

  result.domain.push_back(domainSet);
  SmallVector<SmallVector<int64_t, 8>, 8> lexmin;
  for (unsigned i = nParam; i < var.size(); ++i) {
    if (var[i].orientation == Orientation::Column) {
      lexmin.push_back(SmallVector<int64_t, 8>(nParam + 1, 0));
      continue;
    }

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
