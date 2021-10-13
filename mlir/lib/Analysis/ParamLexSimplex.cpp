//===- ParamLexSimplex.cpp - MLIR ParamLexSimplex Class -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/ParamLexSimplex.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Support/MathExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

/// Construct a Simplex object with `nVar` variables.
ParamLexSimplex::ParamLexSimplex(unsigned nVar, unsigned oNParam)
    : Simplex(nVar + 1), nParam(oNParam), nDiv(0) {
  for (unsigned i = 1; i <= nParam; ++i) {
    colUnknown[nCol - nParam + i - 1] = i;
    var[i].pos = nCol - nParam + i - 1;
  }
  for (unsigned i = nParam + 1; i < var.size(); ++i) {
    colUnknown[3 + i - (nParam + 1)] = i;
    var[i].pos = 3 + i - (nParam + 1);
  }
}

ParamLexSimplex::ParamLexSimplex(const FlatAffineConstraints &constraints)
    : ParamLexSimplex(constraints.getNumIds(), constraints.getNumIds()) {
  // TODO get symbol count from the FAC!
  llvm_unreachable("not yet implemented!");
  // for (unsigned i = 0, numIneqs = constraints.getNumInequalities();
  //      i < numIneqs; ++i)
  //   addInequality(constraints.getInequality(i));
  // for (unsigned i = 0, numEqs = constraints.getNumEqualities(); i < numEqs;
  // ++i)
  //   addEquality(constraints.getEquality(i));
}

/// Add an inequality to the tableau. If coeffs is c_0, c_1, ... c_n, where n
/// is the curent number of variables, then the corresponding inequality is
/// c_n + c_0*x_0 + c_1*x_1 + ... + c_{n-1}*x_{n-1} >= 0.
///
/// We add the inequality and mark it as restricted. We then try to make its
/// sample value non-negative. If this is not possible, the tableau has become
/// empty and we mark it as such.
///
/// Our internal nonparameters are M + x1, M + x2...
/// Such that the initial state M + x1 = 0 => x1 = -M, so x1 starts at min. val.
/// Therefore ax1 + bx2 = a(M + x1) + b(M + x2) - (a + b)M.
void ParamLexSimplex::addInequality(ArrayRef<int64_t> coeffs) {
  llvm::SmallVector<int64_t, 8> newCoeffs;
  newCoeffs.push_back(0);
  newCoeffs.insert(newCoeffs.end(), coeffs.begin(), coeffs.end());
  for (unsigned i = nParam - nDiv; i < coeffs.size() - 1 - nDiv;
       ++i) // -1 for constant at the end; - nDiv because divisions don't have M
    newCoeffs[0] -= coeffs[i];

  assert(newCoeffs.size() == var.size() + 1);
  unsigned conIndex = addRow(newCoeffs);
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
  assert(coeffs.size() == var.size() + 1 - 1); // - 1 for M
  addInequality(coeffs);
  con.back().zero = true;
  SmallVector<int64_t, 8> negatedCoeffs;
  for (int64_t coeff : coeffs)
    negatedCoeffs.emplace_back(-coeff);
  addInequality(negatedCoeffs);
  con.back().zero = true;
}

/// Add a division variable to the tableau. If coeffs is c_0, c_1, ... c_n,
/// where n is the curent number of variables, then the corresponding variable
/// is q = floor(c_n + c_0*x_0 + c_1*x_1 + ... + c_{n-1}*x_{n-1})/denom.
///
/// This is implemented by adding a new variable, q, and adding the two
/// inequalities:
///
/// 0 <= c_n + c_0*x_0 + c_1*x_1 + ... + c_{n-1}*x_{n-1} - denom*q <= denom - 1.
///
/// We simply add two opposing inequalities, which force the expression to
/// be zero.
void ParamLexSimplex::addDivisionVariable(ArrayRef<int64_t> coeffs,
                                          int64_t denom) {
  // assert(coeffs.size() == var.size() + 1 - 1); // - 1 for M
  // llvm::errs() << "adding div: ";
  // for (auto x : coeffs) {
  //   llvm::errs() << x << ' ';
  // }
  // llvm::errs() << '\n';
  appendVariable();
  nParam++;
  nDiv++;

  // SmallVector<int64_t, 8> ineq(coeffs.begin(), coeffs.end());
  // int64_t constTerm = ineq.back();
  // ineq.back() = -denom;
  // ineq.push_back(constTerm);
  // addInequality(ineq);

  // for (int64_t &coeff : ineq)
  //   coeff = -coeff;
  // ineq.back() += denom - 1;
  // addInequality(ineq);
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
//
// We don't need to care about the big param. Suppose p, q are the coeffs for
// the big param.
// A pivot causes the following change:
//            pivot col   bigparam col    const col                   pivot col
//            bigparam col    const col
// pivot row     a            p               b       ->   pivot row     1/a
// -p/a          -b/a other row     c            q               d other row c/a
// q - pc/a       d - bc/a
//
// if p is zero, no issues. otherwise, it has to be negative and behaves just
// like b. taking (-p) as a common factor, the bigparam changes would be
// less/greater/equal exactly when the const col changes are.
Optional<unsigned> ParamLexSimplex::findPivot(unsigned row) const {
  // assert(tableau(row, 1) < 0 && "Pivot row must be violated!");

  auto getLexChange = [this, row](unsigned col) -> SmallVector<Fraction, 8> {
    SmallVector<Fraction, 8> change;
    auto a = tableau(row, col);
    for (unsigned i = 1; i < var.size(); ++i) {
      if (var[i].orientation == Orientation::Column) {
        if (var[i].pos == col)
          change.emplace_back(1, a);
        else
          change.emplace_back(0, 1);
      } else {
        assert(var[i].pos != row && "pivot row should be a violated constraint "
                                    "and so cannot be a variable");
        change.emplace_back(tableau(var[i].pos, col), a);
      }
    }
    return change;
  };

  Optional<unsigned> maybeColumn;
  SmallVector<Fraction, 8> change;
  for (unsigned col = 3; col < nCol - nParam; ++col) {
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
  assert(tableau(row, 2) <=
         0); // if bigparam is positive, moving to col is lexneg change.
  Optional<unsigned> maybeColumn = findPivot(row);
  if (!maybeColumn)
    return failure();
  pivot(row, *maybeColumn);
  return success();
}

void ParamLexSimplex::restoreConsistency() {
  auto maybeGetViolatedRow = [this]() -> Optional<unsigned> {
    for (unsigned row = 0; row < nRow; ++row) {
      if (tableau(row, 2) < 0)
        return row;
      if (tableau(row, 2) == 0 && tableau(row, 1) < 0)
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

PWAFunction ParamLexSimplex::findParamLexmin() {
  PWAFunction result;
  FlatAffineConstraints domainSet(nParam, 0, 0);
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

// // SmallVector<int64_t, 8>
// ParamLexSimplex::varCoeffsFromRowCoeffs(ArrayRef<int64_t> rowCoeffs)
// const {
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

unsigned ParamLexSimplex::getSnapshot() { return getSnapshotBasis(); }

void ParamLexSimplex::findParamLexminRecursively(Simplex &domainSimplex,
                                                 FlatAffineConstraints &domainSet,
                                                 PWAFunction &result) {
  // dump();
  // domainSet.dump();
  // llvm::errs() << "nParam = " << nParam << '\n';
  if (empty || domainSimplex.isEmpty())
    return;

  for (unsigned row = 0; row < nRow; ++row) {
    if (!unknownFromRow(row).restricted)
      continue;

    if (tableau(row, 2) > 0) // nonNegative
      continue;
    if (tableau(row, 2) < 0) { // negative
      auto status = moveRowUnknownToColumn(row);
      if (failed(status))
        return;
      findParamLexminRecursively(domainSimplex, domainSet, result);
      return;
    }

    auto paramSample = getRowParamSample(row);
    auto maybeMin = domainSimplex.computeOptimum(Simplex::Direction::Down, paramSample);
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
    auto idx = rowUnknown[row];

    findParamLexminRecursively(domainSimplex, domainSet, result);

    domainSet.removeInequality(domainSet.getNumInequalities() - 1);
    domainSimplex.rollback(domainSnapshot);
    rollback(snapshot);

    SmallVector<int64_t, 8> complementIneq;
    for (int64_t coeff : paramSample)
      complementIneq.push_back(-coeff);
    complementIneq.back() -= 1;

    snapshot = getSnapshot();
    domainSnapshot = domainSimplex.getSnapshot();
    domainSimplex.addInequality(complementIneq);
    domainSet.addInequality(complementIneq);

    auto &u = unknownFromIndex(idx);
    assert(u.orientation == Orientation::Row);
    if (succeeded(moveRowUnknownToColumn(u.pos)))
      findParamLexminRecursively(domainSimplex, domainSet, result);

    domainSet.removeInequality(domainSet.getNumInequalities() - 1);
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

  for (unsigned varI = 1 + nParam - nDiv; varI < var.size() - nDiv; ++varI) {
    const auto &u = var[varI];
    if (u.orientation == Orientation::Column)
      continue;

    unsigned row = u.pos;
    if (rowHasIntegerCoeffs(row))
      continue;

    int64_t denom = tableau(row, 0);
    bool paramCoeffsIntegral = true;
    for (unsigned col = nCol - nParam; col < nCol; col++) {
      if (mod(tableau(row, col), denom) != 0) {
        paramCoeffsIntegral = false;
        break;
      }
    }

    bool otherCoeffsIntegral = true;
    for (unsigned col = 3; col < nCol - nParam; ++col) {
      if (unknownFromColumn(col).zero)
        continue;
      if (mod(tableau(row, col), denom) != 0) {
        otherCoeffsIntegral = false;
        break;
      }
    }

    // bool constIntegral = mod(tableau(row, 1), denom) == 0;

    SmallVector<int64_t, 8> domainDivCoeffs;
    if (otherCoeffsIntegral) {
      for (unsigned col = nCol - nParam; col < nCol; ++col)
        domainDivCoeffs.push_back(mod(tableau(row, col), denom));
      domainDivCoeffs.push_back(mod(tableau(row, 1), denom));
      unsigned snapshot = getSnapshot();
      unsigned domainSnapshot = domainSimplex.getSnapshot();
      domainSimplex.addDivisionVariable(domainDivCoeffs, denom);
      domainSet.addLocalFloorDiv(domainDivCoeffs, denom);

      SmallVector<int64_t, 8> ineqCoeffs;
      for (auto x : domainDivCoeffs)
        ineqCoeffs.push_back(-x);
      ineqCoeffs.back() = denom;
      ineqCoeffs.push_back(-domainDivCoeffs.back());
      domainSimplex.addInequality(ineqCoeffs);
      domainSet.addInequality(ineqCoeffs);

      // This has to be after we extract the coeffs above!
      appendVariable();
      nParam++;
      nDiv++;

      SmallVector<int64_t, 8> oldRow;
      oldRow.reserve(nCol);
      for (unsigned col = 0; col < nCol; ++col) {
        oldRow.push_back(tableau(row, col));
        tableau(row, col) /= denom;
      }
      tableau(row, nCol - 1) += 1;

      findParamLexminRecursively(domainSimplex, domainSet, result);

      domainSet.removeInequalityRange(domainSet.getNumInequalities() - 3, domainSet.getNumInequalities());
      domainSet.removeId(FlatAffineConstraints::IdKind::Local, domainSet.getNumLocalIds() - 1);
      for (unsigned col = 0; col < nCol; ++col)
        tableau(row, col) = oldRow[col];
      domainSimplex.rollback(domainSnapshot);
      nParam--;
      nDiv--;
      rollback(snapshot);
      return;
    }
    // llvm::errs() << "const: " << constIntegral << ", param: " << paramCoeffsIntegral <<  '\n';

    for (unsigned col = nCol - nParam; col < nCol; ++col)
      domainDivCoeffs.push_back(mod(int64_t(-tableau(row, col)), denom));
    domainDivCoeffs.push_back(mod(int64_t(-tableau(row, 1)), denom));

    unsigned snapshot = getSnapshot();
    unsigned domainSnapshot = domainSimplex.getSnapshot();
    domainSimplex.addDivisionVariable(domainDivCoeffs, denom);
    domainSet.addLocalFloorDiv(domainDivCoeffs, denom);

    // SmallVector<int64_t, 8> divCoeffs = domainDivCoeffs;
    // divCoeffs.insert(divCoeffs.end(),
    //   domainDivCoeffs.begin(), domainDivCoeffs.end());
    domainDivCoeffs.insert(domainDivCoeffs.begin() + nParam - nDiv,
                           var.size() - nParam - 1, 0); // -1 for M
    addDivisionVariable(domainDivCoeffs, denom);

    addZeroConstraint();
    con.back().restricted = true;
    tableau(nRow - 1, 0) = denom;
    tableau(nRow - 1, 1) = -mod(int64_t(-tableau(row, 1)), denom);
    tableau(nRow - 1, 2) = 0;
    for (unsigned col = 3; col < nCol - nParam; ++col)
      tableau(nRow - 1, col) = mod(tableau(row, col), denom);
    for (unsigned col = nCol - nParam; col < nCol - 1; ++col)
      tableau(nRow - 1, col) = -mod(int64_t(-tableau(row, col)), denom);
    tableau(nRow - 1, nCol - 1) = denom;
    LogicalResult success = moveRowUnknownToColumn(nRow - 1);
    assert(succeeded(success));
    // restoreConsistency();

    findParamLexminRecursively(domainSimplex, domainSet, result);

    domainSimplex.rollback(domainSnapshot);
    domainSet.removeInequalityRange(domainSet.getNumInequalities() - 2, domainSet.getNumInequalities());
    domainSet.removeId(FlatAffineConstraints::IdKind::Local, domainSet.getNumLocalIds() - 1);
    nParam--;
    nDiv--;
    rollback(snapshot);

    return;
  }

  result.domain.push_back(domainSet);
  SmallVector<SmallVector<int64_t, 8>, 8> lexmin;
  // domainSet.dump();
  // dump();
  for (unsigned i = 1 + nParam - nDiv; i < var.size() - nDiv;
       ++i) { // 1 + for bigM
    if (var[i].orientation == Orientation::Column) {
      lexmin.push_back(SmallVector<int64_t, 8>(nParam + 1, 0));
      continue;
    }

    unsigned row = var[i].pos;
    // assert(tableau(row, 2) <= 1); // M + x = kM + ...; x = (k-1)M + ...; k-1<=0
    if (tableau(row, 2) <= 0) {
      // lexmin is unbounded; we push an empty entry for this lexmin.
      lexmin.clear();
      break;
    }

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
  // result.dump();
}

