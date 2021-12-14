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

ParamLexSimplex::ParamLexSimplex(unsigned nVar, unsigned paramBegin, unsigned oNParam)
    : LexSimplex(nVar) {
  nParam = oNParam;
  for (unsigned i = 0; i < nParam; ++i) {
    var[paramBegin + i].isParam = true;
    swapColumns(var[paramBegin + i].pos, 3 + i);
  }
}

ParamLexSimplex::ParamLexSimplex(const FlatAffineConstraints &constraints)
    : ParamLexSimplex(constraints.getNumIds(), constraints.getIdKindOffset(FlatAffineConstraints::IdKind::Symbol), constraints.getNumSymbolIds()) {
  for (unsigned i = 0, numIneqs = constraints.getNumInequalities(); i < numIneqs; ++i)
    addInequality(constraints.getInequality(i));
  for (unsigned i = 0, numEqs = constraints.getNumEqualities(); i < numEqs; ++i)
    addEquality(constraints.getEquality(i));
}

void ParamLexSimplex::appendParameter() {
  appendVariable();
  swapColumns(3 + nParam, nCol - 1);
  var.back().isParam = true;
  nParam++;
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
  for (unsigned col = 3; col < 3 + nParam; ++col)
    sample.push_back(tableau(row, col));
  sample.push_back(tableau(row, 1));
  return sample;
}

void ParamLexSimplex::findParamLexminRecursively(Simplex &domainSimplex,
                                                 FlatAffineConstraints &domainSet,
                                                 PWAFunction &result) {
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
    for (unsigned col = 3; col < 3 + nParam; col++) {
      if (tableau(row, col) % denom != 0)
        return false;
    }
    return true;
  };

  for (const Unknown &u : var) {
    if (u.orientation == Orientation::Column)
      continue;

    unsigned row = u.pos;
    if (rowHasIntegerCoeffs(row))
      continue;

    int64_t denom = tableau(row, 0);
    bool paramCoeffsIntegral = true;
    for (unsigned col = 3; col < 3 + nParam; col++) {
      if (mod(tableau(row, col), denom) != 0) {
        paramCoeffsIntegral = false;
        break;
      }
    }

    bool otherCoeffsIntegral = true;
    for (unsigned col = 3 + nParam; col < nCol; ++col) {
      if (mod(tableau(row, col), denom) != 0) {
        otherCoeffsIntegral = false;
        break;
      }
    }

    // bool constIntegral = mod(tableau(row, 1), denom) == 0;

    SmallVector<int64_t, 8> domainDivCoeffs;
    if (otherCoeffsIntegral) {
      for (unsigned col = 3; col < 3 + nParam; ++col)
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
      appendParameter();

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
      rollback(snapshot);
      return;
    }

    SmallVector<int64_t, 8> divCoeffs;
    for (unsigned col = 3; col < 3 + nParam; ++col)
      domainDivCoeffs.push_back(mod(int64_t(-tableau(row, col)), denom));
    domainDivCoeffs.push_back(mod(int64_t(-tableau(row, 1)), denom));

    unsigned snapshot = getSnapshot();
    unsigned domainSnapshot = domainSimplex.getSnapshot();
    domainSimplex.addDivisionVariable(domainDivCoeffs, denom);
    domainSet.addLocalFloorDiv(domainDivCoeffs, denom);

    appendParameter();

    addZeroConstraint();
    con.back().restricted = true;
    tableau(nRow - 1, 0) = denom;
    tableau(nRow - 1, 1) = -mod(int64_t(-tableau(row, 1)), denom);
    tableau(nRow - 1, 2) = 0;
    for (unsigned col = 3 + nParam; col < nCol; ++col)
      tableau(nRow - 1, col) = mod(tableau(row, col), denom);
    for (unsigned col = 3; col < 3 + nParam - 1; ++col)
      tableau(nRow - 1, col) = -mod(int64_t(-tableau(row, col)), denom);
    tableau(nRow - 1, 3 + nParam - 1) = denom;
    LogicalResult success = moveRowUnknownToColumn(nRow - 1);
    assert(succeeded(success));

    findParamLexminRecursively(domainSimplex, domainSet, result);

    domainSimplex.rollback(domainSnapshot);
    domainSet.removeInequalityRange(domainSet.getNumInequalities() - 2, domainSet.getNumInequalities());
    domainSet.removeId(FlatAffineConstraints::IdKind::Local, domainSet.getNumLocalIds() - 1);
    rollback(snapshot);

    return;
  }

  result.domain.push_back(domainSet);
  SmallVector<SmallVector<int64_t, 8>, 8> lexmin;
  for (unsigned i = 0; i < var.size(); ++i) {
    if (var[i].isParam)
      continue;
    if (var[i].orientation == Orientation::Column) {
      lexmin.push_back(SmallVector<int64_t, 8>(nParam + 1, 0));
      continue;
    }

    unsigned row = var[i].pos;
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
}

