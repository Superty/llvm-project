//===- PresburgerBasicSet.cpp - MLIR PresburgerBasicSet Class -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/ParamLexSimplex.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::analysis;
using namespace mlir::analysis::presburger;

void PresburgerBasicSet::addInequality(ArrayRef<int64_t> coeffs) {
  ineqs.emplace_back(coeffs);
}

void PresburgerBasicSet::removeLastInequality() {
  ineqs.pop_back();
}

void PresburgerBasicSet::removeLastEquality() {
  eqs.pop_back();
}

const InequalityConstraint &PresburgerBasicSet::getInequality(unsigned i) const {
  return ineqs[i];
}
const EqualityConstraint &PresburgerBasicSet::getEquality(unsigned i) const {
  return eqs[i];
}
ArrayRef<InequalityConstraint> PresburgerBasicSet::getInequalities() const {
  return ineqs;
}
ArrayRef<EqualityConstraint> PresburgerBasicSet::getEqualities() const {
  return eqs;
}

void PresburgerBasicSet::removeLastDivision() {
  divs.pop_back();
  for (auto &ineq : ineqs)
    ineq.removeLastDimension();
  for (auto &eq : eqs)
    eq.removeLastDimension();
  for (auto &div : divs)
    div.removeLastDimension();
}

void PresburgerBasicSet::addEquality(ArrayRef<int64_t> coeffs) {
  eqs.emplace_back(coeffs);
}


void PresburgerBasicSet::removeInequality(unsigned i) {
  ineqs.erase(ineqs.begin() + i, ineqs.begin() + i + 1);
}

void PresburgerBasicSet::removeEquality(unsigned i) {
  eqs.erase(eqs.begin() + i, eqs.begin() + i + 1);
}
void PresburgerBasicSet::appendDivisionVariable(ArrayRef<int64_t> coeffs, int64_t denom) {
  divs.emplace_back(coeffs, denom, /*variable = */getNumTotalDims());
  for (auto &ineq : ineqs)
    ineq.appendDimension();
  for (auto &eq : eqs)
    eq.appendDimension();
  for (auto &div : divs)
    div.appendDimension();
}
void PresburgerBasicSet::dump() const {
  // auto printName = [&](unsigned idx) {
  //   assert(idx < nDim && "Out of bounds index!");
  //   if (idx > nDim - nDiv) llvm::errs() << "d" << idx - (nDim - nDiv);
  //   else if (idx > nDim - nDiv - nParam) llvm::errs() << "p" << idx - (nDim - nDiv - nParam);
  //   else llvm::errs() << "x" << idx;
  // };

  // auto printExpr = [&](ArrayRef<int64_t> expr) {
  //   for (unsigned idx = 0; idx < expr.size() - 1; ++idx) {
  //     if (expr[idx] == 0) continue;
  //     if (expr[idx] == -1) llvm::errs() << "-";
  //     else if (expr[idx] != 1) llvm::errs() << expr[idx];
  //     printName(idx);
  //     llvm::errs() << " + ";
  //   }
  //   llvm::errs() << expr.back();
  // };

  // llvm::errs() << "nDim = " << nDim << ", nParam = " << nParam << ", nDiv = " << nDiv << '\n';
  // for (unsigned i = 0; i < nDiv; ++i) {
  //   llvm::errs() << "d" << i << " = [(";
  //   printExpr(divs[i].num);
  //   llvm::errs() << ")/" << divs[i].den << "], ";
  // }
  // llvm::errs() << '\n';
  // for (auto &ineq : ineqs) {
  //   printExpr(ineq);
  //   llvm::errs() << " >= 0 and ";
  // }
  // for (auto &eq : eqs) {
  //   printExpr(eq);
  //   llvm::errs() << " == 0 and ";
  // }
}
