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

namespace mlir {

void PresburgerBasicSet::appendDivisionVariable(ArrayRef<int64_t> coeffs, int64_t denom) {
  divs.push_back(DivisionVariable{SmallVector<int64_t, 8>(coeffs.begin(), coeffs.end()), denom});

  for (auto &ineq : ineqs)
    ineq.push_back(0);
  for (auto &eq : eqs)
    eq.push_back(0);
  for (auto &div : divs)
    div.num.push_back(0);
  nDim++;
  nDiv++;
}

void PresburgerBasicSet::addInequality(ArrayRef<int64_t> coeffs) {
  ineqs.push_back(SmallVector<int64_t, 8>(coeffs.begin(), coeffs.end()));
}

void PresburgerBasicSet::removeLastInequality() {
  ineqs.pop_back();
}

void PresburgerBasicSet::removeLastDivision() {
  divs.pop_back();
  nDim--;
  nDiv--;

  for (auto &ineq : ineqs)
    ineq.pop_back();
  for (auto &eq : eqs)
    eq.pop_back();
  for (auto &div : divs)
    div.num.pop_back();
}

void PresburgerBasicSet::addEquality(ArrayRef<int64_t> coeffs) {
  eqs.push_back(SmallVector<int64_t, 8>(coeffs.begin(), coeffs.end()));
}

} // namespace mlir