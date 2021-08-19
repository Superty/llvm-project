//===- PresburgerBasicMap.cpp - MLIR PresburgerMap Class -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_PRESBURGERMAP_IMPL_H
#define MLIR_ANALYSIS_PRESBURGER_PRESBURGERMAP_IMPL_H

#include "mlir/Analysis/Presburger/ParamLexSimplex.h"
#include "mlir/Analysis/Presburger/PresburgerMap.h"

using namespace mlir;
using namespace mlir::analysis;
using namespace mlir::analysis::presburger;

template <typename Int>
void PresburgerMap<Int>::dump() const {
  print(llvm::errs());
  llvm::errs() << "\n";
}

template <typename Int>
void PresburgerMap<Int>::print(raw_ostream &os) const {
  os << "Relation"
     << ": [";
  for (unsigned i = 0; i < domainDim; ++i) {
    os << "d" << i;
    if (i != domainDim - 1)
      os << ", ";
  }
  os << "]";

  os << " -> [";

  for (unsigned i = 0; i < rangeDim; ++i) {
    os << "d" << domainDim + i;
    if (i != rangeDim - 1)
      os << ", ";
  }
  os << "]\n";

  PresburgerSet<Int>::print(os);
}

template <typename Int>
PresburgerSet<Int> PresburgerMap<Int>::getRangeSet() const {
  PresburgerSet rangeSet = *this;
  for (PresburgerBasicSet<Int> &bs : rangeSet.basicSets)
    bs.convertDimsToExists(0, getDomainDims());
  return rangeSet;
}

template <typename Int>
PresburgerSet<Int> PresburgerMap<Int>::getDomainSet() const {
  PresburgerSet domainSet = *this;
  for (PresburgerBasicSet<Int> &bs : domainSet.basicSets)
    bs.convertDimsToExists(getDomainDims(), getDomainDims() + getRangeDims());
  return domainSet;
}

template <typename Int>
SmallVector<Int, 8>
PresburgerMap<Int>::convertToRequiredForm(ArrayRef<Int> coeffs) {
  SmallVector<Int, 8> newCoeffs;

  // Add domain elements
  for (unsigned i = 0; i < getDomainDims(); ++i)
    newCoeffs.push_back(coeffs[i]);

  // Add remaining variables
  for (unsigned i = this->getNumDims(); i < coeffs.size() - 1; ++i)
    newCoeffs.push_back(coeffs[i]);

  // Add range elements
  for (unsigned i = getDomainDims(); i < this->getNumDims(); ++i)
    newCoeffs.push_back(coeffs[i]);

  // Add constant
  newCoeffs.push_back(coeffs.back());

  return newCoeffs;
}

template <typename Int>
void PresburgerMap<Int>::lexMinRange() {
  PresburgerMap lexMinMap(getDomainDims(), getRangeDims(), this->getNumSyms());
  for (const auto &bs : this->getBasicSets()) {
    ParamLexSimplex<Int> paramLexSimplex(bs.getNumTotalDims(),
                                         bs.getNumTotalDims() - getRangeDims());
    for (const auto &div : bs.getDivisions()) {
      // The division variables must be in the same order they are stored in the
      // basic set.
      paramLexSimplex.addInequality(
          convertToRequiredForm(div.getInequalityLowerBound().getCoeffs()));
      paramLexSimplex.addInequality(
          convertToRequiredForm(div.getInequalityUpperBound().getCoeffs()));
    }
    for (const auto &ineq : bs.getInequalities())
      paramLexSimplex.addInequality(convertToRequiredForm(ineq.getCoeffs()));
    for (const auto &eq : bs.getEqualities())
      paramLexSimplex.addEquality(convertToRequiredForm(eq.getCoeffs()));

    pwaFunction<Int> pwa = paramLexSimplex.findParamLexmin();

    for (unsigned i = 0; i < pwa.domain.size(); ++i) {
      auto &bsdom = pwa.domain[i];
      auto &rangeVars = pwa.value[i];

      // Move the non domain variables to be existentials and symbols
      bsdom.nExist = bsdom.getNumDims() - getDomainDims() - this->getNumSyms();
      bsdom.nDim = getDomainDims();
      bsdom.nParam = this->getNumSyms();

      bsdom.insertDimensions(getDomainDims(), getRangeDims());
      bsdom.nDim += getRangeDims();

      for (unsigned var = 0; var < rangeVars.size(); ++var) {
        EqualityConstraint<Int> eq(rangeVars[var]);   
        eq.insertDimensions(getDomainDims(), getRangeDims());
        eq.setCoeff(getDomainDims() + var, -1);
        bsdom.addEquality(eq.getCoeffs());
      }

      lexMinMap.addBasicSet(bsdom);
    }
  }

  *this = lexMinMap;
  this->simplify();
}

template <typename Int>
static void negateVariables(PresburgerMap<Int> &set) {
  for (auto &bs : set.basicSets) {
    for (auto &con : bs.ineqs)
      for (unsigned i = 0; i < bs.getNumTotalDims(); i++)
        con.setCoeff(i, -con.getCoeffs()[i]);
    for (auto &con : bs.eqs)
      for (unsigned i = 0; i < bs.getNumTotalDims(); i++)
        con.setCoeff(i, -con.getCoeffs()[i]);
    for (auto &con : bs.divs)
      for (unsigned i = 0; i < bs.getNumTotalDims(); i++)
        con.setCoeff(i, -con.getCoeffs()[i]);
  }
}

template <typename Int>
void PresburgerMap<Int>::lexMaxRange() {
  negateVariables(*this);
  lexMinRange();
  negateVariables(*this);
}

#endif // MLIR_ANALYSIS_PRESBURGER_PRESBURGERMAP_IMPL_H
