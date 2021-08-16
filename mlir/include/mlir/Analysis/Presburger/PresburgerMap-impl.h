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
void PresburgerMap<Int>::lexMinRange() {
  PresburgerMap lexMinMap(getDomainDims(), getRangeDims(), this->getNumSyms());
  for (const auto &bs : this->getBasicSets()) {
    ParamLexSimplex<Int> paramLexSimplex(bs.getNumTotalDims(), getDomainDims());
    for (const auto &div : bs.getDivisions()) {
      // The division variables must be in the same order they are stored in the
      // basic set.
      paramLexSimplex.addInequality(div.getInequalityLowerBound().getCoeffs());
      paramLexSimplex.addInequality(div.getInequalityUpperBound().getCoeffs());
    }
    for (const auto &ineq : bs.getInequalities()) {
      paramLexSimplex.addInequality(ineq.getCoeffs());
    }
    for (const auto &eq : bs.getEqualities()) {
      paramLexSimplex.addEquality(eq.getCoeffs());
    }

    paramLexSimplex.findParamLexmin().dump();
  }
}

/* template <typename Int> */
/* void PresburgerMap<Int>::lexMaxRange() { */
/*   for (auto &bs : this->basicSets) { */
/*     for (auto &con : bs.ineqs) */
/*       for (unsigned i = getDomainDims(); i < getNumDims(); i++) */
/*         con.setCoeff(i, -con.getCoeffs()[i]); */
/*     for (auto &con : bs.eqs) */
/*       for (unsigned i = getDomainDims(); i < getNumDims(); i++) */
/*         con.setCoeff(i, -con.getCoeffs()[i]); */
/*     for (auto &con : bs.divs) */
/*       for (unsigned i = getDomainDims(); i < getNumDims(); i++) */
/*         con.setCoeff(i, -con.getCoeffs()[i]); */
/*   } */

/*   lexMinRange(); */
/* } */

#endif // MLIR_ANALYSIS_PRESBURGER_PRESBURGERMAP_IMPL_H
