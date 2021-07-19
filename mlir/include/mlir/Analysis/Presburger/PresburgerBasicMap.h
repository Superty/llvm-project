//===- PresburgerBasicMap.h - MLIR PresburgerBasicMap Class -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functionality to perform analysis on FlatAffineConstraints as relations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_PRESBURGERBASICMAP_H
#define MLIR_ANALYSIS_PRESBURGER_PRESBURGERBASICMAP_H

#include "mlir/Analysis/Presburger/PresburgerBasicSet.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace analysis {
namespace presburger {

class PresburgerBasicMap : public PresburgerBasicSet {
public:
  PresburgerBasicMap(unsigned oDomainDim, unsigned oRangeDim, unsigned oNParam,
                     unsigned oNExist, std::string oName = "Unnamed")
      : PresburgerBasicSet(oDomainDim + oRangeDim, oNParam, oNExist),
        domainDim(oDomainDim), rangeDim(oRangeDim), name(oName) {}

  PresburgerBasicMap(unsigned oDomainDim, unsigned oRangeDim, unsigned oNParam,
                     unsigned oNExist, ArrayRef<DivisionConstraint> divs,
                     std::string oName = "Unnamed")
      : PresburgerBasicSet(oDomainDim + oRangeDim, oNParam, oNExist, divs),
        domainDim(oDomainDim), rangeDim(oRangeDim), name(oName) {}

  PresburgerBasicMap(unsigned oDomainDim, unsigned oRangeDim,
                     const PresburgerBasicSet &bset,
                     std::string oName = "Unnamed")
      : PresburgerBasicSet(oDomainDim + oRangeDim, bset.getNumParams(),
                           bset.getNumExists(), bset.getInequalities(),
                           bset.getEqualities(), bset.getDivisions()),
        domainDim(oDomainDim), rangeDim(oRangeDim), name(oName) {}

  unsigned getDomainDims() const { return domainDim; }
  unsigned getRangeDims() const { return rangeDim; }

  /// Range set is created by converting all domain variables to existentials
  PresburgerBasicSet getRangeSet() const;

  /// Domain set is created by converting all range variables to existentials
  PresburgerBasicSet getDomainSet() const;

  /// Statement that produces rel1 should come before rel2 in code
  static SmallVector<PresburgerBasicMap, 8> relLexMax(PresburgerBasicMap &rel1,
                                                      PresburgerBasicMap &rel2,
                                                      unsigned level,
                                                      unsigned maxLevel);

  void dump() const;

  void print(raw_ostream &os) const;

private:
  unsigned domainDim, rangeDim;
  std::string name;
};

} // namespace presburger
} // namespace analysis
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_PRESBURGERBASICMAP_H
