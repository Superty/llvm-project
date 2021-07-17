//===- PresburgerBasicMap.cpp - MLIR PresburgerBasicMap Class -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/PresburgerBasicMap.h"
#include "mlir/Analysis/Presburger/Set.h"

using namespace mlir;
using namespace mlir::analysis;
using namespace mlir::analysis::presburger;

void PresburgerBasicMap::dump() const {
  print(llvm::errs());
  llvm::errs() << "\n";
}

void PresburgerBasicMap::print(raw_ostream &os) const {
  os << "Relation : [";
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

  PresburgerBasicSet::print(os);
}

PresburgerBasicSet PresburgerBasicMap::getRangeSet() const {
  PresburgerBasicSet rangeSet = *this;
  rangeSet.convertDimsToExists(0, getDomainDims());
  return rangeSet;
}

PresburgerBasicSet PresburgerBasicMap::getDomainSet() const {
  PresburgerBasicSet domainSet = *this;
  domainSet.convertDimsToExists(getDomainDims(),
                                getDomainDims() + getRangeDims());
  return domainSet;
}

SmallVector<PresburgerBasicMap, 8>
PresburgerBasicMap::relLexMax(PresburgerBasicMap &rel1,
                              PresburgerBasicMap &rel2, unsigned level,
                              unsigned maxLevel) {
  SmallVector<PresburgerBasicMap, 8> maxRel;

  // If level excedes maxLevel, use order in which they were called
  if (level > maxLevel) {
    maxRel.push_back(rel2);
    return maxRel;
  }

  // Build set `pd` corressponding to the intersection of the read by both the
  // relations.
  //
  // Dimensions are : [\delta w, rel1 domain, rel2 domain, rel1 range]
  PresburgerBasicSet pd(1 + rel1.getDomainDims() + rel2.getDomainDims() +
                            rel1.getRangeDims(),
                        rel1.getNumParams());

  // Create set from rel1 with additional dimensions
  PresburgerBasicSet pd1 = rel1;
  pd1.insertDimensions(0, 1);
  pd1.insertDimensions(1 + rel1.getDomainDims(), rel2.getDomainDims());
  pd.intersect(pd1);

  // Create set from rel2 with additional dimensions
  PresburgerBasicSet pd2 = rel2;
  pd2.insertDimensions(0, 1 + rel1.getDomainDims());
  pd.intersect(pd2);

  // Add constraint \delta w = w1[level] - w2[level] where w1, w2 are domains
  // variables of rel1, rel2 respectively
  SmallVector<int64_t, 8> wCoeffs(pd.getNumTotalDims() + 1, 0);
  wCoeffs[0] = 1;
  wCoeffs[1 + level] = -1;
  wCoeffs[1 + rel1.getDomainDims() + level] = 1;
  pd.addEquality(wCoeffs);

  // Project out write variables other than \delta w
  pd.convertDimsToExists(1, 1 + rel1.getDomainDims() + rel2.getDomainDims());

  PresburgerBasicSet test = pd;

  // TODO: Use non-aggressive simplification
  // Simplify constraints
  pd.simplify();

  // wSet1 = rel1 and pi(pd and \delta w > 0), where pi implies converting
  // \delta w to an existential
  PresburgerBasicMap wSet1 = rel1;
  PresburgerBasicSet wPd1 = pd; 
  SmallVector<int64_t, 8> coeffsPd1(pd.getNumTotalDims() + 1, 0);
  coeffsPd1[0] = 1;
  coeffsPd1.back() = -1;
  wPd1.addInequality(coeffsPd1);
  wPd1.convertDimsToExists(0, 0);
  wPd1.insertDimensions(0, rel1.getDomainDims());
  wSet1.intersect(wPd1);

  // wSet2 = rel2 and pi(pd and \delta w < 0), where pi implies converting
  // \delta w to an existential
  PresburgerBasicMap wSet2 = rel2;
  PresburgerBasicSet wPd2 = pd; 
  SmallVector<int64_t, 8> coeffsPd2(pd.getNumTotalDims() + 1, 0);
  coeffsPd2[0] = -1;
  coeffsPd2.back() = -1;
  wPd2.addInequality(coeffsPd2);
  wPd2.convertDimsToExists(0, 0);
  wPd2.insertDimensions(0, rel2.getDomainDims());
  wSet2.intersect(wPd2);

  wSet1.simplify();
  wSet2.simplify();

  maxRel.push_back(wSet1);
  maxRel.push_back(wSet2);

  return maxRel;
}
