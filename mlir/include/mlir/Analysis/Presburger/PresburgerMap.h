//===- PresburgerMap.h - MLIR PresburgerMap Class -----*- C++ -*-===//
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

#ifndef MLIR_ANALYSIS_PRESBURGER_PRESBURGERMAP_H
#define MLIR_ANALYSIS_PRESBURGER_PRESBURGERMAP_H

#include "mlir/Analysis/Presburger/Set.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace analysis {
namespace presburger {

template <typename Int>
class PresburgerMap : public PresburgerSet<Int> {
public:
  PresburgerMap(unsigned oDomainDim, unsigned oRangeDim, unsigned oNParam)
      : PresburgerSet<Int>(oDomainDim + oRangeDim, oNParam), domainDim(oDomainDim),
        rangeDim(oRangeDim) {}

  unsigned getDomainDims() const { return domainDim; }
  unsigned getRangeDims() const { return rangeDim; }

  /// Range set is created by converting all domain variables to existentials
  PresburgerSet<Int> getRangeSet() const;

  /// Domain set is created by converting all range variables to existentials
  PresburgerSet<Int> getDomainSet() const;

  void lexMinRange();
  void lexMaxRange();

  void dump() const;

  void print(raw_ostream &os) const;

private:
  unsigned domainDim, rangeDim;
};

} // namespace presburger
} // namespace analysis
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_PRESBURGERMAP_H
