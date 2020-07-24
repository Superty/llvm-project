//===- PresburgerBasicSet.h - MLIR PresburgerBasicSet Class -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functionality to perform analysis on FlatAffineConstraints. In particular,
// support for performing emptiness checks.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_PRESBURGERBASICSET_H
#define MLIR_ANALYSIS_PRESBURGER_PRESBURGERBASICSET_H

#include "mlir/Analysis/Presburger/Constraint.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace analysis {
namespace presburger {

class PresburgerBasicSet {
public:
  PresburgerBasicSet() = delete;
  PresburgerBasicSet(unsigned oNDim, unsigned oNParam, unsigned oNExist)
    : nDim(oNDim), nParam(oNParam), nExist(oNExist) {}

  unsigned getNumDims() const { return nDim; }
  unsigned getNumTotalDims() const { return nParam + nDim + nExist + divs.size(); }
  unsigned getNumParams() const { return nParam; }
  unsigned getNumExists() const { return nExist; }
  unsigned getNumDivs() const { return divs.size(); }
  unsigned getNumInequalities() const { return ineqs.size(); }
  unsigned getNumEqualities() const { return eqs.size(); }
  void appendDivisionVariable(ArrayRef<int64_t> coeffs, int64_t denom);


  const InequalityConstraint &getInequality(unsigned i) const;
  const EqualityConstraint &getEquality(unsigned i) const;
  ArrayRef<InequalityConstraint> getInequalities() const;
  ArrayRef<EqualityConstraint> getEqualities() const;

  void addInequality(ArrayRef<int64_t> coeffs);
  void addEquality(ArrayRef<int64_t> coeffs);

  void removeLastInequality();
  void removeLastEquality();
  void removeLastDivision();

  void removeInequality(unsigned i);
  void removeEquality(unsigned i);
  void dump() const;

private:
  SmallVector<InequalityConstraint, 8> ineqs;
  SmallVector<EqualityConstraint, 8> eqs;
  SmallVector<DivisionConstraint, 8> divs;
  unsigned nDim, nParam, nExist;
};
} // namespace presburger
} // namespace analysis
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_PRESBURGERBASICSET_H
