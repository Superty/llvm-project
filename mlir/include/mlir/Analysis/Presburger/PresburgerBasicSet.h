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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {

class PresburgerBasicSet {
public:
  PresburgerBasicSet() = delete;
  PresburgerBasicSet(unsigned oNDim, unsigned oNParam, unsigned oNDiv)
    : nDim(oNDim), nParam(oNParam), nDiv(oNDiv) {}

  void appendDivisionVariable(ArrayRef<int64_t> coeffs, int64_t denom);

  void addInequality(ArrayRef<int64_t> coeffs);
  void addEquality(ArrayRef<int64_t> coeffs);

  void removeLastInequality();
  void removeLastDivision();

  void dump() const;

  struct DivisionVariable {
    SmallVector<int64_t, 8> num;
    int64_t den;
  };

private:
  SmallVector<SmallVector<int64_t, 8>, 8> ineqs, eqs;
  SmallVector<DivisionVariable, 8> divs;

  unsigned nDim, nParam, nDiv;
};
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_PRESBURGERBASICSET_H
