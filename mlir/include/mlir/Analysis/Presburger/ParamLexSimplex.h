//===- Simplex.h - MLIR ParamLexSimplex Class -------------------*- C++ -*-===//
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

#ifndef MLIR_ANALYSIS_PRESBURGER_PARAMLEXSIMPLEX_H
#define MLIR_ANALYSIS_PRESBURGER_PARAMLEXSIMPLEX_H

#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Analysis/Presburger/PresburgerBasicSet.h"
// #include "mlir/Analysis/AffineStructures.h"
// #include "mlir/Analysis/Presburger/Fraction.h"
// #include "mlir/Analysis/Presburger/Matrix.h"
// #include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
// #include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
// #include "llvm/Support/raw_ostream.h"

namespace mlir {

struct pwaFunction {
  SmallVector<PresburgerBasicSet, 8> domain;
  SmallVector<SmallVector<SmallVector<int64_t, 8>, 8>, 8> value;
};

class ParamLexSimplex : public Simplex {
public:
  ParamLexSimplex() = delete;
  ParamLexSimplex(unsigned nDim, unsigned nParam);
  explicit ParamLexSimplex(const FlatAffineConstraints &constraints);

  void addInequality(ArrayRef<int64_t> coeffs);
  void addEquality(ArrayRef<int64_t> coeffs);
  void addDivisionVariable(ArrayRef<int64_t> coeffs, int64_t denom);


  pwaFunction findParamLexmin();
  void findParamLexminRecursively(ParamLexSimplex &domainSimplex, PresburgerBasicSet &domainSet, pwaFunction &result);

private:
  ArrayRef<int64_t> getRowParamSample(unsigned row);
  LogicalResult restoreRow(unsigned row);
  void restoreConsistency();

  Optional<unsigned> findPivot(unsigned row) const;

  unsigned nParam;
};
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_PARAMLEXSIMPLEX_H
