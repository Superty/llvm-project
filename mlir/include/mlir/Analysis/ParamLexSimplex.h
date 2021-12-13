//===- ParamLexSimplex.h - MLIR ParamLexSimplex Class -----------*- C++ -*-===//
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

#ifndef MLIR_ANALYSIS_PARAMLEXSIMPLEX_H
#define MLIR_ANALYSIS_PARAMLEXSIMPLEX_H

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Analysis/PWAFunction.h"
// #include "mlir/Analysis/AffineStructures.h"
// #include "mlir/Analysis/Presburger/Fraction.h"
// #include "mlir/Analysis/Presburger/Matrix.h"
// #include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
// #include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
class ParamLexSimplex : public LexSimplex {
public:
  ParamLexSimplex() = delete;
  ParamLexSimplex(unsigned nDim, unsigned paramBegin, unsigned oNParam);
  explicit ParamLexSimplex(const FlatAffineConstraints &constraints);

  void appendParameter();
  PWAFunction findParamLexmin();
  void findParamLexminRecursively(Simplex &domainSimplex,
                                  FlatAffineConstraints &domainSet,
                                  PWAFunction &result);

private:
  SmallVector<int64_t, 8> getRowParamSample(unsigned row);

  SmallVector<SmallVector<int64_t, 8>, 8> originalCoeffs;
};

} // namespace mlir

#endif // MLIR_ANALYSIS_PARAMLEXSIMPLEX_H

