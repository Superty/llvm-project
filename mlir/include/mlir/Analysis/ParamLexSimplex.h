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
// #include "mlir/Analysis/AffineStructures.h"
// #include "mlir/Analysis/Presburger/Fraction.h"
// #include "mlir/Analysis/Presburger/Matrix.h"
// #include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
// #include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
struct PWAFunction {
  SmallVector<FlatAffineConstraints, 8> domain;
  SmallVector<SmallVector<SmallVector<int64_t, 8>, 8>, 8> value;

  void dump() {
    for (unsigned i = 0; i < value.size(); ++i) {
      domain[i].dump();
      llvm::errs() << "\n";
      for (unsigned j = 0; j < value[i].size(); ++j) {
        llvm::errs() << "a" << j << " = ";
        for (unsigned k = 0; k < value[i][j].size() - 1; ++k) {
          if (value[i][j][k] == 0)
            continue;
          llvm::errs() << value[i][j][k] << "x" << k << " + ";
        }
        llvm::errs() << value[i][j].back() << '\n';
      }
      llvm::errs() << '\n';
    }
  }
};

class ParamLexSimplex : public Simplex {
public:
  ParamLexSimplex() = delete;
  ParamLexSimplex(unsigned nDim, unsigned paramBegin, unsigned oNParam);
  explicit ParamLexSimplex(const FlatAffineConstraints &constraints);

  void addInequality(ArrayRef<int64_t> coeffs);
  void addEquality(ArrayRef<int64_t> coeffs);
  void appendParameter();

  PWAFunction findParamLexmin();
  void findParamLexminRecursively(Simplex &domainSimplex,
                                  FlatAffineConstraints &domainSet,
                                  PWAFunction &result);

private:
  SmallVector<int64_t, 8> getRowParamSample(unsigned row);
  LogicalResult moveRowUnknownToColumn(unsigned row);
  void restoreConsistency();
  unsigned getSnapshot();
  // SmallVector<int64_t, 8> varCoeffsFromRowCoeffs(ArrayRef<int64_t>
  // rowCoeffs) const;
  Optional<unsigned> findPivot(unsigned row) const;

  unsigned nParam, nDiv;
  SmallVector<SmallVector<int64_t, 8>, 8> originalCoeffs;
};

} // namespace mlir

#endif // MLIR_ANALYSIS_PARAMLEXSIMPLEX_H

