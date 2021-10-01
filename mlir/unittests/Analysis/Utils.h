//===- Utils.h - utility functions for testing ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_UNITTESTS_ANALYSIS_FACUTILS_H
#define MLIR_UNITTESTS_ANALYSIS_FACUTILS_H

#include "mlir/Analysis/AffineStructures.h"

namespace mlir {

/// Construct a FlatAffineConstraints from a set of inequality and
/// equality constraints. `numIds` is the total number of ids, of which
/// `numLocals` is the number of local ids.
static FlatAffineConstraints
makeFACFromConstraints(unsigned numIds, ArrayRef<SmallVector<int64_t, 4>> ineqs,
                       ArrayRef<SmallVector<int64_t, 4>> eqs,
                       unsigned numLocals = 0) {
  FlatAffineConstraints fac(/*numReservedInequalities=*/ineqs.size(),
                            /*numReservedEqualities=*/eqs.size(),
                            /*numReservedCols=*/numIds + 1,
                            /*numDims=*/numIds - numLocals,
                            /*numSymbols=*/0, numLocals);
  for (const SmallVector<int64_t, 4> &eq : eqs)
    fac.addEquality(eq);
  for (const SmallVector<int64_t, 4> &ineq : ineqs)
    fac.addInequality(ineq);
  return fac;
}

/// Construct a FlatAffineConstraints having `numDims` dimensions from the given
/// set of inequality constraints. This is a convenience function to be used
/// when the FAC to be constructed does not have any local ids and does not have
/// equalties.
static FlatAffineConstraints
makeFACFromIneqs(unsigned dims, ArrayRef<SmallVector<int64_t, 4>> ineqs) {
  return makeFACFromConstraints(dims, ineqs, {});
}

} // namespace mlir

#endif // MLIR_UNITTESTS_ANALYSIS_FACUTILS_H
