/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 * SPDX-FileCopyrightText: 2019 Arjun Pitchanathan <arjunpitchanathan@gmail.com>
 */

#ifndef MLIR_ANALYSIS_PRESBURGER_LINEAR_TRANSFORM_H
#define MLIR_ANALYSIS_PRESBURGER_LINEAR_TRANSFORM_H

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Analysis/Presburger/PresburgerBasicSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace analysis {
namespace presburger {

// TODO this is called LinearTransform but we really just use it like a matrix,
// e.g. we talk about post-multiplying with a row rather than in linear
// algebraic terms

template <typename Int>
class LinearTransform {
public:
  using Vector = typename Matrix<Int>::Vector;
  // Return a unimodular transform which, when postmultiplied to M, brings M to
  // column echelon form.
  static LinearTransform makeTransformToColumnEchelon(Matrix<Int> &M);

  FlatAffineConstraints postMultiplyBasicSet(const FlatAffineConstraints &bs);
  PresburgerBasicSet<Int> postMultiplyBasicSet(const PresburgerBasicSet<Int> &bs);
  void postMultiplyRow(ArrayRef<Int> row, SmallVector<Int, 8> &result);
  SmallVector<Int, 8> preMultiplyColumn(ArrayRef<Int> col);

private:
  explicit LinearTransform(Matrix<Int> oMatrix);

  Matrix<Int> matrix;
};

} // namespace presburger
} // namespace analysis
} // namespace mlir
#endif // MLIR_ANALYSIS_PRESBURGER_LINEAR_TRANSFORM_H
