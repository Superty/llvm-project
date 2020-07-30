/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 * SPDX-FileCopyrightText: 2019 Arjun Pitchanathan <arjunpitchanathan@gmail.com>
 */

#ifndef MLIR_ANALYSIS_PRESBURGER_LINEAR_TRANSFORM_H
#define MLIR_ANALYSIS_PRESBURGER_LINEAR_TRANSFORM_H

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace analysis {
namespace presburger {

// TODO this is called LinearTransform but we really just use it like a matrix,
// e.g. we talk about post-multiplying with a row rather than in linear
// algebraic terms
class LinearTransform {
public:
  using MatrixType = Matrix;
  // Return a unimodular transform which, when postmultiplied to M, brings M to
  // column echelon form.
  static LinearTransform makeTransformToColumnEchelon(MatrixType M);

  FlatAffineConstraints postMultiplyBasicSet(const FlatAffineConstraints &bs);
  SmallVector<int64_t, 8> postMultiplyRow(ArrayRef<int64_t> row);
  SmallVector<int64_t, 8> preMultiplyColumn(ArrayRef<int64_t> col);

private:
  explicit LinearTransform(MatrixType oMatrix);

  MatrixType matrix;
};

} // namespace presburger
} // namespace analysis
} // namespace mlir
#endif // MLIR_ANALYSIS_PRESBURGER_LINEAR_TRANSFORM_H
