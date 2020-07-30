/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 * SPDX-FileCopyrightText: 2019 Arjun Pitchanathan <arjunpitchanathan@gmail.com>
 */

#include "mlir/Analysis/Presburger/LinearTransform.h"

using namespace mlir;
using namespace analysis::presburger;

LinearTransform::LinearTransform(MatrixType oMatrix)
    : matrix(std::move(oMatrix)) {}

// Normalize M(row, targetCol) to the range [0, M(row, sourceCol)) by
// subtracting from targetCol an appropriate integer multiple of sourceCol.
// Apply the same operation to otherMatrix. (with the same multiple)
static void subtractColumns(LinearTransform::MatrixType &m, unsigned row,
                            unsigned sourceCol, unsigned targetCol,
                            LinearTransform::MatrixType &otherMatrix) {
  assert(m(row, sourceCol) != 0 && "cannot divide by zero");
  auto ratio = m(row, targetCol) / m(row, sourceCol);
  m.addToColumn(sourceCol, targetCol, -ratio);
  otherMatrix.addToColumn(sourceCol, targetCol, -ratio);
}

// TODO possible optimization: would it be better to take transpose and convert
// to row echelon form, and then transpose back? Then we have row ops instead of
// column ops and we can vectorize.
//
// But at some point we need the pre-multiply version as well, so optimisation
// doesn't help in that case or only helps 1/3rd of the time (when we need both)
LinearTransform LinearTransform::makeTransformToColumnEchelon(MatrixType m) {
  // Padding of one is required by the LinearTransform constructor.
  MatrixType resultMatrix = MatrixType::identity(m.getNumColumns());

  for (unsigned row = 0, col = 0; row < m.getNumRows(); ++row) {
    bool found = false;
    for (unsigned i = col; i < m.getNumColumns(); i++) {
      // TODO possible optimization: M.elementIsNonZero(...) and use sparsity.
      if (m(row, i) == 0)
        continue;
      found = true;

      if (i != col) {
        // TODO possible optimization: swap only elements from row onwards,
        // since the rest are zero. (isl does this)
        m.swapColumns(i, col);
        resultMatrix.swapColumns(i, col);
      }
      if (m(row, i) < 0) {
        m.negateColumn(i);
        resultMatrix.negateColumn(i);
      }
      break;
    }

    // Continue to the next row on the same column, since this row doesn't
    // have any new columns.
    if (!found)
      continue;

    for (unsigned i = col + 1; i < m.getNumColumns(); ++i) {
      // TODO possible optimization: Would it be better to directly take the gcd
      // of the top elements, multiply and subtract, instead of subtracting
      // entire columns in each step of the euclidean algorithm? Or does it
      // cause enough overflows that we lose overall?

      // NOTE on isl: in isl the whole columns are swapped each time instead
      // of just swapping the indices. It is unclear if there is a special
      // reason for this.
      for (unsigned targetCol = i, sourceCol = col;
           m(row, targetCol) != 0 && m(row, sourceCol) != 0;
           std::swap(targetCol, sourceCol)) {
        subtractColumns(m, row, sourceCol, targetCol, resultMatrix);
      }

      if (m(row, col) == 0) {
        m.swapColumns(i, col);
        resultMatrix.swapColumns(i, col);
      }
    }

    for (unsigned targetCol = 0; targetCol < col; targetCol++) {
      if (m(row, targetCol) == 0)
        continue;
      subtractColumns(m, row, targetCol, col, resultMatrix);
    }

    ++col;
  }

  return LinearTransform(std::move(resultMatrix));
}

SmallVector<int64_t, 8>
LinearTransform::postMultiplyRow(ArrayRef<int64_t> row) {
  assert(row.size() == matrix.getNumRows() &&
         "row vector dimension should be matrix output dimension");

  SmallVector<int64_t, 8> result;
  for (unsigned col = 0, e = matrix.getNumColumns(); col < e; col++) {
    int64_t elem = 0;
    for (unsigned i = 0, e = matrix.getNumRows(); i < e; i++)
      elem += row[i] * matrix(i, col);
    result.push_back(elem);
  }
  return result;
}

SmallVector<int64_t, 8>
LinearTransform::preMultiplyColumn(ArrayRef<int64_t> col) {
  assert(matrix.getNumColumns() == col.size() &&
         "row vector dimension should be matrix output dimension");

  SmallVector<int64_t, 8> result;
  for (unsigned row = 0, e = matrix.getNumRows(); row < e; row++) {
    int64_t elem = 0;
    for (unsigned i = 0, e = matrix.getNumColumns(); i < e; i++)
      elem += matrix(row, i) * col[i];
    result.push_back(elem);
  }
  return result;
}

FlatAffineConstraints
LinearTransform::postMultiplyBasicSet(const FlatAffineConstraints &bs) {
  FlatAffineConstraints result(bs.getNumDimIds());

  for (unsigned i = 0; i < bs.getNumEqualities(); ++i) {
    ArrayRef<int64_t> eq = bs.getEquality(i);

    int64_t c = eq.back();

    SmallVector<int64_t, 8> newEq = postMultiplyRow(eq.drop_back());
    newEq.push_back(c);
    result.addEquality(newEq);
  }

  for (unsigned i = 0; i < bs.getNumInequalities(); ++i) {
    ArrayRef<int64_t> ineq = bs.getInequality(i);

    int64_t c = ineq.back();

    SmallVector<int64_t, 8> newIneq = postMultiplyRow(ineq.drop_back());
    newIneq.push_back(c);
    result.addInequality(newIneq);
  }

  // bs.simplify(); // isl does this here
  return result;
}
