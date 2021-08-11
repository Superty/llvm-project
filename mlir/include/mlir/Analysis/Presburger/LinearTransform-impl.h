/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 * SPDX-FileCopyrightText: 2019 Arjun Pitchanathan <arjunpitchanathan@gmail.com>
 */

#ifndef MLIR_ANALYSIS_PRESBURGER_LINEARTRANSFORM_IMPL_H
#define MLIR_ANALYSIS_PRESBURGER_LINEARTRANSFORM_IMPL_H

#include "mlir/Analysis/Presburger/LinearTransform.h"
#include "mlir/Analysis/Presburger/PresburgerBasicSet.h"

using namespace mlir;
using namespace analysis::presburger;

template <typename Int>
LinearTransform<Int>::LinearTransform(Matrix<Int> oMatrix)
    : matrix(std::move(oMatrix)) {}

// Normalize M(row, targetCol) to the range [0, M(row, sourceCol)) by
// subtracting from targetCol an appropriate integer multiple of sourceCol.
// Apply the same operation to otherMatrix. (with the same multiple)
template <typename Int>
static void subtractColumns(Matrix<Int> &m, unsigned row,
                            unsigned sourceCol, unsigned targetCol,
                            Matrix<Int> &otherMatrix) {
  assert(m(row, sourceCol) != 0 && "cannot divide by zero");
  auto ratio = floorDiv(m(row, targetCol), m(row, sourceCol));
  m.addToColumn(sourceCol, targetCol, -ratio);
  otherMatrix.addToColumn(sourceCol, targetCol, -ratio);
}

template <typename T>
T extendedEuclid(T a, T b, T &x, T &y) {
  if (b == 0) {
    x = 1;
    y = 0;
    return a;
  }
  T x1, y1;
  T d = extendedEuclid(b, T(a % b), x1, y1);
  // x1 * b + y1 * (a % b) = d
  // x1 * b + y1 * (a - (a/b)*b) = d
  // (x1 - y1 * (a/b)) *b + y1 * a = d
  x = y1;
  y = x1 - y1 * (a / b);
  return d;
}

// TODO possible optimization: would it be better to take transpose and convert
// to row echelon form, and then transpose back? Then we have row ops instead of
// column ops and we can vectorize.
//
// But at some point we need the pre-multiply version as well, so optimisation
// doesn't help in that case or only helps 1/3rd of the time (when we need both)
template <typename Int>
LinearTransform<Int> LinearTransform<Int>::makeTransformToColumnEchelon(Matrix<Int> &m) {
  // Padding of one is required by the LinearTransform constructor.
  Matrix<Int> resultMatrix = Matrix<Int>::identity(m.getNumColumns());

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
      if (m(row, col) < 0) {
        m.negateColumn(col);
        resultMatrix.negateColumn(col);
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
      if (m(row, i) < 0) {
        m.negateColumn(i);
        resultMatrix.negateColumn(i);
      }
      auto m_i = m(row, i), m_col = m(row, col);
      if (m_i % m_col == 0) {
        // If m_col divides m we directly subtract.
        subtractColumns(m, row, col, i, resultMatrix);
      } else if (m_col % m_i == 0) {
        m.swapColumns(i, col);
        resultMatrix.swapColumns(i, col);
        subtractColumns(m, row, col, i, resultMatrix);
      } else {
        Int a_i, a_col;
        extendedEuclid(m_i, m_col, a_i, a_col);

        // a_i m_i + a_col m_col = g
        // This sclaing is not valid if a_col is zero, but that only occurs if
        // m_i is the gcd, i.e., if m_i divides m_col, which will be caught by
        // the above case.
        assert(a_col != 0);
        m.scaleColumn(col, a_col);
        resultMatrix.scaleColumn(col, a_col);

        // m_col = g - a_i m_i
        m.addToColumn(i, col, a_i);
        resultMatrix.addToColumn(i, col, a_i);

        // m_col = g; make m_i zero
        subtractColumns(m, row, col, i, resultMatrix);
      }
    }

    for (unsigned targetCol = 0; targetCol < col; targetCol++) {
      if (m(row, targetCol) == 0)
        continue;
      subtractColumns(m, row, col, targetCol, resultMatrix);
    }

    ++col;
  }

  return LinearTransform(std::move(resultMatrix));
}

template <typename Int>
void LinearTransform<Int>::postMultiplyRow(ArrayRef<Int> row, SmallVector<Int, 8> &result) {
  assert(row.size() == matrix.getNumRows() &&
         "row vector dimension should be matrix output dimension");

  if constexpr (Matrix<Int>::isVectorized) {
    Vector resVec = 0;
    for (unsigned i = 0, e = matrix.getNumRows(); i < e; i++)
      resVec += mul<Matrix<Int>::isChecked>(UnderlyingInt<Int>(row[i]), matrix.getRowVector(i));
    result.reserve(matrix.getNumColumns());
    for (unsigned col = 0, e = matrix.getNumColumns(); col < e; ++col)
      result.push_back(resVec[col]);
  } else {
    result.resize(matrix.getNumColumns(), 0);
    for (unsigned col = 0, e = matrix.getNumColumns(); col < e; col++)
      for (unsigned i = 0, e = matrix.getNumRows(); i < e; i++)
        result[col] += row[i] * matrix(i, col);
  }
}

template <typename Int>
SmallVector<Int, 8>
LinearTransform<Int>::preMultiplyColumn(ArrayRef<Int> col) {
  assert(matrix.getNumColumns() == col.size() &&
         "row vector dimension should be matrix output dimension");

  SmallVector<Int, 8> result;
  for (unsigned row = 0, e = matrix.getNumRows(); row < e; row++) {
    Int elem = 0;
    for (unsigned i = 0, e = matrix.getNumColumns(); i < e; i++)
      elem += matrix(row, i) * col[i];
    result.push_back(elem);
  }
  return result;
}

// Note: only plain basic sets are passed, so no divisions.
template <typename Int>
PresburgerBasicSet<Int>
LinearTransform<Int>::postMultiplyBasicSet(const PresburgerBasicSet<Int> &bs) {
  assert(bs.isPlainBasicSet());
  PresburgerBasicSet<Int> result(bs.getNumTotalDims(), 0, 0);

  for (unsigned i = 0; i < bs.getNumEqualities(); ++i) {
    ArrayRef<Int> eq = bs.getEquality(i).getCoeffs();

    Int c = eq.back();

    SmallVector<Int, 8> newEq;
    postMultiplyRow(eq.drop_back(), newEq);
    newEq.push_back(c);
    result.addEquality(newEq);
  }

  for (unsigned i = 0; i < bs.getNumInequalities(); ++i) {
    ArrayRef<Int> ineq = bs.getInequality(i).getCoeffs();

    Int c = ineq.back();

    SmallVector<Int, 8> newIneq;
    postMultiplyRow(ineq.drop_back(), newIneq);
    newIneq.push_back(c);
    result.addInequality(newIneq);
  }

  // bs.simplify(); // isl does this here
  return result;
}

#endif // MLIR_ANALYSIS_PRESBURGER_LINEARTRANSFORM_IMPL_H
