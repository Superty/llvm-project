//===- Matrix.cpp - MLIR Matrix Class -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Matrix.h"

#include <algorithm>

using namespace mlir;
using namespace analysis::presburger;

Matrix::Matrix(unsigned rows, unsigned columns)
    : nRows(rows), nColumns(columns) {
	    data.resize(256);
	    std::fill(data.begin(), data.end(), 0);
    }

std::vector<SafeInteger, AlignedAllocator<SafeInteger, 64>> Matrix::data;

Matrix Matrix::identity(unsigned dimension) {
  Matrix matrix(dimension, dimension);
  for (unsigned i = 0; i < dimension; ++i)
    matrix(i, i) = 1;
  return matrix;
}

unsigned Matrix::getNumRows() const { return nRows; }

unsigned Matrix::getNumColumns() const { return nColumns; }

void Matrix::resize(unsigned newNRows, unsigned newNColumns) {
  return;	
  nRows = newNRows;
  data.resize(nRows * MATRIX_COLUMN_COUNT);
  if (newNColumns < nColumns) {
    for (unsigned row = 0; row < nRows; ++row) {
      for (unsigned col = newNColumns; col < nColumns; ++col) {
        at(row, col) = 0;
      }
    }
  }
  nColumns = newNColumns;
}

void Matrix::swapRows(unsigned row, unsigned otherRow) {
  assert((row < getNumRows() && otherRow < getNumRows()) &&
         "Given row out of bounds");
  if (row == otherRow)
    return;
  for (unsigned col = 0; col < nColumns; col++)
    std::swap(at(row, col), at(otherRow, col));
}

void Matrix::swapColumns(unsigned column, unsigned otherColumn) {
  assert((column < getNumColumns() && otherColumn < getNumColumns()) &&
         "Given column out of bounds");
  if (column == otherColumn)
    return;
  for (unsigned row = 0; row < nRows; row++)
    std::swap(at(row, column), at(row, otherColumn));
}

void Matrix::negateColumn(unsigned column) {
  assert(column < getNumColumns() && "Given column out of bounds");
  for (unsigned row = 0, e = getNumRows(); row < e; ++row) {
    // TODO not overflow safe
    at(row, column) = -at(row, column);
  }
}

ArrayRef<SafeInteger> Matrix::getRow(unsigned row) const {
  return {&data[row * MATRIX_COLUMN_COUNT], nColumns};
}

void Matrix::addToRow(unsigned sourceRow, unsigned targetRow,
                      SafeInteger scale) {
  if (scale == 0)
    return;
  for (unsigned col = 0; col < nColumns; ++col)
    at(targetRow, col) += scale * at(sourceRow, col);
  return;
}

void Matrix::scaleColumn(unsigned column, SafeInteger scale) {
  if (scale == 0)
    return;
  for (unsigned row = 0, e = getNumRows(); row < e; ++row)
    at(row, column) *= scale;
  return;
}

void Matrix::addToColumn(unsigned sourceColumn, unsigned targetColumn,
                         SafeInteger scale) {
  if (scale == 0)
    return;
  for (unsigned row = 0, e = getNumRows(); row < e; ++row)
    at(row, targetColumn) += scale * at(row, sourceColumn);
  return;
}

void Matrix::print(raw_ostream &os) const {
  for (unsigned row = 0; row < nRows; ++row) {
    for (unsigned column = 0; column < nColumns; ++column)
      os << at(row, column) << '\t';
    os << '\n';
  }
}

void Matrix::dump() const { print(llvm::errs()); }
