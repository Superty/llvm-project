//===- Matrix.cpp - MLIR Matrix Class -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Matrix.h"

using namespace mlir;
using namespace analysis::presburger;

Matrix::Matrix(unsigned rows, unsigned columns)
    : nRows(rows), nColumns(columns), data(nRows * nColumns) {}

Matrix Matrix::identity(unsigned dimension) {
  Matrix matrix(dimension, dimension);
  for (unsigned i = 0; i < dimension; ++i)
    matrix(i, i) = 1;
  return matrix;
}

int64_t &Matrix::at(unsigned row, unsigned column) {
  assert(row < getNumRows() && "Row outside of range");
  assert(column < getNumColumns() && "Column outside of range");
  return data[row * nColumns + column];
}

int64_t Matrix::at(unsigned row, unsigned column) const {
  assert(row < getNumRows() && "Row outside of range");
  assert(column < getNumColumns() && "Column outside of range");
  return data[row * nColumns + column];
}

int64_t &Matrix::operator()(unsigned row, unsigned column) {
  return at(row, column);
}

int64_t Matrix::operator()(unsigned row, unsigned column) const {
  return at(row, column);
}

unsigned Matrix::getNumRows() const { return nRows; }

unsigned Matrix::getNumColumns() const { return nColumns; }

void Matrix::resize(unsigned newNRows, unsigned newNColumns) {
  if (newNColumns == nColumns) {
    nRows = newNRows;
    data.resize(nRows * nColumns);
  } else {
    SmallVector<int64_t, 8> newData;
    newData.reserve(newNRows * newNColumns);
    for (unsigned row = 0; row < newNRows; row++) {
      for (unsigned col = 0; col < newNColumns; col++) {
        newData.push_back(row < nRows && col < nColumns ? at(row, col) : 0);
      }
    }
    data = std::move(newData);
    nRows = newNRows;
    nColumns = newNColumns;
  }
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

ArrayRef<int64_t> Matrix::getRow(unsigned row) const {
  return {&data[row * nColumns], nColumns};
}

void Matrix::addToRow(unsigned sourceRow, unsigned targetRow, int64_t scale) {
  if (scale == 0)
    return;
  for (unsigned col = 0; col < nColumns; ++col)
    at(targetRow, col) += scale * at(sourceRow, col);
  return;
}

void Matrix::addToColumn(unsigned sourceColumn, unsigned targetColumn,
                         int64_t scale) {
  if (scale == 0)
    return;
  for (unsigned row = 0, e = getNumRows(); row < e; ++row)
    at(row, targetColumn) += scale * at(row, sourceColumn);
  return;
}

void Matrix::print(raw_ostream &os) const {
  for (unsigned row = 0; row < nRows; ++row) {
    for (unsigned column = 0; column < nColumns; ++column)
      os << at(row, column) << ' ';
    os << '\n';
  }
}

void Matrix::dump() const { print(llvm::errs()); }

