//===- Matrix.h - MLIR Matrix Class -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This a simplex 2D matrix class that supports reading, writing, resizing,
// and swapping rows, and swapping columns.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_MATRIX_H
#define MLIR_ANALYSIS_PRESBURGER_MATRIX_H

#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <vector>

namespace mlir {

/// This is a simple class to represent a resizable matrix.
///
/// The data is stored in the form of a vector of vectors.
template <typename INT>
class Matrix {
public:
  Matrix() = delete;

  /// Construct a matrix with the specified number of rows and columns.
  /// Initially, the values are default initialized.
  Matrix(unsigned rows, unsigned columns);

  static Matrix getIdentityMatrix(unsigned dimension);

  /// Access the element at the specified row and column.
  INT &operator()(unsigned row, unsigned column);
  INT operator()(unsigned row, unsigned column) const;

  /// Swap the given columns.
  void swapColumns(unsigned column, unsigned otherColumn);

  /// Swap the given rows.
  void swapRows(unsigned row, unsigned otherRow);

  /// Negate the column.
  ///
  /// \returns True if overflow occurs, False otherwise.
  void negateColumn(unsigned column);

  unsigned getNumRows() const;

  unsigned getNumColumns() const;

  const std::vector<INT> getRow(unsigned row) const;

  void addToRow(unsigned sourceRow, unsigned targetRow, INT scale);

  void addToColumn(unsigned sourceColumn, unsigned targetColumn, INT scale);

  /// Resize the matrix to the specified dimensions. If a dimension is smaller,
  /// the values are truncated; if it is bigger, the new values are default
  /// initialized.
  void resize(unsigned newNRows, unsigned newNColumns);

  void print(llvm::raw_ostream &os) const;
  void dump() const;

private:
  unsigned nColumns;
  std::vector<std::vector<INT>> data;
};

template <typename INT>
Matrix<INT>::Matrix(unsigned rows, unsigned columns)
    : nColumns(columns), data(rows, std::vector<INT>(columns)) {}

template <typename INT>
Matrix<INT> Matrix<INT>::getIdentityMatrix(unsigned dimension) {
  Matrix<INT> matrix(dimension, dimension);
  for (size_t i = 0; i < dimension; ++i)
    matrix(i, i) = 1;
  return matrix;
}

template <typename INT>
INT &Matrix<INT>::operator()(unsigned row, unsigned column) {
  assert(row < getNumRows() && "Row outside of range");
  assert(column < getNumColumns() && "Column outside of range");
  return data[row][column];
}

template <typename INT>
INT Matrix<INT>::operator()(unsigned row, unsigned column) const {
  assert(row < getNumRows() && "Row outside of range");
  assert(column < getNumColumns() && "Column outside of range");
  return data[row][column];
}

template <typename INT>
unsigned Matrix<INT>::getNumRows() const {
  return data.size();
}

template <typename INT>
unsigned Matrix<INT>::getNumColumns() const {
  return nColumns;
}

template <typename INT>
void Matrix<INT>::resize(unsigned newNRows, unsigned newNColumns) {
  nColumns = newNColumns;
  for (auto &row : data)
    row.resize(nColumns);
  data.resize(newNRows, std::vector<INT>(nColumns));
}

template <typename INT>
void Matrix<INT>::swapRows(unsigned row, unsigned otherRow) {
  assert((row < getNumRows() && otherRow < getNumRows()) &&
         "Given row out of bounds");
  if (row == otherRow)
    return;
  swap(data[row], data[otherRow]);
}

template <typename INT>
void Matrix<INT>::swapColumns(unsigned column, unsigned otherColumn) {
  assert((column < getNumColumns() && otherColumn < getNumColumns()) &&
         "Given column out of bounds");
  if (column == otherColumn)
    return;
  for (auto &row : data)
    std::swap(row[column], row[otherColumn]);
}

template <typename INT>
void Matrix<INT>::negateColumn(unsigned column) {
  assert(column < getNumColumns() && "Given column out of bounds");
  for (unsigned row = 0, e = getNumRows(); row < e; ++row) {
    // TODO not overflow safe
    data[row][column] = -data[row][column];
  }
}

template <typename INT>
const std::vector<INT> Matrix<INT>::getRow(unsigned row) const {
  return data[row];
}

template <typename INT>
void Matrix<INT>::addToRow(unsigned sourceRow, unsigned targetRow, INT scale) {
  if (scale == 0)
    return;
  for (unsigned col = 0; col < getNumColumns(); ++col)
    data[targetRow][col] += scale * data[sourceRow][col];
  return;
}

template <typename INT>
void Matrix<INT>::addToColumn(unsigned sourceColumn, unsigned targetColumn,
                              INT scale) {
  if (scale == 0)
    return;
  for (unsigned row = 0, e = getNumRows(); row < e; ++row)
    data[row][targetColumn] += scale * data[row][sourceColumn];
  return;
}

template <typename INT>
void Matrix<INT>::print(llvm::raw_ostream &os) const {
  os << "Dumping matrix, rows = " << getNumRows()
     << ", columns: " << getNumColumns() << '\n';
  os << "r/c  ";
  for (unsigned column = 0; column < getNumColumns(); ++column)
    os << "| " << column << " ";
  os << '\n';
  os << std::string(5 + getNumColumns() * 5, '-') << '\n';
  for (unsigned row = 0; row < getNumRows(); ++row) {
    os << row << " | ";
    for (unsigned column = 0; column < getNumColumns(); ++column)
      os << data[row][column] << " ";
    os << '\n';
  }
}

template <typename INT>
void Matrix<INT>::dump() const {
  print(llvm::errs());
}

} // namespace mlir
#endif // MLIR_ANALYSIS_PRESBURGER_MATRIX_H
