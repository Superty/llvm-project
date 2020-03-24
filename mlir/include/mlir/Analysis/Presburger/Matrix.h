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

#ifndef MLIR_ANALYSIS_MATRIX_H
#define MLIR_ANALYSIS_MATRIX_H

#include <llvm/Support/raw_ostream.h>
#include <llvm/ADT/SmallVector.h>

#include <cassert>
#include <iomanip>
#include <iostream>
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

  Matrix(const Matrix &);

  /// Access the element at the specified row and column.
  INT &operator()(unsigned row, unsigned column);
  INT operator()(unsigned row, unsigned column) const;

  /// Swap the given columns.
  void swapColumns(unsigned column, unsigned otherColumn);

  /// Swap the given rows.
  void swapRows(unsigned row, unsigned otherRow);

  unsigned getNRows() const;

  unsigned getNColumns() const;

  /// Resize the matrix to the specified dimensions. If a dimension is smaller,
  /// the values are truncated; if it is bigger, the new values are default
  /// initialized.
  void resize(unsigned newNRows, unsigned newNColumns);

  void dump() const;

private:
  unsigned nColumns;
  std::vector<std::vector<INT>> data;
};

template <typename INT>
Matrix<INT>::Matrix(unsigned rows, unsigned columns)
    : nColumns{columns}, data(rows, std::vector<INT>(columns)) {}

template <typename INT>
INT &Matrix<INT>::operator()(unsigned row, unsigned column) {
  assert(row < getNRows() && "Row outside of range");
  assert(column < getNColumns() && "Column outside of range");
  return data[row][column];
}

template <typename INT>
INT Matrix<INT>::operator()(unsigned row, unsigned column) const {
  assert(row < getNRows() && "Row outside of range");
  assert(column < getNColumns() && "Column outside of range");
  return data[row][column];
}

template <typename INT>
unsigned Matrix<INT>::getNRows() const {
  return data.size();
}

template <typename INT>
unsigned Matrix<INT>::getNColumns() const {
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
  assert(row < getNRows() && otherRow < getNRows() &&
         "Given row out of bounds");
  swap(data[row], data[otherRow]);
}

template <typename INT>
void Matrix<INT>::swapColumns(unsigned column, unsigned otherColumn) {
  assert(column < getNColumns() && otherColumn < getNColumns() &&
         "Given column out of bounds");
  for (auto &row : data)
    std::swap(row[column], row[otherColumn]);
}

template <typename INT>
void Matrix<INT>::dump() const {
  llvm::errs() << "Dumping matrix, rows = " << getNRows()
               << ", columns: " << getNColumns() << '\n';
  llvm::errs() << "r/c  ";
  for (unsigned column = 0; column < getNColumns(); ++column) {
    llvm::errs() << "| " << column << " ";
  }
  llvm::errs() << '\n';
  llvm::errs() << std::string(5 + getNColumns() * 5, '-') << '\n';
  for (unsigned row = 0; row < getNRows(); ++row) {
    llvm::errs() << row << " | ";
    for (unsigned column = 0; column < getNColumns(); ++column) {
      llvm::errs() << data[row][column] << " ";
    }
    llvm::errs() << '\n';
  }
}
} // namespace mlir
#endif // MLIR_ANALYSIS_MATRIX_H
