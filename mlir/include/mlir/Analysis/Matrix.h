#ifndef MLIR_ANALYSIS_MATRIX_H
#define MLIR_ANALYSIS_MATRIX_H

#include <cassert>
#include <iomanip>
#include <iostream>
#include <vector>

namespace mlir {

/// This is a simple class to represent a resizable matrix.
///
/// The data is stored in the form of a vector of vectors.
template <typename INT> class Matrix {
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

  friend std::ostream &operator<<(std::ostream &out, const Matrix &c);

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

template <typename INT> unsigned Matrix<INT>::getNRows() const {
  return data.size();
}

template <typename INT> unsigned Matrix<INT>::getNColumns() const {
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
  assert(row < getNRows() && otherRow < getNRows()
                          && "Given row out of bounds");
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
std::ostream &operator<<(std::ostream &out, const Matrix<INT> &c) {
  out << "Dumping matrix, rows = " << c.getNRows()
      << ", columns: " << c.getNColumns() << '\n';
  out << "r/c  ";
  for (unsigned column = 0; column < c.getNColumns(); ++column) {
    out << "| " << std::setw(2) << column << " ";
  }
  out << std::endl;
  out << std::string(5 + c.getNColumns() * 5, '-') << std::endl;
  for (unsigned row = 0; row < c.getNRows(); ++row) {
    out << std::setw(2) << row << " | ";
    for (unsigned column = 0; column < c.getNColumns(); ++column) {
      out << std::setw(4) << c(row, column) << " ";
    }
    out << std::endl;
  }
  return out;
}
} // namespace mlir
#endif // MLIR_ANALYSIS_MATRIX_H
