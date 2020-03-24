#ifndef LIBINT_MATRIX_H
#define LIBINT_MATRIX_H

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

  Matrix(size_t rows, size_t columns);

  Matrix(const Matrix &);

  INT &operator()(size_t row, size_t column);
  INT operator()(size_t row, size_t column) const;

  void swapColumns(size_t column, size_t otherColumn);

  void swapRows(size_t row, size_t otherRow);

  size_t getNRows() const;

  size_t getNColumns() const;

  void resize(size_t newNRows, size_t newNColumns);

  friend std::ostream &operator<<(std::ostream &out, const Matrix &c);

private:
  size_t nColumns;
  std::vector<std::vector<INT>> data;
};

template <typename INT>
Matrix<INT>::Matrix(size_t rows, size_t columns)
    : nColumns{columns}, data(rows, std::vector<INT>(columns, 0)) {}

template <typename INT>
INT &Matrix<INT>::operator()(size_t row, size_t column) {
  assert(row < getNRows() && "Row outside of range");
  assert(column < getNColumns() && "Column outside of range");
  return data[row][column];
}

template <typename INT>
INT Matrix<INT>::operator()(size_t row, size_t column) const {
  assert(row < getNRows() && "Row outside of range");
  assert(column < getNColumns() && "Column outside of range");
  return data[row][column];
}

template <typename INT> size_t Matrix<INT>::getNRows() const {
  return data.size();
}

template <typename INT> size_t Matrix<INT>::getNColumns() const {
  return nColumns;
}

template <typename INT>
void Matrix<INT>::resize(size_t newNRows, size_t newNColumns) {
  nColumns = newNColumns;
  for (auto &row : data)
    row.resize(nColumns, 0);
  data.resize(newNRows, std::vector<INT>(nColumns, 0));
}

template <typename INT>
void Matrix<INT>::swapRows(size_t row, size_t otherRow) {
  assert(row < getNRows() && otherRow < getNRows()
                          && "Given row out of bounds");
  swap(data[row], data[otherRow]);
}

template <typename INT>
void Matrix<INT>::swapColumns(size_t column, size_t otherColumn) {
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
  for (size_t column = 0; column < c.getNColumns(); ++column) {
    out << "| " << std::setw(2) << column << " ";
  }
  out << std::endl;
  out << std::string(5 + c.getNColumns() * 5, '-') << std::endl;
  for (size_t row = 0; row < c.getNRows(); ++row) {
    out << std::setw(2) << row << " | ";
    for (size_t column = 0; column < c.getNColumns(); ++column) {
      out << std::setw(4) << c(row, column) << " ";
    }
    out << std::endl;
  }
  return out;
}
} // namespace mlir
#endif // LIBINT_MATRIX_H
