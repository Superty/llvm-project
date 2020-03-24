#include "mlir/Analysis/Matrix.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {

TEST(MatrixTest, ReadWrite) {
  Matrix<int> mat(5, 5);
  for (size_t row = 0; row < 5; ++row) {
    for (size_t col = 0; col < 5; ++col) {
      mat(row, col) = 10*row + col;
    }
  }
  for (size_t row = 0; row < 5; ++row) {
    for (size_t col = 0; col < 5; ++col) {
      EXPECT_EQ(mat(row, col), int(10*row + col));
    }
  }
}

TEST(MatrixTest, SwapColumns) {
  Matrix<int> mat(5, 5);
  for (size_t row = 0; row < 5; ++row) {
    for (size_t col = 0; col < 5; ++col) {
      mat(row, col) = col == 3 ? 1 : 0;
    }
  }
  mat.swapColumns(3, 1);
  for (size_t row = 0; row < 5; ++row) {
    for (size_t col = 0; col < 5; ++col) {
      EXPECT_EQ(mat(row, col), col == 1 ? 1 : 0);
    }
  }

  // swap around all the other columns, swap (1, 3) twice for no effect.
  mat.swapColumns(3, 1);
  mat.swapColumns(2, 4);
  mat.swapColumns(1, 3);
  mat.swapColumns(0, 4);

  for (size_t row = 0; row < 5; ++row) {
    for (size_t col = 0; col < 5; ++col) {
      EXPECT_EQ(mat(row, col), col == 1 ? 1 : 0);
    }
  }
}

TEST(MatrixTest, SwapRows) {
  Matrix<int> mat(5, 5);
  for (size_t row = 0; row < 5; ++row) {
    for (size_t col = 0; col < 5; ++col) {
      mat(row, col) = row == 2 ? 1 : 0;
    }
  }
  mat.swapRows(2, 0);
  for (size_t row = 0; row < 5; ++row) {
    for (size_t col = 0; col < 5; ++col) {
      EXPECT_EQ(mat(row, col), row == 0 ? 1 : 0);
    }
  }

  // swap around all the other rows, swap (2, 0) twice for no effect.
  mat.swapRows(3, 4);
  mat.swapRows(1, 4);
  mat.swapRows(2, 0);
  mat.swapRows(0, 2);

  for (size_t row = 0; row < 5; ++row) {
    for (size_t col = 0; col < 5; ++col) {
      EXPECT_EQ(mat(row, col), row == 0 ? 1 : 0);
    }
  }
}

TEST(MatrixTest, Resize) {
  Matrix<int> mat(5, 5);
  EXPECT_EQ(mat.getNRows(), 5u);
  EXPECT_EQ(mat.getNColumns(), 5u);
  for (size_t row = 0; row < 5; ++row) {
    for (size_t col = 0; col < 5; ++col) {
      mat(row, col) = 10*row + col;
    }
  }

  mat.resize(3, 3);
  EXPECT_EQ(mat.getNRows(), 3u);
  EXPECT_EQ(mat.getNColumns(), 3u);
  for (size_t row = 0; row < 3; ++row) {
    for (size_t col = 0; col < 3; ++col) {
      EXPECT_EQ(mat(row, col), int(10*row + col));
    }
  }

  mat.resize(5, 5);
  EXPECT_EQ(mat.getNRows(), 5u);
  EXPECT_EQ(mat.getNColumns(), 5u);
  for (size_t row = 0; row < 5; ++row) {
    for (size_t col = 0; col < 5; ++col) {
      EXPECT_EQ(mat(row, col), row >= 3 || col >= 3 ? 0 : int(10*row + col));
    }
  }
}
} // namespace mlir
