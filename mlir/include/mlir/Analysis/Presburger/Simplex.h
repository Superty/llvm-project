//===- Simplex.h - MLIR Simplex Class ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class can perform analysis on FlatAffineConstraints. In particular,
// it can be used to perform emptiness checks.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_SIMPLEX_H
#define MLIR_ANALYSIS_PRESBURGER_SIMPLEX_H

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Presburger/Fraction.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

class GBRSimplex;

/// This class implements the Simplex algorithm. It supports adding affine
/// equalities and inequalities, and can perform emptiness checks, i.e., it can
/// find a solution to the set of constraints if one exists, or say that the
/// set is empty if no solution exists.
///
/// An unknown is addressed by its index. If the index i is non-negative, then
/// the ith variable is the Unknown being addressed. If the index is negative,
/// then a constraint is being addressed, having index ~i.
///
/// The unknown corresponding to each row r (resp. column c) has index rowVar[r]
/// (resp. colVar[c]). If at some point it is detected that the set of
/// constraints are mutually contradictory and have no solution, then empty will
/// be set to true.
class Simplex {
public:
  enum class Direction { UP, DOWN };
  enum class UndoOp {
    DEALLOCATE,
    UNMARK_EMPTY,
  };

  Simplex() = delete;
  explicit Simplex(unsigned nVar);
  explicit Simplex(const FlatAffineConstraints &constraints);
  unsigned getNumRows() const;
  unsigned getNumColumns() const;

  /// \returns True if the tableau is empty (has conflicting constraints),
  /// False otherwise.
  bool isEmpty() const;

  /// Add an inequality to the tableau. If coeffs is c_0, c_1, ... c_n, where n
  /// is the current number of variables, then the corresponding inequality is
  /// c_n + c_0*x_0 + c_1*x_1 + ... + c_{n-1}*x_{n-1} >= 0.
  void addInequality(ArrayRef<int64_t> coeffs);

  /// \returns the number of variables in the tableau.
  unsigned numberVariables() const;

  /// \returns the number of constraints in the tableau.
  unsigned numberConstraints() const;

  /// Add an equality to the tableau. If coeffs is c_0, c_1, ... c_n, where n
  /// is the current number of variables, then the corresponding inequality is
  /// c_n + c_0*x_0 + c_1*x_1 + ... + c_{n-1}*x_{n-1} == 0.
  void addEquality(ArrayRef<int64_t> coeffs);

  /// Mark the tableau as being empty.
  void markEmpty();

  /// Get a snapshot of the current state. This is used for rolling back.
  unsigned getSnapshot() const;

  /// Rollback to a snapshot. This invalidates all later snapshots.
  void rollback(unsigned snapshot);

  /// Compute the maximum or minimum value of the given row, depending on
  /// \p direction.
  ///
  /// \returns a {num, den} pair denoting the optimum, or a null value if no
  /// optimum exists, i.e., if the expression is unbounded in this direction.
  llvm::Optional<Fraction<int64_t>> computeRowOptimum(Direction direction,
                                                      unsigned row);

  /// Compute the maximum or minimum value of the given expression, depending on
  /// \p direction.
  ///
  /// \returns a {num, den} pair denoting the optimum, or a null value if no
  /// optimum exists, i.e., if the expression is unbounded in this direction.
  llvm::Optional<Fraction<int64_t>> computeOptimum(Direction direction,
                                                   ArrayRef<int64_t> coeffs);

  /// Make a tableau to represent a pair of points in the given tableaus, one in
  /// tableau \p A and one in \p B.
  static Simplex makeProduct(const Simplex &A, const Simplex &B);

  /// \returns the current sample point if it is integral. Otherwise, returns an
  /// empty llvm::Optional.
  llvm::Optional<std::vector<int64_t>> getSamplePointIfIntegral() const;

  /// \returns an integral sample point if one exists, or an empty
  /// llvm::Optional otherwise.
  llvm::Optional<std::vector<int64_t>> findIntegerSample();

  // Print the tableau's internal state.
  void print(llvm::raw_ostream &os) const;
  void dump() const;

private:
  friend class GBRSimplex;

  struct Unknown {
    Unknown(bool oOwnsRow, bool oRestricted, unsigned oPos)
        : ownsRow(oOwnsRow), restricted(oRestricted), pos(oPos) {}
    Unknown() : Unknown(false, false, -1) {}
    bool ownsRow;
    bool restricted;
    unsigned pos;
  };

  // Dump the internal state of the unknown.
  void printUnknown(llvm::raw_ostream &os, const Unknown &u) const;

  /// Find a pivot to change the sample value of \p row in the specified
  /// direction. The returned pivot row will be \p row if and only
  /// if the unknown is unbounded in the specified direction.
  ///
  /// \returns a [row, col] pair denoting a pivot, or an empty llvm::Optional if
  /// no valid pivot exists.
  llvm::Optional<std::pair<unsigned, unsigned>>
  findPivot(int row, Direction direction) const;

  /// Swap the row with the column in the tableau's data structures but not the
  /// tableau itself. This is used by pivot.
  void swapRowWithCol(unsigned row, unsigned col);

  /// Pivot the row with the column.
  void pivot(unsigned row, unsigned col);
  void pivot(const std::pair<unsigned, unsigned> &p);

  /// \returns the unknown associated with \p index.
  const Unknown &unknownFromIndex(int index) const;
  /// \returns the unknown associated with \p col.
  const Unknown &unknownFromColumn(unsigned col) const;
  /// \returns the unknown associated with \p row.
  const Unknown &unknownFromRow(unsigned row) const;
  /// \returns the unknown associated with \p index.
  Unknown &unknownFromIndex(int index);
  /// \returns the unknown associated with \p col.
  Unknown &unknownFromColumn(unsigned col);
  /// \returns the unknown associated with \p row.
  Unknown &unknownFromRow(unsigned row);

  /// Add a new row to the tableau and the associated data structures.
  unsigned addRow(ArrayRef<int64_t> coeffs);

  /// Normalize the given row by removing common factors between the numerator
  /// and the denominator.
  void normalizeRow(unsigned row);

  /// Swap the two rows in the tableau and associated data structures.
  void swapRows(unsigned i, unsigned j);

  /// Restore the unknown to a non-negative sample value.
  ///
  /// \returns True if the unknown was successfully restored to a non-negative
  /// sample value, False otherwise.
  bool restoreRow(Unknown &u);

  void undoOp(UndoOp op, llvm::Optional<int> index);

  /// Find a row that can be used to pivot the column in the specified
  /// direction. If \p skipRow is not null, then this row is excluded
  /// from consideration. The returned pivot will maintain all constraints
  /// except the column itself and \p skipRow, if it is set. (if these unknowns
  /// are restricted).
  ///
  /// \returns the row to pivot to, or an empty llvm::Optional if the column
  /// is unbounded in the specified direction.
  llvm::Optional<unsigned> findPivotRow(llvm::Optional<unsigned> skipRow,
                                        Direction direction,
                                        unsigned col) const;

  /// \returns True \p value is positive and direction is Direction::UP, or if
  /// \p value is negative and direction is Direction::DOWN. Returns False
  /// otherwise.
  bool signMatchesDirection(int64_t value, Direction direction) const;

  /// \returns Direction::UP if \p direction is Direction::DOWN and vice versa.
  Direction flippedDirection(Direction direction) const;

  /// Searches for an integer sample point recursively using a branch and bound
  /// algorithm and general basis reduction.
  llvm::Optional<std::vector<int64_t>>
  findIntegerSampleRecursively(Matrix<int64_t> &basis, unsigned level);

  /// Reduce the given basis, starting at the specified level, using general
  /// basis reduction.
  void reduceBasis(Matrix<int64_t> &basis, unsigned level);

  /// The number of rows in the tableau.
  unsigned nRow;

  /// The number of columns in the tableau, including the common denominator
  /// and the constant column.
  unsigned nCol;

  /// The matrix representing the tableau.
  Matrix<int64_t> tableau;

  /// True if the tableau has been detected to be empty, False otherwise.
  bool empty;

  /// Holds a log of operations, used for rolling back to a previoous state.
  std::vector<std::pair<UndoOp, llvm::Optional<int>>> undoLog;

  /// These hold the indexes of the unknown at a given row or column position.
  std::vector<int> rowVar, colVar;

  /// These hold information about each unknown.
  std::vector<Unknown> con, var;
};
} // namespace mlir
#endif // MLIR_ANALYSIS_PRESBURGER_SIMPLEX_H
