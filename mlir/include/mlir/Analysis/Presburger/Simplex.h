//===- Simplex.h - MLIR Simplex Class ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class can perform analysis on FlatAffineConstraints. In particular,
// it can be used to simplify the constraint set by detecting constraints
// which are redundant.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_SIMPLEX_H
#define MLIR_ANALYSIS_PRESBURGER_SIMPLEX_H

#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Analysis/AffineStructures.h"
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/Optional.h>
#include <llvm/ADT/SmallVector.h>

#include <iostream>
#include <limits>
#include <stack>
#include <vector>

namespace mlir {

/// This class implements the Simplex algorithm. It supports adding affine
/// equalities and inequalities, and can find a subset of these that are
/// redundant, i.e. these don't constraint the affine set further after
/// adding the non-redundant constraints.
///
/// An unknown is addressed by its index. If the index i is non-negative, then
/// ith variable is the Unknown being addressed. If the index is negative,
/// then a constraint is being addressed, having index ~i.
///
/// The unknown corresponding to each row r (resp. column c) has index rowVar[r]
/// (resp. colVar[c]). The first nRedundant rows of the tableau correspond to
/// rows which have been marked redundant. If at some point it is detected that
/// that the set of constraints are mutually contradictory and have no solution,
/// then empty will be set to true.
class Simplex {
public:
  enum class Direction { UP, DOWN };

  Simplex() = delete;
  Simplex(unsigned nVar);
  Simplex(FlatAffineConstraints constraints);
  ~Simplex() = default;
  unsigned getNumRows() const;
  unsigned getNumColumns() const;

  /// \returns True is the tableau is empty (has conflicting constraints),
  /// False otherwise.
  bool isEmpty() const;

  /// Check for redundant constraints and mark them as redundant.
  void detectRedundant();

  friend std::ostream &operator<<(std::ostream &out, const Simplex &s);

  /// Check whether the constraint has been marked redundant.
  bool isMarkedRedundant(int conIndex) const;

  /// Add an inequality to the tableau. The inequality is represented as
  /// constTerm + sum (coeffs[i].first * var(coeffs[i].second]) >= 0.
  void addInequality(int64_t constTerm, ArrayRef<int64_t> coeffs);

  /// \returns the number of variables in the tableau.
  unsigned numberVariables() const;

  /// \returns the number of constraints in the tableau.
  unsigned numberConstraints() const;

  /// Add an equality to the tableau. The equality is represented as
  /// constTerm + sum (coeffs[i].first * var(coeffs[i].second]) == 0.
  void addEquality(int64_t constTerm, ArrayRef<int64_t> coeffs);

  /// Mark the tableau as being empty.
  void markEmpty();

  // Dump the tableau's internal state.
  void dump() const;

private:
  struct Unknown {
    Unknown(bool oOwnsRow, bool oRestricted, unsigned oPos)
        : ownsRow(oOwnsRow), restricted(oRestricted), pos(oPos),
          redundant(false) {}
    Unknown() : Unknown(false, false, -1) {}
    bool ownsRow;
    bool restricted;
    unsigned pos;
    bool redundant;
  };

  // Dump the internal state of the unknown.
  void dumpUnknown(const Unknown &u) const;

  /// Find a pivot to change the sample value of \p row in the specified
  /// direction. The returned pivot row will be \p row if and only
  /// if the unknown is unbounded in the specified direction.
  ///
  /// \returns a [row, col] pair denoting a pivot, or an empty llvm::Optional if
  /// no valid pivot exists.
  llvm::Optional<std::pair<unsigned, unsigned>>
  findPivot(int row, Direction direction) const;

  /// Swap the row with the column in the tableau's data structures but not the
  // tableau itself. This is used by pivot.
  void swapRowWithCol(unsigned row, unsigned col);

  // Pivot the row with the column.
  void pivot(unsigned row, unsigned col);
  void pivot(const std::pair<unsigned, unsigned> &p);

  /// Check if the constraint is redundant by computing its minimum value in
  /// the tableau. If this returns true, the constraint is left in row position
  /// upon return.
  ///
  /// \param conIndex must be a constraint that is not a dead column
  ///
  /// \returns True if the constraint is redundant, False otherwise.
  bool constraintIsRedundant(unsigned conIndex);

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
  unsigned addRow(int64_t constTerm, ArrayRef<int64_t> coeffs);

  /// Check if there is obviously no lower bound on \p unknown.
  ///
  /// \returns True if \p unknown is obviously unbounded from below, False
  /// otherwise.
  bool minIsObviouslyUnbounded(Unknown &unknown) const;

  /// Normalize the given row by removing common factors between the numerator
  /// and the denominator.
  void normalizeRow(unsigned row);

  /// Mark the row as being redundant.
  ///
  /// \returns True if the row is interchanged with a later row, False
  /// otherwise. This is used when iterating through the rows; if the return is
  /// true, the same row index must be processed again.
  bool markRedundant(unsigned row);

  /// Swap the two rows in the tableau and associated data structures.
  void swapRows(unsigned i, unsigned j);

  /// Restore the unknown to a non-negative sample value.
  ///
  /// \returns True is the unknown was successfully restored to a non-negative
  /// sample value, False otherwise.
  bool restoreRow(Unknown &u);

  /// Find a row that can be used to pivot the column in the specified
  /// direction. If no direction is specified, any direction is allowed. The
  /// column unknown is assumed to be bounded in the specified direction. If no
  /// direction is specified then the column unknown is assumed to be unbounded
  /// in both directions. If \p skipRow is not null, then this row is excluded
  /// from consideration.
  ///
  /// \returns the row to pivot to, or an empty llvm::Optional if no row was
  /// found.
  llvm::Optional<unsigned> findPivotRow(llvm::Optional<unsigned> skipRow,
                                        Direction direction,
                                        unsigned col) const;

  /// \returns True is diff is positive and direction is Direction::UP, or if
  /// diff is negative and direction is Direction::DOWN. Returns False
  /// otherwise.
  bool diffMatchesDirection(int64_t diff, Direction direction) const;

  /// \returns Direction::UP if \p direction is Direction::DOWN and vice versa.
  Direction flippedDirection(Direction direction) const;

  /// The number of rows in the tableau.
  unsigned nRow;

  /// The number of columns in the tableau, including the common denominator
  /// and the constant column.
  unsigned nCol;

  /// The number of constraints marked redundant.
  unsigned nRedundant;

  /// The matrix represnting the tableau.
  Matrix<int64_t> tableau;

  /// True if the tableau has been detected to be empty, False otherwise.
  bool empty;

  /// These hold the indexes of the unknown at a given row or column position.
  std::vector<int> rowVar, colVar;

  /// These hold information about each unknown.
  std::vector<Unknown> con, var;
};
} // namespace mlir
#endif // MLIR_ANALYSIS_PRESBURGER_SIMPLEX_H
