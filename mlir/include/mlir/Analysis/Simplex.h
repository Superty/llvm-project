#ifndef LIBINT_SIMPLEX_H
#define LIBINT_SIMPLEX_H

#include "AffineStructures.h"
#include "Matrix.h"

#include <llvm/ADT/Optional.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>

#include <iostream>
#include <limits>
#include <stack>
#include <vector>

namespace mlir {

/// This class implements the Simplex algorithm. It supports adding affine
/// equaliities and inequalities, and can find a subset of these that are
/// redundant, i.e. these don't constraint the affine set further after
/// adding the non-redundant constraints.
///
/// The unknown corresponding to each row r (resp. column c) is rowVar[r] (resp.
/// colVar[c]). The first nRedundant rows of the tableau correspond to rows
/// which have been marked redundant. If at some point it is detected that
/// that the set of constraints are mutually contradictory and have no solution,
/// then empty will be set to true.
class Simplex {
public:
  enum class Direction { UP, DOWN };

  Simplex() = delete;
  Simplex(size_t nVar);
  Simplex(FlatAffineConstraints constraints);
  ~Simplex() = default;
  size_t getNRows() const;
  size_t getNColumns() const;

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
  void addIneq(int64_t constTerm, ArrayRef<int64_t> coeffs);

  /// \returns the number of variables in the tableau.
  size_t numberVariables() const;

  /// \returns the number of constraints in the tableau.
  size_t numberConstraints() const;

  /// Add an equality to the tableau. The equality is represented as
  /// constTerm + sum (coeffs[i].first * var(coeffs[i].second]) == 0.
  void addEq(int64_t constTerm, ArrayRef<int64_t> coeffs);

  /// Mark the tableau as being empty.
  void markEmpty();

  // Dump the tableau's internal state.
  void dump() const;

private:
  struct Unknown {
    Unknown(bool oOwnsRow, bool oRestricted, size_t oPos)
        : ownsRow(oOwnsRow), restricted(oRestricted), pos(oPos),
          redundant(false) {}
    Unknown() : Unknown(false, false, -1) {}
    bool ownsRow;
    bool restricted;
    size_t pos;
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
  llvm::Optional<std::pair<size_t, size_t>> findPivot(int row,
                                                     Direction direction) const;

  /// Swap the row with the column in the tableau's data structures but not the
  // tableau itself. This is used by pivot.
  void swapRowWithCol(size_t row, size_t col);

  // Pivot the row with the column.
  void pivot(size_t row, size_t col);
  void pivot(const std::pair<size_t, size_t> &p);

  /// Check if the constraint is redundant by computing its minimum value in
  /// the tableau. If this returns true, the constraint is left in row position
  /// upon return.
  ///
  /// \param conIndex must be a constraint that is not a dead column
  ///
  /// \returns True if the constraint is redundant, False otherwise.
  bool constraintIsRedundant(size_t conIndex);

  /// \returns the unknown associated with \p index.
  const Unknown &unknownFromIndex(int index) const;
  /// \returns the unknown associated with \p col.
  const Unknown &unknownFromColumn(size_t col) const;
  /// \returns the unknown associated with \p row.
  const Unknown &unknownFromRow(size_t row) const;
  /// \returns the unknown associated with \p index.
  Unknown &unknownFromIndex(int index);
  /// \returns the unknown associated with \p col.
  Unknown &unknownFromColumn(size_t col);
  /// \returns the unknown associated with \p row.
  Unknown &unknownFromRow(size_t row);

  /// Add a new row to the tableau and the associated data structures.
  size_t addRow(int64_t constTerm, ArrayRef<int64_t> coeffs);

  /// Check if there is obviously no lower bound on \p unknown.
  ///
  /// \returns True if \p unknown is obviously unbounded from below, False
  /// otherwise.
  bool minIsObviouslyUnbounded(Unknown &unknown) const;

  /// Normalize the given row by removing common factors between the numerator
  /// and the denominator.
  void normalizeRow(size_t row);

  /// Mark the row as being redundant.
  ///
  /// \returns True if the row is interchanged with a later row, False
  /// otherwise. This is used when iterating through the rows; if the return is
  /// true, the same row index must be processed again.
  bool markRedundant(size_t row);

  /// Swap the two rows in the tableau and associated data structures.
  void swapRows(size_t i, size_t j);

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
  llvm::Optional<size_t> findPivotRow(llvm::Optional<size_t> skipRow,
                                     Direction direction, size_t col) const;

  /// \returns True is diff is positive and direction is Direction::UP, or if
  /// diff is negative and direction is Direction::DOWN. Returns False otherwise.
  bool diffMatchesDirection(int64_t diff, Direction direction) const;

  /// \returns Direction::UP if \p direction is Direction::DOWN and vice versa.
  Direction flippedDirection(Direction direction) const;

  size_t nRow;
  size_t nCol;

  size_t nRedundant;
  Matrix<int64_t> tableau;
  bool empty;
  std::vector<int> rowVar, colVar;
  std::vector<Unknown> con, var;
};
} // namespace mlir
#endif // LIBINT_SIMPLEX_H
