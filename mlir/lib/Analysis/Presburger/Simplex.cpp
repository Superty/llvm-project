//===- Simplex.cpp - MLIR Simplex Class -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Support/MathExtras.h"

using namespace mlir;
using namespace analysis::presburger;

using Direction = Simplex::Direction;
const int nullIndex = std::numeric_limits<int>::max();

/// Construct a Simplex object with `nVar` variables.
Simplex::Simplex(unsigned nVar)
    : nRow(0), nCol(2), nRedundant(0), liveColBegin(2), tableau(0, 2 + nVar),
      empty(false) {
  colUnknown.push_back(nullIndex);
  colUnknown.push_back(nullIndex);
  for (unsigned i = 0; i < nVar; ++i) {
    var.emplace_back(Orientation::Column, /*restricted=*/false, /*pos=*/nCol);
    colUnknown.push_back(i);
    nCol++;
  }
}

Simplex::Simplex(const FlatAffineConstraints &constraints)
    : Simplex(constraints.getNumIds()) {
  addFlatAffineConstraints(constraints);
}

const Simplex::Unknown &Simplex::unknownFromIndex(int index) const {
  assert(index != nullIndex && "nullIndex passed to unknownFromIndex");
  return index >= 0 ? var[index] : con[~index];
}

const Simplex::Unknown &Simplex::unknownFromColumn(unsigned col) const {
  assert(col < nCol && "Invalid column");
  return unknownFromIndex(colUnknown[col]);
}

const Simplex::Unknown &Simplex::unknownFromRow(unsigned row) const {
  assert(row < nRow && "Invalid row");
  return unknownFromIndex(rowUnknown[row]);
}

Simplex::Unknown &Simplex::unknownFromIndex(int index) {
  assert(index != nullIndex && "nullIndex passed to unknownFromIndex");
  return index >= 0 ? var[index] : con[~index];
}

Simplex::Unknown &Simplex::unknownFromColumn(unsigned col) {
  assert(col < nCol && "Invalid column");
  return unknownFromIndex(colUnknown[col]);
}

Simplex::Unknown &Simplex::unknownFromRow(unsigned row) {
  assert(row < nRow && "Invalid row");
  return unknownFromIndex(rowUnknown[row]);
}

void Simplex::addZeroConstraint() {
  ++nRow;
  // If the tableau is not big enough to accomodate the extra row, we extend it.
  if (nRow >= tableau.getNumRows())
    tableau.resize(nRow, nCol);
  rowUnknown.push_back(~con.size());
  con.emplace_back(Orientation::Row, false, nRow - 1);

  tableau(nRow - 1, 0) = 1;
  tableau(nRow - 1, 1) = 0;
  // TODO: Should this start at 2 and not liveColBegin?
  for (unsigned col = liveColBegin; col < nCol; ++col)
    tableau(nRow - 1, col) = 0;

  // Push to undo log along with the index of the new constraint.
  undoLog.push_back(UndoLogEntry::RemoveLastConstraint);
}

void Simplex::addDivisionVariable(ArrayRef<int64_t> coeffs, int64_t denom) {
  addVariable();
  
  SmallVector<int64_t, 8> ineq(coeffs.begin(), coeffs.end());
  int64_t constTerm = ineq.back();
  ineq.back() = -denom;
  ineq.push_back(constTerm);
  addInequality(ineq);

  for (int64_t &coeff : ineq)
    coeff = -coeff;
  ineq.back() += denom - 1;
  addInequality(ineq);
}

/// Add a new row to the tableau corresponding to the given constant term and
/// list of coefficients. The coefficients are specified as a vector of
/// (variable index, coefficient) pairs.
unsigned Simplex::addRow(ArrayRef<int64_t> coeffs) {
  assert(coeffs.size() == 1 + var.size() &&
         "Incorrect number of coefficients!");
  
  addZeroConstraint();

  tableau(nRow - 1, 1) = coeffs.back();
  // Process each given variable coefficient.
  for (unsigned i = 0; i < var.size(); ++i) {
    unsigned pos = var[i].pos;
    if (coeffs[i] == 0)
      continue;

    if (var[i].orientation == Orientation::Column) {
      // If a variable is in column position at column col, then we just add the
      // coefficient for that variable (scaled by the common row denominator) to
      // the corresponding entry in the new row.
      tableau(nRow - 1, pos) += coeffs[i] * tableau(nRow - 1, 0);
      continue;
    }

    // If the variable is in row position, we need to add that row to the new
    // row, scaled by the coefficient for the variable, accounting for the two
    // rows potentially having different denominators. The new denominator is
    // the lcm of the two.
    int64_t lcm = mlir::lcm(tableau(nRow - 1, 0), tableau(pos, 0));
    int64_t nRowCoeff = lcm / tableau(nRow - 1, 0);
    int64_t idxRowCoeff = coeffs[i] * (lcm / tableau(pos, 0));
    tableau(nRow - 1, 0) = lcm;
    for (unsigned col = 1; col < nCol; ++col)
      tableau(nRow - 1, col) =
          nRowCoeff * tableau(nRow - 1, col) + idxRowCoeff * tableau(pos, col);
  }

  normalizeRow(nRow - 1);
  // Push to undo log along with the index of the new constraint.
  undoLog.emplace_back(UndoLogEntry::RemoveLastConstraint, Optional<int>());
  return con.size() - 1;
}

/// Normalize the row by removing factors that are common between the
/// denominator and all the numerator coefficients.
void Simplex::normalizeRow(unsigned row) {
  int64_t gcd = 0;
  for (unsigned col = 0; col < nCol; ++col) {
    if (gcd == 1)
      break;
    gcd = llvm::greatestCommonDivisor(gcd, std::abs(tableau(row, col)));
  }
  if (gcd == 0 || gcd == 1)
    return;
  for (unsigned col = 0; col < nCol; ++col)
    tableau(row, col) /= gcd;
}

namespace {
bool signMatchesDirection(int64_t elem, Direction direction) {
  assert(elem != 0 && "elem should not be 0");
  return direction == Direction::Up ? elem > 0 : elem < 0;
}

Direction flippedDirection(Direction direction) {
  return direction == Direction::Up ? Direction::Down : Direction::Up;
}
} // anonymous namespace

/// Find a pivot to change the sample value of the row in the specified
/// direction. The returned pivot row will involve `row` if and only if the
/// unknown is unbounded in the specified direction.
///
/// To increase (resp. decrease) the value of a row, we need to find a live
/// column with a non-zero coefficient. If the coefficient is positive, we need
/// to increase (decrease) the value of the column, and if the coefficient is
/// negative, we need to decrease (increase) the value of the column. Also,
/// we cannot decrease the sample value of restricted columns.
///
/// If multiple columns are valid, we break ties by considering a lexicographic
/// ordering where we prefer unknowns with lower index.
Optional<Simplex::Pivot> Simplex::findPivot(int row,
                                            Direction direction) const {
  Optional<unsigned> col;
  for (unsigned j = liveColBegin; j < nCol; ++j) {
    int64_t elem = tableau(row, j);
    if (elem == 0)
      continue;

    if (unknownFromColumn(j).restricted &&
        !signMatchesDirection(elem, direction))
      continue;
    if (!col || colUnknown[j] < colUnknown[*col])
      col = j;
  }

  if (!col)
    return {};

  Direction newDirection =
      tableau(row, *col) < 0 ? flippedDirection(direction) : direction;
  Optional<unsigned> maybePivotRow = findPivotRow(row, newDirection, *col);
  return Pivot{maybePivotRow.getValueOr(row), *col};
}

/// Swap the associated unknowns for the row and the column.
///
/// First we swap the index associated with the row and column. Then we update
/// the unknowns to reflect their new position and orientation.
void Simplex::swapRowWithCol(unsigned row, unsigned col) {
  std::swap(rowUnknown[row], colUnknown[col]);
  Unknown &uCol = unknownFromColumn(col);
  Unknown &uRow = unknownFromRow(row);
  uCol.orientation = Orientation::Column;
  uRow.orientation = Orientation::Row;
  uCol.pos = col;
  uRow.pos = row;
}

void Simplex::pivot(Pivot pair) { pivot(pair.row, pair.column); }

/// Pivot pivotRow and pivotCol.
///
/// Let R be the pivot row unknown and let C be the pivot col unknown.
/// Since initially R = a*C + sum b_i * X_i
/// (where the sum is over the other column's unknowns, x_i)
/// C = (R - (sum b_i * X_i))/a
///
/// Let u be some other row unknown.
/// u = c*C + sum d_i * X_i
/// So u = c*(R - sum b_i * X_i)/a + sum d_i * X_i
///
/// This results in the following transform:
///            pivot col    other col                   pivot col    other col
/// pivot row     a             b       ->   pivot row     1/a         -b/a
/// other row     c             d            other row     c/a        d - bc/a
///
/// Taking into account the common denominators p and q:
///
///            pivot col    other col                    pivot col   other col
/// pivot row     a/p          b/p     ->   pivot row      p/a         -b/a
/// other row     c/q          d/q          other row     cp/aq    (da - bc)/aq
///
/// The pivot row transform is accomplished be swapping a with the pivot row's
/// common denominator and negating the pivot row except for the pivot column
/// element.
void Simplex::pivot(unsigned pivotRow, unsigned pivotCol) {
  assert((pivotRow >= nRedundant && pivotCol >= liveColBegin) &&
         "Refusing to pivot redundant row or invalid column");

  swapRowWithCol(pivotRow, pivotCol);
  std::swap(tableau(pivotRow, 0), tableau(pivotRow, pivotCol));
  // We need to negate the whole pivot row except for the pivot column.
  if (tableau(pivotRow, 0) < 0) {
    // If the denominator is negative, we negate the row by simply negating the
    // denominator.
    tableau(pivotRow, 0) = -tableau(pivotRow, 0);
    tableau(pivotRow, pivotCol) = -tableau(pivotRow, pivotCol);
  } else {
    for (unsigned col = 1; col < nCol; ++col) {
      if (col == pivotCol)
        continue;
      tableau(pivotRow, col) = -tableau(pivotRow, col);
    }
  }
  normalizeRow(pivotRow);

  for (unsigned row = 0; row < nRow; ++row) {
    if (row == pivotRow)
      continue;
    if (tableau(row, pivotCol) == 0) // Nothing to do.
      continue;
    tableau(row, 0) *= tableau(pivotRow, 0);
    for (unsigned j = 1; j < nCol; ++j) {
      if (j == pivotCol)
        continue;
      // Add rather than subtract because the pivot row has been negated.
      tableau(row, j) = tableau(row, j) * tableau(pivotRow, 0) +
                        tableau(row, pivotCol) * tableau(pivotRow, j);
    }
    tableau(row, pivotCol) *= tableau(pivotRow, pivotCol);
    normalizeRow(row);
  }
}

/// Perform pivots until the unknown has a non-negative sample value or until
/// no more upward pivots can be performed. Return the sign of the final sample
/// value.
LogicalResult Simplex::restoreRow(Unknown &u) {
  assert(u.orientation == Orientation::Row &&
         "unknown should be in row position");

  while (tableau(u.pos, 1) < 0) {
    Optional<Pivot> maybePivot = findPivot(u.pos, Direction::Up);
    if (!maybePivot)
      break;

    pivot(*maybePivot);
    if (u.orientation == Orientation::Column)
      return LogicalResult::Success; // the unknown is unbounded above.
  }
  return success(tableau(u.pos, 1) >= 0);
}

/// Find a row that can be used to pivot the column in the specified direction.
/// This returns an empty optional if and only if the column is unbounded in the
/// specified direction (ignoring skipRow, if skipRow is set).
///
/// If skipRow is set, this row is not considered, and (if it is restricted) its
/// restriction may be violated by the returned pivot. Usually, skipRow is set
/// because we don't want to move it to column position unless it is unbounded,
/// and we are either trying to increase the value of skipRow or explicitly
/// trying to make skipRow negative, so we are not concerned about this.
///
/// If the direction is up (resp. down) and a restricted row has a negative
/// (positive) coefficient for the column, then this row imposes a bound on how
/// much the sample value of the column can change. Such a row with constant
/// term c and coefficient f for the column imposes a bound of c/|f| on the
/// change in sample value (in the specified direction). (note that c is
/// non-negative here since the row is restricted and the tableau is consistent)
///
/// We iterate through the rows and pick the row which imposes the most
/// stringent bound, since pivoting with a row changes the row's sample value to
/// 0 and hence saturates the bound it imposes. We break ties between rows that
/// impose the same bound by considering a lexicographic ordering where we
/// prefer unknowns with lower index value.
Optional<unsigned> Simplex::findPivotRow(Optional<unsigned> skipRow,
                                         Direction direction,
                                         unsigned col) const {
  Optional<unsigned> retRow;
  int64_t retElem, retConst;
  for (unsigned row = nRedundant; row < nRow; ++row) {
    if (skipRow && row == *skipRow)
      continue;
    int64_t elem = tableau(row, col);
    if (elem == 0)
      continue;
    if (!unknownFromRow(row).restricted)
      continue;
    if (signMatchesDirection(elem, direction))
      continue;
    int64_t constTerm = tableau(row, 1);

    if (!retRow) {
      retRow = row;
      retElem = elem;
      retConst = constTerm;
      continue;
    }

    int64_t diff = retConst * elem - constTerm * retElem;
    if ((diff == 0 && rowUnknown[row] < rowUnknown[*retRow]) ||
        (diff != 0 && !signMatchesDirection(diff, direction))) {
      retRow = row;
      retElem = elem;
      retConst = constTerm;
    }
  }
  return retRow;
}

bool Simplex::isEmpty() const { return empty; }

void Simplex::swapRows(unsigned i, unsigned j) {
  if (i == j)
    return;
  tableau.swapRows(i, j);
  std::swap(rowUnknown[i], rowUnknown[j]);
  unknownFromRow(i).pos = i;
  unknownFromRow(j).pos = j;
}

/// Mark this tableau empty and push an entry to the undo stack.
void Simplex::markEmpty() {
  // If the set is already empty, then we shouldn't add another UnmarkEmpty log
  // entry, since in that case the Simplex will be erroneously marked as
  // non-empty when rolling back past this point.
  if (empty)
    return;
  undoLog.emplace_back(UndoLogEntry::UnmarkEmpty, Optional<int>());
  empty = true;
}

/// Find out if the constraint is redundant by computing its minimum value in
/// the tableau. If this returns true, the constraint is left in row position
/// upon return.
///
/// The constraint is redundant if the minimal value of the unknown (while
/// respecting the other non-redundant constraints) is non-negative.
///
/// If the unknown is in column position, we try to pivot it down to a row. If
/// no pivot is found, this means that the constraint is unbounded below, i.e.
/// it is not redundant, so we return false.
///
/// Otherwise, the constraint is in row position. We keep trying to pivot
/// downwards until the sample value becomes negative. If the next pivot would
/// move the unknown to column position, then it is unbounded below and we can
/// return false. If no more pivots are possible and the sample value is still
/// non-negative, return true.
///
/// Otherwise, if the unknown has a negative sample value, then it is
/// not redundant, so we restore the row to a non-negative value and return.
bool Simplex::constraintIsRedundant(unsigned conIndex) {
  if (con[conIndex].redundant)
    return true;

  if (con[conIndex].orientation == Orientation::Column) {
    unsigned col = con[conIndex].pos;
    auto maybeRow = findPivotRow({}, Direction::Down, col);
    if (!maybeRow)
      return false;
    assert(col >= 2 && "bullshit");
    pivot(*maybeRow, col);
  }

  while (tableau(con[conIndex].pos, 1) >= 0) {
    auto maybePivot = findPivot(con[conIndex].pos, Direction::Down);
    if (!maybePivot)
      return true;

    if (maybePivot->row == con[conIndex].pos)
      return false;
    pivot(*maybePivot);
  }

  if (tableau(con[conIndex].pos, 1) >= 0)
    return true;

  LogicalResult result = restoreRow(con[conIndex]);
  assert(succeeded(result) && "Constraint was not restored succesfully!");
  return false;
}

bool Simplex::isMarkedRedundant(int conIndex) const {
  return con[conIndex].redundant;
}

/// Check whether the constraint is an equality.
///
/// The constraint is an equality if it has been marked zero, if it is a dead
/// column, or if the row is obviously equal to zero.
bool Simplex::constraintIsEquality(int conIndex) const {
  const Unknown &u = con[conIndex];
  if (u.zero)
    return true;
  if (u.redundant)
    return false;
  if (u.orientation == Orientation::Column)
    return u.pos < liveColBegin;
  return rowIsObviouslyZero(u.pos);
}

/// Add an inequality to the tableau. If coeffs is c_0, c_1, ... c_n, where n
/// is the curent number of variables, then the corresponding inequality is
/// c_n + c_0*x_0 + c_1*x_1 + ... + c_{n-1}*x_{n-1} >= 0.
///
/// We add the inequality and mark it as restricted. We then try to make its
/// sample value non-negative. If this is not possible, the tableau has become
/// empty and we mark it as such.
void Simplex::addInequality(ArrayRef<int64_t> coeffs) {
  unsigned conIndex = addRow(coeffs);
  Unknown &u = con[conIndex];
  u.restricted = true;
  LogicalResult result = restoreRow(u);
  if (failed(result))
    markEmpty();
}

/// Add an equality to the tableau. If coeffs is c_0, c_1, ... c_n, where n
/// is the curent number of variables, then the corresponding equality is
/// c_n + c_0*x_0 + c_1*x_1 + ... + c_{n-1}*x_{n-1} == 0.
///
/// We simply add two opposing inequalities, which force the expression to
/// be zero.
void Simplex::addEquality(ArrayRef<int64_t> coeffs) {
  addInequality(coeffs);
  SmallVector<int64_t, 8> negatedCoeffs;
  for (int64_t coeff : coeffs)
    negatedCoeffs.emplace_back(-coeff);
  addInequality(negatedCoeffs);
}

/// Mark the row as being redundant and push an entry to the undo stack.
///
/// Since all the rows are stored contiguously as the first nRedundant rows,
/// we move our row to the row at position nRedundant.
bool Simplex::markRedundant(unsigned row) {
  assert(!unknownFromRow(row).redundant && "Row is already marked redundant");
  assert(row >= nRedundant &&
         "Row is not marked redundant but row < nRedundant");

  undoLog.emplace_back(UndoLogEntry::UnmarkRedundant, rowUnknown[row]);

  Unknown &unknown = unknownFromRow(row);
  unknown.redundant = true;
  swapRows(row, nRedundant);
  nRedundant++;
  return false;
}

/// Check for redundant constraints and mark them as redundant.
/// A constraint is considered redundant if the other non-redundant constraints
/// already force this constraint to be non-negative.
///
/// Now for each constraint that hasn't already been marked redundant, we check
/// if it is redundant via constraintIsRedundant, and if it is, mark it as such.
void Simplex::detectRedundant() {
  if (empty)
    return;
  for (int i = con.size() - 1; i >= 0; i--) {
    if (con[i].redundant)
      continue;
    if (constraintIsRedundant(i)) {
      // constraintIsRedundant must leave the constraint in row position if it
      // returns true.
      assert(con[i].orientation == Orientation::Row &&
             "Constraint to be marked redundant must be a row!");
      markRedundant(con[i].pos);
    }
  }
}

void Simplex::addVariable() {
  undoLog.push_back(UndoLogEntry::RemoveLastVariable);
  nCol++;
  tableau.resize(nRow, nCol);
  var.emplace_back(Orientation::Column, /*restricted=*/false, /*pos=*/nCol - 1);
  colUnknown.push_back(var.size() - 1);
}

unsigned Simplex::numVariables() const { return var.size(); }
unsigned Simplex::numConstraints() const { return con.size(); }

/// Return a snapshot of the curent state. This is just the current size of the
/// undo log.
unsigned Simplex::getSnapshot() const { return undoLog.size(); }
unsigned Simplex::getSnapshotBasis() {
  SmallVector<int, 8> basis;
  for (int index : colUnknown) {
    if (index != nullIndex)
      basis.push_back(index);
  }
  savedBases.push_back(std::move(basis));
  
  return getSnapshot();
}

void Simplex::undo(UndoLogEntry entry, Optional<int> index) {
  if (entry == UndoLogEntry::RemoveLastConstraint) {
    Unknown &constraint = con.back();
    if (constraint.orientation == Orientation::Column) {
      unsigned column = constraint.pos;
      Optional<unsigned> row;

      // Try to find any pivot row for this column that preserves tableau
      // consistency (except possibly the column itself, which is going to be
      // deallocated anyway).
      //
      // If no pivot row is found in either direction, then the unknown is
      // unbounded in both directions and we are free to
      // perform any pivot at all. To do this, we just need to find any row with
      // a non-zero coefficient for the column.
      if (Optional<unsigned> maybeRow =
              findPivotRow({}, Direction::Up, column)) {
        row = *maybeRow;
      } else if (Optional<unsigned> maybeRow =
                     findPivotRow({}, Direction::Down, column)) {
        row = *maybeRow;
      } else {
        // The loop doesn't find a pivot row only if the column has zero
        // coefficients for every row. But the unknown is a constraint,
        // so it was added initially as a row. Such a row could never have been
        // pivoted to a column. So a pivot row will always be found.
        for (unsigned i = nRedundant; i < nRow; ++i) {
          if (tableau(i, column) != 0) {
            row = i;
            break;
          }
        }
      }
      assert(row.hasValue() && "No pivot row found!");
      pivot(*row, column);
    }

    // Move this unknown to the last row and remove the last row from the
    // tableau.
    swapRows(constraint.pos, nRow - 1);
    // It is not strictly necessary to shrink the tableau, but for now we
    // maintain the invariant that the tableau has exactly nRow rows.
    tableau.resize(nRow - 1, nCol);
    nRow--;
    rowUnknown.pop_back();
    con.pop_back();
  } else if (entry == UndoLogEntry::UnmarkEmpty) {
    empty = false;
  } else if (entry == UndoLogEntry::UnmarkRedundant) {
    assert(index.hasValue() &&
           "UNMARK_REDUNDANT undo entry must be accompanied by an index");
    assert(nRedundant != 0 && "No redundant constraints present.");

    Unknown &unknown = unknownFromIndex(*index);
    assert(unknown.orientation == Orientation::Row &&
           "Constraint to be unmarked as redundant must be a row");
    swapRows(unknown.pos, nRedundant - 1);
    unknown.redundant = false;
    nRedundant--;
  } else if (entry == UndoLogEntry::UnmarkZero) {
    Unknown &unknown = unknownFromIndex(*index);
    if (unknown.orientation == Orientation::Column) {
      assert(unknown.pos == liveColBegin - 1 &&
             "Column to be revived should be the last dead column");
      liveColBegin--;
    }
    unknown.zero = false;
  } else if (entry == UndoLogEntry::RemoveLastVariable) {
    assert(var.back().orientation == Orientation::Column && "Variable was moved to row position!");
    tableau.resize(nRow, nCol - 1);
    var.pop_back();
    colUnknown.pop_back();
    nCol--;
  } else if (entry == UndoLogEntry::RestoreBasis) {
    assert(!savedBases.empty() && "No bases saved!");

    auto basis = std::move(savedBases.back());
    savedBases.pop_back();

    for (int index : basis) {
      Unknown &u = unknownFromIndex(index);
      if (u.orientation == Orientation::Column)
        continue;
      for (unsigned col = 0; col < nCol; col++) {
        if (colUnknown[col] == nullIndex)
          continue;
        if (std::count(basis.begin(), basis.end(), colUnknown[col]) == 0)
          continue;
        if (tableau(u.pos, col) == 0)
          continue;
        pivot(u.pos, col);
        break;
      }

      assert(u.orientation == Orientation::Column && "Basis unknown is still a row!");
    }
  }
}

/// Rollback to the specified snapshot.
///
/// We undo all the log entries until the log size when the snapshot was taken
/// is reached.
void Simplex::rollback(unsigned snapshot) {
  while (undoLog.size() > snapshot) {
    auto entry = undoLog.back();
    undoLog.pop_back();
    undo(entry.first, entry.second);
  }
}

Optional<Fraction> Simplex::computeRowOptimum(Direction direction,
                                              unsigned row) {
  // Keep trying to find a pivot for the row in the specified direction.
  while (Optional<Pivot> maybePivot = findPivot(row, direction)) {
    // If findPivot returns a pivot involving the row itself, then the optimum
    // is unbounded, so we return None.
    if (maybePivot->row == row)
      return {};
    pivot(*maybePivot);
  }

  // The row has reached its optimal sample value, which we return.
  // The sample value is the entry in the constant column divided by the common
  // denominator for this row.
  return Fraction(tableau(row, 1), tableau(row, 0));
}

/// Compute the optimum of the specified expression in the specified direction,
/// or None if it is unbounded.
Optional<Fraction> Simplex::computeOptimum(Direction direction,
                                           ArrayRef<int64_t> coeffs) {
  assert(!empty && "Tableau should not be empty");

  unsigned snapshot = getSnapshot();
  unsigned conIndex = addRow(coeffs);
  unsigned row = con[conIndex].pos;
  Optional<Fraction> optimum = computeRowOptimum(direction, row);
  rollback(snapshot);
  return optimum;
}

bool Simplex::isUnbounded() {
  if (empty)
    return false;

  SmallVector<int64_t, 8> dir(var.size() + 1);
  for (unsigned i = 0; i < var.size(); ++i) {
    dir[i] = 1;

    Optional<Fraction> maybeMax = computeOptimum(Direction::Up, dir);
    if (!maybeMax)
      return true;

    Optional<Fraction> maybeMin = computeOptimum(Direction::Down, dir);
    if (!maybeMin)
      return true;

    dir[i] = 0;
  }
  return false;
}

/// Make a tableau to represent a pair of points in the original tableau.
///
/// The product constraints and variables are stored as: first A's, then B's.
///
/// The product tableau has row layout:
///   A's redundant rows, B's redundant rows, A's other rows, B's other rows.
///
/// It has column layout:
///   denominator, constant, A's columns, B's columns.
/// TODO reconsider the following:
/// TODO we don't need the dead columns or the redundant constraints. The caller
/// only cares about duals of the new constraints that are added after returning
/// from this function, so we can safely drop redundant constraints from the
/// original simplex.
Simplex Simplex::makeProduct(const Simplex &a, const Simplex &b) {
  unsigned numVar = a.numVariables() + b.numVariables();
  unsigned numCon = a.numConstraints() + b.numConstraints();
  Simplex result(numVar);

  result.tableau.resize(numCon, result.tableau.getNumColumns());
  result.nRedundant = a.nRedundant + b.nRedundant;
  result.liveColBegin = a.liveColBegin + b.liveColBegin - 2;
  result.empty = a.empty || b.empty;

  auto concat = [](ArrayRef<Unknown> v, ArrayRef<Unknown> w) {
    SmallVector<Unknown, 8> result;
    result.reserve(v.size() + w.size());
    result.insert(result.end(), v.begin(), v.end());
    result.insert(result.end(), w.begin(), w.end());
    return result;
  };
  result.con = concat(a.con, b.con);
  result.var = concat(a.var, b.var);

  auto indexFromBIndex = [&](int index) {
    return index >= 0 ? a.numVariables() + index
                      : ~(a.numConstraints() + ~index);
  };

  result.colUnknown.assign(2, nullIndex);
  for (unsigned i = 2; i < a.liveColBegin; ++i)
    result.colUnknown.push_back(a.colUnknown[i]);
  for (unsigned i = 2; i < b.liveColBegin; ++i) {
    result.colUnknown.push_back(indexFromBIndex(b.colUnknown[i]));
    result.unknownFromIndex(result.colUnknown.back()).pos =
        result.colUnknown.size() - 1;
  }
  for (unsigned i = a.liveColBegin; i < a.nCol; ++i) {
    result.colUnknown.push_back(a.colUnknown[i]);
    result.unknownFromIndex(result.colUnknown.back()).pos =
        result.colUnknown.size() - 1;
  }
  for (unsigned i = b.liveColBegin; i < b.nCol; ++i) {
    result.colUnknown.push_back(indexFromBIndex(b.colUnknown[i]));
    result.unknownFromIndex(result.colUnknown.back()).pos =
        result.colUnknown.size() - 1;
  }

  // TODO check if this is correct
  auto appendRowFromA = [&](unsigned row) {
    for (unsigned col = 0; col < result.liveColBegin; ++col)
      result.tableau(result.nRow, col) = a.tableau(row, col);
    unsigned offset = b.liveColBegin - 2;
    for (unsigned col = result.liveColBegin; col < a.nCol; ++col)
      result.tableau(result.nRow, offset + col) = a.tableau(row, col);
    result.rowUnknown.push_back(a.rowUnknown[row]);
    result.unknownFromIndex(result.rowUnknown.back()).pos =
        result.rowUnknown.size() - 1;
    result.nRow++;
  };

  // Also fixes the corresponding entry in rowUnknown and var/con (as the case
  // may be).
  auto appendRowFromB = [&](unsigned row) {
    result.tableau(result.nRow, 0) = b.tableau(row, 0);
    result.tableau(result.nRow, 1) = b.tableau(row, 1);
    unsigned offset = a.liveColBegin - 2;
    for (unsigned col = 2; col < b.liveColBegin; ++col)
      result.tableau(result.nRow, offset + col) = b.tableau(row, col);

    offset = a.nCol - 2;
    for (unsigned col = b.liveColBegin; col < b.nCol; ++col)
      result.tableau(result.nRow, offset + col) = b.tableau(row, col);
    result.rowUnknown.push_back(indexFromBIndex(b.rowUnknown[row]));
    result.unknownFromIndex(result.rowUnknown.back()).pos =
        result.rowUnknown.size() - 1;
    result.nRow++;
  };

  for (unsigned row = 0; row < a.nRedundant; ++row)
    appendRowFromA(row);
  for (unsigned row = 0; row < b.nRedundant; ++row)
    appendRowFromB(row);
  for (unsigned row = a.nRedundant; row < a.nRow; ++row)
    appendRowFromA(row);
  for (unsigned row = b.nRedundant; row < b.nRow; ++row)
    appendRowFromB(row);

  return result;
}

SmallVector<Fraction, 8> Simplex::getSamplePoint() const {
  // The tableau is empty, so no sample point exists.
  assert(!empty && "Simplex should not be empty!");

  SmallVector<Fraction, 8> sample;
  // Push the sample value for each variable into the vector.
  for (const Unknown &u : var) {
    if (u.orientation == Orientation::Column) {
      // If the variable is in column position, its sample value is zero.
      sample.emplace_back(0, 1);
    } else {
      // If the variable is in row position, its sample value is the entry in
      // the constant column divided by the entry in the common denominator
      // column.
      sample.emplace_back(tableau(u.pos, 1), tableau(u.pos, 0));
    }
  }
  return sample;
}

void Simplex::addFlatAffineConstraints(const FlatAffineConstraints &cs) {
  assert(cs.getNumIds() == numVariables() &&
         "FlatAffineConstraints must have same dimensionality as simplex");
  for (unsigned i = 0; i < cs.getNumInequalities(); ++i)
    addInequality(cs.getInequality(i));
  for (unsigned i = 0; i < cs.getNumEqualities(); ++i)
    addEquality(cs.getEquality(i));
}

Optional<SmallVector<int64_t, 8>> Simplex::getSamplePointIfIntegral() const {
  // The tableau is empty, so no sample point exists.
  if (empty)
    return {};

  SmallVector<Fraction, 8> sample = getSamplePoint();

  SmallVector<int64_t, 8> integralSample;
  for (Fraction f : sample) {
    if (f.num % f.den != 0)
      return {};
    integralSample.push_back(f.num / f.den);
  }
  return integralSample;
}

namespace mlir {
namespace analysis {
namespace presburger {

/// Given a simplex for a polytope, construct a new simplex whose variables
/// are identified with a pair of points (x, y) in the original polytope.
/// Supports some operations needed for generalized basis reduction. In what
/// follows, dotProduct(x, y) = x_1 * y_1 + x_2 * y_2 + ... x_n * y_n where n
/// is the dimension of the original polytope.
///
/// This supports adding equality constraints dotProduct(dir, x - y) == 0. It
/// also supports rolling back this addition, by maintaining a snapshot stack
/// that contains a snapshot of the Simplex's state for each equality, just
/// before that equality was added.
class GBRSimplex {
  using Orientation = Simplex::Orientation;

public:
  GBRSimplex(const Simplex &originalSimplex)
      : simplex(Simplex::makeProduct(originalSimplex, originalSimplex)),
        simplexConstraintOffset(simplex.numConstraints()) {}

  /// Add an equality dotProduct(dir, x - y) == 0.
  /// First pushes a snapshot for the current simplex state to the stack so
  /// that this can be rolled back later.
  void addEqualityForDirection(ArrayRef<int64_t> dir) {
    assert(
        std::any_of(dir.begin(), dir.end(), [](int64_t x) { return x != 0; }) &&
        "Direction passed is the zero vector!");
    snapshotStack.push_back(simplex.getSnapshot());
    simplex.addEquality(getCoeffsForDirection(dir));
  }

  /// Compute max(dotProduct(dir, x - y)) and save the dual variables for only
  /// the direction equalities to `dual`.
  Fraction computeWidthAndDuals(ArrayRef<int64_t> dir,
                                SmallVectorImpl<int64_t> &dual,
                                int64_t &dualDenom) {
    unsigned snap = simplex.getSnapshot();
    unsigned conIndex = simplex.addRow(getCoeffsForDirection(dir));
    unsigned row = simplex.con[conIndex].pos;
    Optional<Fraction> maybeWidth =
        simplex.computeRowOptimum(Simplex::Direction::Up, row);
    assert(maybeWidth.hasValue() && "Width should not be unbounded!");
    dualDenom = simplex.tableau(row, 0);
    dual.clear();
    // The increment is i += 2 because equalities are added as two
    // inequalities, one positive and one negative. Each iteration processes
    // one equality.
    for (unsigned i = simplexConstraintOffset; i < conIndex; i += 2) {
      // The dual variable is the negative of the coefficient of the new row
      // in the column of the constraint, if the constraint is in a column.
      // Note that the second inequality for the equality is negated.
      //
      // We want the dual for the original equality. If the positive
      // inequality is in column position, the negative of its row coefficient
      // is the desired dual. If the negative inequality is in column
      // position, its row coefficient is the desired dual. (its coefficients
      // are already the negated coefficients of the original equality, so we
      // don't need to negate it now.)
      //
      // If neither are in column position, we move the negated inequality to
      // column position. Since the inequality must have sample value zero
      // (since it corresponds to an equality), we are free to pivot with
      // any column. Since both the unknowns have sample value before and
      // after pivoting, no other sample values will change and the tableau
      // will remain consistent. To pivot, we just need to find a column that
      // has a non-zero coefficient in this row. There must be one since
      // otherwise the equality would be 0 == 0, which should never be passed
      // to addEqualityForDirection.
      //
      // After finding a column, we pivot with the column, after which we can
      // get the dual from the inequality in column position as explained
      // above.
      if (simplex.con[i].orientation == Orientation::Column) {
        dual.push_back(-simplex.tableau(row, simplex.con[i].pos));
      } else {
        if (simplex.con[i + 1].orientation == Orientation::Row) {
          unsigned ineqRow = simplex.con[i + 1].pos;
          // Since it is an equality, the the sample value must be zero.
          assert(simplex.tableau(ineqRow, 1) == 0 &&
                 "Equality's sample value must be zero.");
          for (unsigned col = 2; col < simplex.nCol; ++col) {
            if (simplex.tableau(ineqRow, col) != 0) {
              simplex.pivot(ineqRow, col);
              break;
            }
          }
          assert(simplex.con[i + 1].orientation == Orientation::Column &&
                 "No pivot found. Equality has all-zeros row in tableau!");
        }
        dual.push_back(simplex.tableau(row, simplex.con[i + 1].pos));
      }
    }
    simplex.rollback(snap);
    return *maybeWidth;
  }

  /// Remove the last equality that was added through addEqualityForDirection.
  ///
  /// We do this by rolling back to the snapshot at the top of the stack,
  /// which should be a snapshot taken just before the last equality was
  /// added.
  void removeLastEquality() {
    assert(!snapshotStack.empty() && "Snapshot stack is empty!");
    simplex.rollback(snapshotStack.back());
    snapshotStack.pop_back();
  }

private:
  /// Returns coefficients of the expression 'dot_product(dir, x - y)',
  /// i.e.,   dir_1 * x_1 + dir_2 * x_2 + ... + dir_n * x_n
  ///       - dir_1 * y_1 - dir_2 * y_2 - ... - dir_n * y_n,
  /// where n is the dimension of the original polytope.
  SmallVector<int64_t, 8> getCoeffsForDirection(ArrayRef<int64_t> dir) {
    assert(2 * dir.size() == simplex.numVariables() &&
           "Direction vector has wrong dimensionality");
    SmallVector<int64_t, 8> coeffs(dir.begin(), dir.end());
    coeffs.reserve(2 * dir.size());
    for (int64_t coeff : dir)
      coeffs.push_back(-coeff);
    coeffs.push_back(0); // constant term
    return coeffs;
  }

  Simplex simplex;
  /// The first index of the equality constraints, the index immediately after
  /// the last constraint in the initial product simplex.
  unsigned simplexConstraintOffset;
  /// A stack of snapshots, used for rolling back.
  SmallVector<unsigned, 8> snapshotStack;
};
} // namespace presburger
} // namespace analysis
} // namespace mlir

/// Reduce the basis to try and find a direction in which the polytope is
/// "thin". This only works for bounded polytopes.
///
/// This is an implementation of the algorithm described in the paper
/// "An Implementation of Generalized Basis Reduction for Integer Programming"
/// by W. Cook, T. Rutherford, H. E. Scarf, D. Shallcross.
///
/// Let b_{level}, b_{level + 1}, ... b_n be the current basis.
/// Let width_i(v) = max <v, x - y> where x and y are points in the original
/// polytope such that <b_j, x - y> = 0 is satisfied for all level <= j < i.
///
/// In every iteration, we first replace b_{i+1} with b_{i+1} + u*b_i, where u
/// is the integer such that width_i(b_{i+1} + u*b_i) is minimized. Let dual_i
/// be the dual variable associated with the constraint <b_i, x - y> = 0 when
/// computing width_{i+1}(b_{i+1}). It can be shown that dual_i is the
/// minimizing value of u, if it were allowed to be fractional. Due to
/// convexity, the minimizing integer value is either floor(dual_i) or
/// ceil(dual_i), so we just need to check which of these gives a lower
/// width_{i+1} value. If dual_i turned out to be an integer, then u = dual_i.
///
/// Now if width_i(b_{i+1}) < 0.75 * width_i(b_i), we swap b_i and (the new)
/// b_{i + 1} and decrement i (unless i = level, in which case we stay at the
/// same i). Otherwise, we increment i.
///
/// We keep f values and duals cached and invalidate them when necessary.
/// Whenever possible, we use them instead of recomputing them. We implement
/// the algorithm as follows.
///
/// In an iteration at i we need to compute:
///   a) width_i(b_{i + 1})
///   b) width_i(b_i)
///   c) the integer u that minimizes width_i(b_{i + 1} + u*b_i)
///
/// If width_i(b_i) is not already cached, we compute it.
///
/// If the duals are not already cached, we compute width_{i+1}(b_{i+1}) and
/// store the duals from this computation.
///
/// We call updateBasisWithUAndGetFCandidate, which finds the minimizing value
/// of u as explained before, caches the duals from this computation, sets
/// b_{i+1} to b_{i+1} + u*b_i, and returns the new value of width_i(b_{i+1}).
///
/// Now if width_i(b_{i+1}) < 0.75 * width_i(b_i), we swap b_i and b_{i+1} and
/// decrement i, resulting in the basis
/// ... b_{i - 1}, b_{i + 1} + u*b_i, b_i, b_{i+2}, ...
/// with corresponding f values
/// ... width_{i-1}(b_{i-1}), width_i(b_{i+1} + u*b_i), width_{i+1}(b_i), ...
/// The values up to i - 1 remain unchanged. We have just gotten the middle
/// value from updateBasisWithUAndGetFCandidate, so we can update that in the
/// cache. The value at width_{i+1}(b_i) is unknown, so we evict this value
/// from the cache. The iteration after decrementing needs exactly the duals
/// from the computation of width_i(b_{i + 1} + u*b_i), so we keep these in
/// the cache.
///
/// When incrementing i, no cached f values get invalidated. However, the
/// cached duals do get invalidated as the duals for the higher levels are
/// different.
void Simplex::reduceBasis(Matrix &basis, unsigned level) {
  const Fraction epsilon(3, 4);

  if (level == basis.getNumRows() - 1)
    return;

  GBRSimplex gbrSimplex(*this);
  SmallVector<Fraction, 8> width;
  SmallVector<int64_t, 8> dual;
  int64_t dualDenom;

  // Finds the value of u that minimizes width_i(b_{i+1} + u*b_i), caches the
  // duals from this computation, sets b_{i+1} to b_{i+1} + u*b_i, and returns
  // the new value of width_i(b_{i+1}).
  //
  // If dual_i is not an integer, the minimizing value must be either
  // floor(dual_i) or ceil(dual_i). We compute the expression for both and
  // choose the minimizing value.
  //
  // If dual_i is an integer, we don't need to perform these computations. We
  // know that in this case,
  //   a) u = dual_i.
  //   b) one can show that dual_j for j < i are the same duals we would have
  //      gotten from computing width_i(b_{i + 1} + u*b_i), so the correct
  //      duals are the ones already in the cache.
  //   c) width_i(b_{i+1} + u*b_i) = min_{alpha} width_i(b_{i+1} + alpha *
  //   b_i), which
  //      one can show is equal to width_{i+1}(b_{i+1}). The latter value must
  //      be in the cache, so we get it from there and return it.
  auto updateBasisWithUAndGetFCandidate = [&](unsigned i) -> Fraction {
    assert(i < level + dual.size() && "dual_i is not known!");

    int64_t u = floorDiv(dual[i - level], dualDenom);
    basis.addToRow(i, i + 1, u);
    if (dual[i - level] % dualDenom != 0) {
      SmallVector<int64_t, 8> candidateDual[2];
      int64_t candidateDualDenom[2];
      Fraction widthI[2];

      // Initially u is floor(dual) and basis reflects this.
      widthI[0] = gbrSimplex.computeWidthAndDuals(
          basis.getRow(i + 1), candidateDual[0], candidateDualDenom[0]);

      // Now try ceil(dual), i.e. floor(dual) + 1.
      ++u;
      basis.addToRow(i, i + 1, 1);
      widthI[1] = gbrSimplex.computeWidthAndDuals(
          basis.getRow(i + 1), candidateDual[1], candidateDualDenom[1]);

      unsigned j = widthI[0] < widthI[1] ? 0 : 1;
      if (j == 0)
        // Subtract 1 to go from u = ceil(dual) back to floor(dual).
        basis.addToRow(i, i + 1, -1);
      dual = std::move(candidateDual[j]);
      dualDenom = candidateDualDenom[j];
      return widthI[j];
    }
    assert(i + 1 - level < width.size() && "width_{i+1} wasn't saved");
    // When dual minimizes f_i(b_{i+1} + dual*b_i), this is equal to
    // width_{i+1}(b_{i+1}).
    return width[i + 1 - level];
  };

  // In the ith iteration of the loop, gbrSimplex has constraints for
  // directions from `level` to i - 1.
  unsigned i = level;
  while (i < basis.getNumRows() - 1) {
    if (i >= level + width.size()) {
      // We don't even know the value of f_i(b_i), so let's find that first.
      // We have to do this first since later we assume that width already
      // contains values up to and including i.

      assert((i == 0 || i - 1 < level + width.size()) &&
             "We are at level i but we don't know the value of width_{i-1}");

      // We don't actually use these duals at all, but it doesn't matter
      // because this case should only occur when i is level, and there are no
      // duals in that case anyway.
      assert(i == level && "This case should only occur when i == level");
      width.push_back(
          gbrSimplex.computeWidthAndDuals(basis.getRow(i), dual, dualDenom));
    }

    if (i >= level + dual.size()) {
      assert(i + 1 >= level + width.size() &&
             "We don't know dual_i but we know width_{i+1}");
      // We don't know dual for our level, so let's find it.
      gbrSimplex.addEqualityForDirection(basis.getRow(i));
      width.push_back(gbrSimplex.computeWidthAndDuals(basis.getRow(i + 1), dual,
                                                      dualDenom));
      gbrSimplex.removeLastEquality();
    }

    // This variable stores width_i(b_{i+1} + u*b_i).
    Fraction widthICandidate = updateBasisWithUAndGetFCandidate(i);
    if (widthICandidate < epsilon * width[i - level]) {
      basis.swapRows(i, i + 1);
      width[i - level] = widthICandidate;
      // The values of width_{i+1}(b_{i+1}) and higher may change after the
      // swap, so we remove the cached values here.
      width.resize(i - level + 1);
      if (i == level) {
        dual.clear();
        continue;
      }

      gbrSimplex.removeLastEquality();
      i--;
      continue;
    }

    // Invalidate duals since the higher level needs to recompute its own
    // duals.
    dual.clear();
    gbrSimplex.addEqualityForDirection(basis.getRow(i));
    i++;
  }
}

/// Search for an integer sample point using a branch and bound algorithm.
///
/// Each row in the basis matrix is a vector, and the set of basis vectors
/// should span the space. Initially this is the identity matrix,
/// i.e., the basis vectors are just the variables.
///
/// In every level, a value is assigned to the level-th basis vector, as
/// follows. Compute the minimum and maximum rational values of this
/// direction. If only one integer point lies in this range, constrain the
/// variable to have this value and recurse to the next variable.
///
/// If the range has multiple values, perform generalized basis reduction via
/// reduceBasis and then compute the bounds again. Now we try constraining
/// this direction in the first value in this range and "recurse" to the next
/// level. If we fail to find a sample, we try assigning the direction the
/// next value in this range, and so on.
///
/// If no integer sample is found from any of the assignments, or if the range
/// contains no integer value, then of course the polytope is empty for the
/// current assignment of the values in previous levels, so we return to
/// the previous level.
///
/// If we reach the last level where all the variables have been assigned
/// values already, then we simply return the current sample point if it is
/// integral, and go back to the previous level otherwise.
///
/// To avoid potentially arbitrarily large recursion depths leading to stack
/// overflows, this algorithm is implemented iteratively.
Optional<SmallVector<int64_t, 8>> Simplex::findIntegerSample() {
  if (empty)
    return {};

  unsigned nDims = var.size();
  Matrix basis = Matrix::identity(nDims);

  unsigned level = 0;
  // The snapshot just before constraining a direction to a value at each
  // level.
  SmallVector<unsigned, 8> snapshotStack;
  // The maximum value in the range of the direction for each level.
  SmallVector<int64_t, 8> upperBoundStack;
  // The next value to try constraining the basis vector to at each level.
  SmallVector<int64_t, 8> nextValueStack;

  snapshotStack.reserve(basis.getNumRows());
  upperBoundStack.reserve(basis.getNumRows());
  nextValueStack.reserve(basis.getNumRows());
  while (level != -1u) {
    if (level == basis.getNumRows()) {
      // We've assigned values to all variables. Return if we have a sample,
      // or go back up to the previous level otherwise.
      if (auto maybeSample = getSamplePointIfIntegral())
        return maybeSample;
      level--;
      continue;
    }

    if (level >= upperBoundStack.size()) {
      // We haven't populated the stack values for this level yet, so we have
      // just come down a level ("recursed"). Find the lower and upper bounds.
      // If there is more than one integer point in the range, perform
      // generalized basis reduction.
      SmallVector<int64_t, 8> basisCoeffs =
          llvm::to_vector<8>(basis.getRow(level));
      basisCoeffs.push_back(0);

      int64_t minRoundedUp, maxRoundedDown;
      std::tie(minRoundedUp, maxRoundedDown) =
          computeIntegerBounds(basisCoeffs);

      // Heuristic: if the sample point is integral at this point, just return
      // it.
      if (auto maybeSample = getSamplePointIfIntegral())
        return *maybeSample;

      if (minRoundedUp < maxRoundedDown) {
        reduceBasis(basis, level);
        basisCoeffs = llvm::to_vector<8>(basis.getRow(level));
        basisCoeffs.push_back(0);
        std::tie(minRoundedUp, maxRoundedDown) =
            computeIntegerBounds(basisCoeffs);
      }

      snapshotStack.push_back(getSnapshot());
      // The smallest value in the range is the next value to try.
      nextValueStack.push_back(minRoundedUp);
      upperBoundStack.push_back(maxRoundedDown);
    }

    assert((snapshotStack.size() - 1 == level &&
            nextValueStack.size() - 1 == level &&
            upperBoundStack.size() - 1 == level) &&
           "Mismatched variable stack sizes!");

    // Whether we "recursed" or "returned" from a lower level, we rollback
    // to the snapshot of the starting state at this level. (in the "recursed"
    // case this has no effect)
    rollback(snapshotStack.back());
    int64_t nextValue = nextValueStack.back();
    nextValueStack.back()++;
    if (nextValue > upperBoundStack.back()) {
      // We have exhausted the range and found no solution. Pop the stack and
      // return up a level.
      snapshotStack.pop_back();
      nextValueStack.pop_back();
      upperBoundStack.pop_back();
      level--;
      continue;
    }

    // Try the next value in the range and "recurse" into the next level.
    SmallVector<int64_t, 8> basisCoeffs(basis.getRow(level).begin(),
                                        basis.getRow(level).end());
    basisCoeffs.push_back(-nextValue);
    addEquality(basisCoeffs);
    level++;
  }

  return {};
}

std::pair<int64_t, SmallVector<int64_t, 8>>
Simplex::findRationalSample() const {
  int64_t denom = 1;
  for (const Unknown &u : var) {
    if (u.orientation == Orientation::Row)
      denom = lcm(denom, tableau(u.pos, 0));
  }
  SmallVector<int64_t, 8> sample;
  int64_t gcd = denom;
  for (const Unknown &u : var) {
    if (u.orientation == Orientation::Column)
      sample.push_back(0);
    else {
      sample.push_back((tableau(u.pos, 1) * denom) / tableau(u.pos, 0));
      gcd = llvm::greatestCommonDivisor(std::abs(gcd), std::abs(sample.back()));
    }
  }
  if (gcd != 0) {
    denom /= gcd;
    for (int64_t &elem : sample)
      elem /= gcd;
  }

  return {denom, std::move(sample)};
}

/// Compute the minimum and maximum integer values the expression can take. We
/// compute each separately.
std::pair<int64_t, int64_t>
Simplex::computeIntegerBounds(ArrayRef<int64_t> coeffs) {
  int64_t minRoundedUp;
  if (Optional<Fraction> maybeMin =
          computeOptimum(Simplex::Direction::Down, coeffs))
    minRoundedUp = ceil(*maybeMin);
  else
    llvm_unreachable("Tableau should not be unbounded");

  int64_t maxRoundedDown;
  if (Optional<Fraction> maybeMax =
          computeOptimum(Simplex::Direction::Up, coeffs))
    maxRoundedDown = floor(*maybeMax);
  else
    llvm_unreachable("Tableau should not be unbounded");

  return {minRoundedUp, maxRoundedDown};
}

/// The minimum of an unknown is obviously unbounded if it is a column variable
/// and no constraint limits its value from below.
///
/// The minimum of a row variable is not obvious because it depends on the
/// boundedness of all referenced column variables.
///
/// A column variable is bounded from below if there a exists a constraint for
/// which the corresponding column coefficient is strictly positive and the row
/// variable is non-negative (restricted).
inline bool Simplex::minIsObviouslyUnbounded(Unknown &unknown) const {
  // tableau.checkSparsity();
  if (unknown.orientation == Orientation::Row)
    return false;

  for (size_t i = nRedundant; i < nRow; i++) {
    if (unknownFromRow(i).restricted && tableau(i, unknown.pos) > 0)
      return false;
  }
  return true;
}

/// The maximum of an unknown is obviously unbounded if it is a column variable
/// and no constraint limits its value from above.
///
/// The maximum of a row variable is not obvious because it depends on the
/// boundedness of all referenced column variables.
///
/// A column variable is surely unbounded from above if there does not exist a
/// constraint for which the corresponding column coefficient is strictly
/// negative and the row variable is non-negative (restricted).
inline bool Simplex::maxIsObviouslyUnbounded(Unknown &unknown) const {
  if (unknown.orientation == Orientation::Row)
    return false;

  for (unsigned row = nRedundant; row < nRow; row++) {
    if (tableau(row, unknown.pos) < 0 && unknownFromRow(row).restricted)
      return false;
  }
  return true;
}

/// A row is obviously not constrained to be zero if
/// - the tableau is rational and the constant term is not zero
/// - the tableau is integer and the constant term is at least one (it is also
///   not zero if the constant term is at most negative one, but this is only
///   called for restricted rows, so it doesn't cost anything to be imprecise)
///
/// This is because of the invariant that the sample value is always a valid
/// point in the tableau (assuming the tableau is not empty).
inline bool Simplex::rowIsObviouslyNotZero(unsigned row) const {
  return tableau(row, 1) >= tableau(row, 0);
}

/// A row is equal to zero if its constant term and all coefficients for live
/// columns are equal to zero.
inline bool Simplex::rowIsObviouslyZero(unsigned row) const {
  if (tableau(row, 1) != 0)
    return false;
  for (unsigned col = liveColBegin; col < nCol; col++) {
    if (tableau(row, col) != 0)
      return false;
  }
  return true;
}

/// An unknown is considered to be relevant if it is neither a redundant row nor
/// a dead column
inline bool Simplex::unknownIsRelevant(Unknown &unknown) const {
  if (unknown.orientation == Orientation::Row && unknown.pos < nRedundant)
    return false;
  else if (unknown.orientation == Orientation::Column &&
           unknown.pos < liveColBegin)
    return false;
  return true;
}

/// A row is obviously non integral if all of its non-dead column entries are
/// zero and the constant term denominator is not divisible by the row
/// denominator.
///
/// If there are non-zero entries then integrality depends on the values of all
/// referenced column variables.
inline bool Simplex::rowIsObviouslyNonIntegral(unsigned row) const {
  for (unsigned j = liveColBegin; j < nCol; j++) {
    if (tableau(row, j) != 0)
      return false;
  }
  return tableau(row, 1) % tableau(row, 0) != 0;
}

/// Pivot the unknown to row position in the specified direction. If no
/// direction is provided, both directions are allowed. The unknown is assumed
/// to be bounded in the specified direction. If no direction is specified, the
/// unknown is assumed to be unbounded in both directions.
///
/// If the unknown is already in row position we need not do anything.
/// Otherwise, we find a row to pivot to via findPivotRow and pivot to it.
/// TODO currently doesn't support an optional direction
inline void Simplex::toRow(Unknown &unknown, Direction direction) {
  if (unknown.orientation == Orientation::Row)
    return;

  auto row = findPivotRow({}, direction, unknown.pos);
  if (row)
    pivot(*row, unknown.pos);
  else
    llvm_unreachable("No pivot row found. The unknown must be bounded in the"
                     "specified directions.");
}

inline int64_t Simplex::sign(int64_t num, int64_t den, int64_t origin) const {
  if (num > origin * den)
    return +1;
  else if (num < origin * den)
    return -1;
  else
    return 0;
}

/// Compare the maximum value of the unknown to origin. Return +1 if the maximum
/// value is greater than origin, 0 if they are equal, and -1 if it is less
/// than origin.
///
/// If the unknown is marked zero, then its maximum value is zero and we can
/// return accordingly.
///
/// If the maximum is obviously unbounded, we can return 1.
///
/// Otherwise, we move the unknown up to a row and keep trying to find upward
/// pivots until the unknown becomes unbounded or becomes maximised.
template <int origin>
int64_t Simplex::signOfMax(Unknown &u) {
  static_assert(origin >= 0, "");
  assert(!u.redundant && "signOfMax called for redundant unknown");

  if (maxIsObviouslyUnbounded(u))
    return 1;

  toRow(u, Direction::Up);

  // The only callsite where origin == 1 only cares whether it's negative or
  // not, so we can save some pivots by quitting early in this case. This is
  // needed for pivot-parity with isl.
  static_assert(origin == 0 || origin == 1, "");
  auto mustContinueLoop = [this](unsigned row) {
    return origin == 1 ? tableau(row, 1) < tableau(row, 0)
                       : tableau(row, 1) <= 0;
  };

  while (mustContinueLoop(u.pos)) {
    auto p = findPivot(u.pos, Direction::Up);
    if (!p) {
      // u is manifestly maximised
      return sign(tableau(u.pos, 1) - origin * tableau(u.pos, 0));
    } else if (p->row == u.pos) { // u is manifestly unbounded
      // In isl, this pivot is performed only when origin == 0 but not when it's
      // 1. This is because the two are different functions with very slightly
      // differing implementations. For pivot-parity with is, we do this too.
      if (origin == 0)
        pivot(*p);
      return 1;
    }
    pivot(*p);
  }
  return 1;
}

inline void Simplex::swapColumns(unsigned i, unsigned j) {
  tableau.swapColumns(i, j);
  std::swap(colUnknown[i], colUnknown[j]);
  unknownFromColumn(i).pos = i;
  unknownFromColumn(j).pos = j;
}

/// Remove the row from the tableau.
///
/// If the row is not already the last one, swap it with the last row.
/// Then decrement the row count, remove the constraint entry, and remove the
/// entry in row_var.
inline void Simplex::dropRow(unsigned row) {
  // It is unclear why this restriction exists. Perhaps because bmaps outside
  // keep track of the number of equalities, and hence moving around constraints
  // would require updating them?
  assert(~rowUnknown[row] == int(con.size() - 1) &&
         "Row to be dropped must be the last constraint");

  if (row != nRow - 1)
    swapRows(row, nRow - 1);
  nRow--;
  rowUnknown.pop_back();
  con.pop_back();
}

inline bool Simplex::killCol(unsigned col) {
  Unknown &unknown = unknownFromColumn(col);
  unknown.zero = true;
  undoLog.emplace_back(UndoLogEntry::UnmarkZero, colUnknown[col]);
  if (col != liveColBegin)
    swapColumns(col, liveColBegin);
  liveColBegin++;
  // tableau.checkSparsity();
  return false;
}

inline int Simplex::indexFromUnknown(const Unknown &u) const {
  if (u.orientation == Orientation::Row)
    return rowUnknown[u.pos];
  return colUnknown[u.pos];
}

/// If temp_row is set, the last row is a temporary unknown and undo entries
/// should not be written for this row.
inline void Simplex::closeRow(unsigned row, bool tempRow) {
  Unknown *u = &unknownFromRow(row);
  assert(u->restricted && "expected restricted variable\n");

  if (!u->zero && !tempRow) {
    // pushUndoEntryIfNeeded(UndoOp::UNMARK_ZERO, *u);
    undoLog.emplace_back(UndoLogEntry::UnmarkZero, indexFromUnknown(*u));
  }
  u->zero = true;

  for (unsigned col = liveColBegin; col < nCol; col++) {
    if (tableau(u->pos, col) == 0)
      continue;
    assert(tableau(u->pos, col) <= 0 &&
           "expecting variable upper bounded by zero; "
           "row cannot have positive coefficients");
    if (killCol(col))
      col--;
  }
  if (!tempRow)
    markRedundant(u->pos);

  if (!empty)
    // Check if there are any vars that can't attain integral values.
    for (const Unknown &u : var)
      if (u.orientation == Orientation::Row &&
          rowIsObviouslyNonIntegral(u.pos)) {
        markEmpty();
        break;
      }
}

inline void Simplex::extendConstraints(unsigned nNew) {
  if (con.capacity() < con.size() + nNew)
    con.reserve(con.size() + nNew);
  if (tableau.getNumRows() < nRow + nNew) {
    tableau.resizeVertically(nRow + nNew);
    rowUnknown.reserve(nRow + nNew);
  }
}

/// Given a constraint con >= 0, add another constraint -con >= 0 so that we cut
/// our polytope to the hyperplane con = 0. Before the function returns the
/// added constraint is removed again, but the effects on the other unknowns
/// remain.
///
/// If the constraint is in row position, add a new row with all the terms
/// negated. If the constrain is in column position, add a row with a single
/// -1 coefficient for that column and all zeroes in other columns.
///
/// If our new constraint cannot achieve non-negative values, then the tableau
/// is empty and we marked it as such. Otherwise, we know that its value must
/// always be zero so we close the row and then drop it. Closing the row results
/// in some columns being killed, so the effect of fixing it to be zero remains
/// even after it is dropped.
inline void Simplex::cutToHyperplane(int conIndex) {
  if (con[conIndex].zero)
    return;
  assert(!con[conIndex].redundant && con[conIndex].restricted &&
         "expecting non-redundant non-negative variable");

  extendConstraints(1);
  rowUnknown.push_back(~con.size());
  con.emplace_back(Orientation::Row, false, nRow);
  Unknown &unknown = con[conIndex];
  Unknown &tempVar = con.back();

  if (unknown.orientation == Orientation::Row) {
    tableau(nRow, 0) = tableau(unknown.pos, 0);
    for (unsigned col = 1; col < nCol; col++)
      tableau(nRow, col) = -tableau(unknown.pos, col);
  } else {
    tableau(nRow, 0) = 1;
    for (unsigned col = 1; col < nCol; col++)
      tableau(nRow, col) = 0;
    tableau(nRow, unknown.pos) = -1;
  }
  // tableau.updateRowSparsity(nRow);
  nRow++;

  int64_t sgn = signOfMax<0>(tempVar);
  if (sgn < 0) {
    assert(tempVar.orientation == Orientation::Row &&
           "temp_var is in column position");
    dropRow(nRow - 1);
    markEmpty();
    return;
  }
  tempVar.restricted = true;
  assert(sgn <= 0 && "signOfMax is positive for the negated constraint");

  assert(tempVar.orientation == Orientation::Row &&
         "temp_var is in column position");
  if (tempVar.pos != nRow - 1)
    tableau.swapRows(tempVar.pos, nRow - 1);

  closeRow(tempVar.pos, true);
  dropRow(tempVar.pos);
}

/// Check for constraints that are constrained to be equal to zero. i.e. check
/// for non-negative unknowns whose maximal value is
/// - zero (for rational tableaus)
/// - strictly less than one (for integer tableaus)
/// Close unknowns with maximal value zero. In integer tableaus, if an unknown
/// has a maximal value less than one, add a constraint to fix it to zero.
///
/// We first mark unknowns which are restricted, not dead, not redundant, and
/// which aren't obviously able to reach non-zero values.
///
/// We iterate through the marked unknowns. If an unknown's maximal value (found
/// via signOfMax<0>) is zero, then it is an equality, so we close the row. (the
/// unknown is always in row position when signOfMax returns 0)
///
/// Otherwise, if we have an integer tableau, we check if the unknown's maximal
/// value is at least one. If not, then in an integer tableau it must be zero.
/// We add another constraint fixing it to be zero via cutToHyperplane. Since
/// this might have created new equalities, we call detectImplicitEqualities
/// again to find these.
///
/// TODO: consider changing the name, something with 'zero' is perhaps more
/// indicative than equality. And 'detect' sounds more passive than what we're
/// doing (adding constraints, closing rows).
void Simplex::detectImplicitEqualities() {
  if (empty)
    return;
  if (liveColBegin == nCol)
    return;

  for (size_t row = nRedundant; row < nRow; row++) {
    Unknown &unknown = unknownFromRow(row);
    unknown.marked =
        (unknown.restricted && !rowIsObviouslyNotZero(unknown.pos));
  }
  for (size_t col = liveColBegin; col < nCol; col++) {
    Unknown &unknown = unknownFromColumn(col);
    unknown.marked = unknown.restricted;
  }

  for (int i = con.size() - 1; i >= 0; i--) {
    if (!con[i].marked)
      continue;
    con[i].marked = false;
    if (!unknownIsRelevant(con[i]))
      continue;

    int64_t sgn = signOfMax<0>(con[i]);
    if (sgn == 0) {
      closeRow(con[i].pos, false);
    } else if (signOfMax<1>(con[i]) < 0) {
      cutToHyperplane(i);
      detectImplicitEqualities();
      return;
    }

    for (unsigned i = nRedundant; i < nRow; i++) {
      Unknown &unknown = unknownFromRow(i);
      if (unknown.marked && rowIsObviouslyNotZero(i))
        unknown.marked = false;
    }
  }
}

void Simplex::print(raw_ostream &os) const {
  os << "rows = " << nRow << ", columns = " << nCol
     << "\nnRedundant = " << nRedundant << "\n";
  if (empty)
    os << "Simplex marked empty!\n";
  os << "var: ";
  for (unsigned i = 0; i < var.size(); ++i) {
    if (i > 0)
      os << ", ";
    var[i].print(os);
  }
  os << "\ncon: ";
  for (unsigned i = 0; i < con.size(); ++i) {
    if (i > 0)
      os << ", ";
    con[i].print(os);
  }
  os << '\n';
  for (unsigned row = 0; row < nRow; ++row) {
    if (row > 0)
      os << ", ";
    os << "r" << row << ": " << rowUnknown[row];
  }
  os << '\n';
  os << "c0: denom, c1: const";
  for (unsigned col = 2; col < nCol; ++col)
    os << ", c" << col << ": " << colUnknown[col];
  os << '\n';
  for (unsigned row = 0; row < nRow; ++row) {
    for (unsigned col = 0; col < nCol; ++col)
      os << tableau(row, col) << '\t';
    os << '\n';
  }
  os << '\n';
}

void Simplex::dump() const { print(llvm::errs()); }

