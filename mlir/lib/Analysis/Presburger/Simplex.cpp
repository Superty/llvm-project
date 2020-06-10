//===- Simplex.cpp - MLIR Simplex Class -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Support/MathExtras.h"

#include <algorithm>
#include <cstdio>

namespace mlir {
const int NULL_INDEX = std::numeric_limits<int>::max();

// Construct a Simplex object with `nVar` variables.
Simplex::Simplex(unsigned nVar)
    : nRow(0), nCol(2), nRedundant(0), liveColBegin(2), tableau(0, 2 + nVar),
      empty(false) {
  colVar.push_back(NULL_INDEX);
  colVar.push_back(NULL_INDEX);
  for (unsigned i = 0; i < nVar; ++i) {
    var.push_back(
        Unknown(/*ownsRow=*/false, /*restricted=*/false, /*pos=*/nCol));
    colVar.push_back(i);
    nCol++;
  }
}

Simplex::Simplex(const FlatAffineConstraints &constraints)
    : Simplex(constraints.getNumIds()) {
  addFlatAffineConstraints(constraints);
}

const Simplex::Unknown &Simplex::unknownFromIndex(int index) const {
  assert(index != NULL_INDEX && "NULL_INDEX passed to unknownFromIndex");
  return index >= 0 ? var[index] : con[~index];
}

const Simplex::Unknown &Simplex::unknownFromColumn(unsigned col) const {
  assert(col < nCol && "Invalid column");
  return unknownFromIndex(colVar[col]);
}

const Simplex::Unknown &Simplex::unknownFromRow(unsigned row) const {
  assert(row < nRow && "Invalid row");
  return unknownFromIndex(rowVar[row]);
}

Simplex::Unknown &Simplex::unknownFromIndex(int index) {
  assert(index != NULL_INDEX && "NULL_INDEX passed to unknownFromIndex");
  return index >= 0 ? var[index] : con[~index];
}

Simplex::Unknown &Simplex::unknownFromColumn(unsigned col) {
  assert(col < nCol && "Invalid column");
  return unknownFromIndex(colVar[col]);
}

Simplex::Unknown &Simplex::unknownFromRow(unsigned row) {
  assert(row < nRow && "Invalid row");
  return unknownFromIndex(rowVar[row]);
}

// Add a new row to the tableau corresponding to the given constant term and
// list of coefficients. The coefficients are specified as a vector of
// (variable index, coefficient) pairs.
//
// If the tableau is not big enough to accomodate the extra row, we extend it.
//
// We then process each given variable coefficient.
// If a variable is in column position at column col, then we just add the
// coefficient for that variable (scaled by the common row denominator) to
// the corresponding entry in the new row.
// If the variable is in row position, we need to add that row to the new row,
// scaled by the coefficient for the variable, accounting for the two rows
// potentially having different denominators. The new denominator is the
// lcm of the two.
unsigned Simplex::addRow(ArrayRef<int64_t> coeffs) {
  assert(coeffs.size() == 1 + var.size() &&
         "Incorrect number of coefficients!");

  nRow++;
  if (nRow >= tableau.getNumRows())
    tableau.resize(nRow, tableau.getNumColumns());
  rowVar.push_back(~con.size());
  con.emplace_back(true, false, nRow - 1);

  tableau(nRow - 1, 0) = 1;
  tableau(nRow - 1, 1) = coeffs.back();
  for (unsigned col = liveColBegin; col < nCol; ++col)
    tableau(nRow - 1, col) = 0;

  for (unsigned i = 0; i < var.size(); ++i) {
    unsigned pos = var[i].pos;
    if (coeffs[i] == 0)
      continue;

    if (!var[i].ownsRow) {
      tableau(nRow - 1, pos) += coeffs[i] * tableau(nRow - 1, 0);
      continue;
    }

    int64_t lcm = mlir::lcm(tableau(nRow - 1, 0), tableau(pos, 0));
    int64_t nRowCoeff = lcm / tableau(nRow - 1, 0);
    int64_t idxRowCoeff = coeffs[i] * (lcm / tableau(pos, 0));
    tableau(nRow - 1, 0) = lcm;
    for (unsigned col = 1; col < nCol; ++col)
      tableau(nRow - 1, col) =
          nRowCoeff * tableau(nRow - 1, col) + idxRowCoeff * tableau(pos, col);
  }

  normalizeRow(nRow - 1);
  // Push to undo stack along with the index of the new constraint.
  undoLog.emplace_back(UndoOp::DEALLOCATE, ~(con.size() - 1));
  return con.size() - 1;
}

unsigned Simplex::getNumColumns() const { return nCol; }

unsigned Simplex::getNumRows() const { return nRow; }

// Normalize the row by removing common factors that are common between the
// denominator and all the numerator coefficients.
void Simplex::normalizeRow(unsigned row) {
  int64_t gcd = 0;
  for (unsigned col = 0; col < nCol; ++col) {
    if (gcd == 1)
      break;
    gcd = llvm::greatestCommonDivisor(gcd, std::abs(tableau(row, col)));
  }
  for (unsigned col = 0; col < nCol; ++col)
    tableau(row, col) /= gcd;
}

bool Simplex::signMatchesDirection(int64_t elem, Direction direction) const {
  assert(elem != 0 && "elem should not be 0");
  return direction == Direction::UP ? elem > 0 : elem < 0;
}

Simplex::Direction Simplex::flippedDirection(Direction direction) const {
  return direction == Direction::UP ? Direction::DOWN : Direction::UP;
}

// Find a pivot to change the sample value of `row` in the specified direction.
// The returned pivot row will involve `row` if and only if the unknown is
// unbounded in the specified direction.
//
// To increase (resp. decrease) the value of a row, we need to find a live
// column with a non-zero coefficient. If the coefficient is positive, we need
// to increase (decrease) the value of the column, and if the coefficient is
// negative, we need to decrease (increase) the value of the column. Also,
// we cannot decrease the sample value of restricted columns.
//
// If multiple columns are valid, we break ties by considering a lexicographic
// ordering where we prefer unknowns with lower index.
llvm::Optional<std::pair<unsigned, unsigned>>
Simplex::findPivot(int row, Direction direction) const {
  llvm::Optional<unsigned> col;
  for (unsigned j = liveColBegin; j < nCol; ++j) {
    int64_t elem = tableau(row, j);
    if (elem == 0)
      continue;

    if (unknownFromColumn(j).restricted &&
        !signMatchesDirection(elem, direction))
      continue;
    if (!col || colVar[j] < colVar[*col])
      col = j;
  }

  if (!col)
    return {};

  Direction newDirection =
      tableau(row, *col) < 0 ? flippedDirection(direction) : direction;
  auto opt = findPivotRow(row, newDirection, *col);
  return std::pair<unsigned, unsigned>{opt.getValueOr(row), *col};
}

void Simplex::swapRowWithCol(unsigned row, unsigned col) {
  std::swap(rowVar[row], colVar[col]);
  Unknown *uCol = &unknownFromColumn(col);
  Unknown *uRow = &unknownFromRow(row);
  uCol->ownsRow = false;
  uRow->ownsRow = true;
  uCol->pos = col;
  uRow->pos = row;
}

void Simplex::pivot(const std::pair<unsigned, unsigned> &p) {
  pivot(p.first, p.second);
}

// Pivot pivotRow and pivotCol.
//
// Let R be the pivot row unknown and let C be the pivot col unknown.
// Since initially R = a*C + sum b_i * X_i
// (where the sum is over the other column's unknowns, x_i)
// C = (R - (sum b_i * X_i))/a
//
// Let u be some other row unknown.
// u = c*C + sum d_i * X_i
// So u = c*(R - sum b_i * X_i)/a + sum d_i * X_i
//
// This results in the following transform:
//            pivot col    other col                   pivot col    other col
// pivot row     a             b       ->   pivot row     1/a         -b/a
// other row     c             d            other row     c/a        d - bc/a
//
// Taking into account the common denominators p and q:
//
//            pivot col    other col                    pivot col   other col
// pivot row     a/p          b/p     ->   pivot row      p/a         -b/a
// other row     c/q          d/q          other row     cp/aq    (da - bc)/aq
//
// The pivot row transform is accomplished be swapping a with the pivot row's
// common denominator and negating the pivot row except for the pivot column
// element.
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

// Perform pivots until the unknown has a non-negative sample value or until
// no more upward pivots can be performed. Return the sign of the final sample
// value.
bool Simplex::restoreRow(Unknown &u) {
  assert(u.ownsRow && "unknown should be in row position");

  while (tableau(u.pos, 1) < 0) {
    auto p = findPivot(u.pos, Direction::UP);
    if (!p)
      break;

    pivot(*p);
    if (!u.ownsRow)
      return 1; // the unknown is unbounded above.
  }
  return tableau(u.pos, 1) >= 0;
}

// Find a row that can be used to pivot the column in the specified direction.
// This returns an empty optional if and only if the column is unbounded in the
// specified direction (ignoring skipRow, if skipRow is set).
//
// If skipRow is set, this row is not considered, and (if it is restricted) its
// restriction may be violated by the returned pivot. Usually, skipRow is set
// because we don't want to move it to column position unless it is unbounded,
// and we are either trying to increase the value of skipRow or explicitly
// trying to make skipRow negative, so we are not concerned about this.
//
// If the direction is up (resp. down) and a restricted row has a negative
// (positive) coefficient for the column, then this row imposes a bound on how
// much the sample value of the column can change. Such a row with constant term
// c and coefficient f for the column imposes a bound of c/|f| on the change in
// sample value (in the specified direction). (note that c is non-negative here
// since the row is restricted and the tableau is consistent)
//
// We iterate through the rows and pick the row which imposes the most stringent
// bound, since pivoting with a row changes the row's sample value to 0 and
// hence saturates the bound it imposes. We break ties between rows that impose
// the same bound by considering a lexicographic ordering where we prefer
// unknowns with lower index value.
llvm::Optional<unsigned> Simplex::findPivotRow(llvm::Optional<unsigned> skipRow,
                                               Direction direction,
                                               unsigned col) const {
  llvm::Optional<unsigned> retRow;
  int64_t retElem, retConst;
  for (unsigned row = nRedundant; row < nRow; ++row) {
    if (skipRow && row == *skipRow)
      continue;
    auto elem = tableau(row, col);
    if (elem == 0)
      continue;
    if (!unknownFromRow(row).restricted)
      continue;
    if (signMatchesDirection(elem, direction))
      continue;
    auto constTerm = tableau(row, 1);

    if (!retRow) {
      retRow = row;
      retElem = elem;
      retConst = constTerm;
      continue;
    }

    auto diff = retConst * elem - constTerm * retElem;
    if ((diff == 0 && rowVar[row] < rowVar[*retRow]) ||
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
  std::swap(rowVar[i], rowVar[j]);
  unknownFromRow(i).pos = i;
  unknownFromRow(j).pos = j;
}

// Mark this tableau empty and push an entry to the undo stack.
void Simplex::markEmpty() {
  undoLog.emplace_back(UndoOp::UNMARK_EMPTY, llvm::Optional<int>());
  empty = true;
}

// Find out if the constraint is redundant by computing its minimum value in
// the tableau. If this returns true, the constraint is left in row position
// upon return.
//
// The constraint is redundant if the minimal value of the unknown (while
// respecting the other non-redundant constraints) is non-negative.
//
// If the unknown is in column position, we try to pivot it down to a row. If
// no pivot is found, this means that the constraint is unbounded below, i.e.
// it is not redundant, so we return false.
//
// Otherwise, the constraint is in row position. We keep trying to pivot
// downwards until the sample value becomes negative. If the next pivot would
// move the unknown to column position, then it is unbounded below and we can
// return false. If no more pivots are possible and the sample value is still
// non-negative, return true.
//
// Otherwise, if the unknown has a negative sample value, then it is
// not redundant, so we restore the row to a non-negative value and return.
bool Simplex::constraintIsRedundant(unsigned conIndex) {
  if (con[conIndex].redundant)
    return true;

  if (!con[conIndex].ownsRow) {
    unsigned col = con[conIndex].pos;
    auto maybeRow = findPivotRow({}, Direction::DOWN, col);
    if (!maybeRow)
      return false;
    pivot(*maybeRow, col);
  }

  while (tableau(con[conIndex].pos, 1) >= 0) {
    auto p = findPivot(con[conIndex].pos, Direction::DOWN);
    if (!p)
      return true;

    unsigned row = p->first;
    unsigned col = p->second;
    if (row == con[conIndex].pos)
      return false;
    pivot(row, col);
  }

  if (tableau(con[conIndex].pos, 1) >= 0)
    return true;

  bool success = restoreRow(con[conIndex]);
  assert(success && "Constraint was not restored succesfully!");
  return false;
}

bool Simplex::isMarkedRedundant(int conIndex) const {
  return con[conIndex].redundant;
}

// Check whether the constraint is an equality.
//
// The constraint is an equality if it has been marked zero, if it is a dead
// column, or if the row is obviously equal to zero.
bool Simplex::constraintIsEquality(int con_index) const {
  const Unknown &u = con[con_index];
  if (u.zero)
    return true;
  if (u.redundant)
    return false;
  if (!u.ownsRow)
    return u.pos < liveColBegin;
  return rowIsObviouslyZero(u.pos);
}

// Add an inequality to the tableau. If coeffs is c_0, c_1, ... c_n, where n
// is the curent number of variables, then the corresponding inequality is
// c_n + c_0*x_0 + c_1*x_1 + ... + c_{n-1}*x_{n-1} >= 0.
//
// We add the inequality and mark it as restricted. We then try to make its
// sample value non-negative. If this is not possible, the tableau has become
// empty and we mark it as such.
void Simplex::addInequality(ArrayRef<int64_t> coeffs) {
  unsigned conIndex = addRow(coeffs);
  Unknown &u = con[conIndex];
  u.restricted = true;
  bool success = restoreRow(u);
  if (!success)
    markEmpty();
}

// Add an equality to the tableau. If coeffs is c_0, c_1, ... c_n, where n
// is the curent number of variables, then the corresponding equality is
// c_n + c_0*x_0 + c_1*x_1 + ... + c_{n-1}*x_{n-1} == 0.
//
// We simply add two opposing inequalities, which force the expression to
// be zero.
void Simplex::addEquality(ArrayRef<int64_t> coeffs) {
  addInequality(coeffs);
  SmallVector<int64_t, 64> negatedCoeffs;
  for (auto coeff : coeffs)
    negatedCoeffs.emplace_back(-coeff);
  addInequality(negatedCoeffs);
}

// Mark the row as being redundant and push an entry to the undo stack.
//
// Since all the rows are stored contiguously as the first nRedundant rows,
// we move our row to the row at position nRedundant.
bool Simplex::markRedundant(unsigned row) {
  assert(!unknownFromRow(row).redundant && "Row is already marked redundant");
  assert(row >= nRedundant &&
         "Row is not marked redundant but row < nRedundant");

  undoLog.emplace_back(UndoOp::UNMARK_EMPTY, rowVar[row]);

  Unknown &unknown = unknownFromRow(row);
  unknown.redundant = true;
  swapRows(row, nRedundant);
  nRedundant++;
  return false;
}

// Check for redundant constraints and mark them as redundant.
// A constraint is considered redundant if the other non-redundant constraints
// already force this constraint to be non-negative.
//
// Now for each constraint that hasn't already been marked redundant, we check
// if it is redundant via constraintIsRedundant, and if it is, mark it as such.
void Simplex::detectRedundant() {
  if (empty)
    return;
  for (int i = con.size() - 1; i >= 0; i--) {
    if (con[i].redundant)
      continue;
    if (constraintIsRedundant(i)) {
      // constraintIsRedundant must leave the constraint in row position if it
      // returns true.
      assert(con[i].ownsRow &&
             "Constraint to be marked redundant must be a row!");
      markRedundant(con[i].pos);
    }
  }
}

unsigned Simplex::numberVariables() const { return var.size(); }
unsigned Simplex::numberConstraints() const { return con.size(); }

// Return a snapshot of the curent state. This is just the current size of the
// undo log.
unsigned Simplex::getSnapshot() const { return undoLog.size(); }

void Simplex::undoOp(UndoOp op, llvm::Optional<int> index) {
  // TODO why not switch case?
  if (op == UndoOp::DEALLOCATE) {
    assert(index.hasValue() &&
           "DEALLOCATE undo entry must be accompanied by an index");

    assert(index < 0 && "Unknown to be deallocated must be a constraint");
    Unknown &unknown = unknownFromIndex(*index);

    if (!unknown.ownsRow) {
      unsigned column = unknown.pos;
      llvm::Optional<unsigned> row;

      // Try to find any pivot row for this column that preserves tableau
      // consistency (except possibly the column itself, which is going to be
      // deallocated anyway).
      //
      // If no pivot row is found in either direction,
      // then the column is unbounded in both directions and we are free to
      // perform any pivot at all. To do this, we just need to find any row with
      // a non-zero coefficient for the column.
      if (auto maybeRow = findPivotRow({}, Direction::UP, column))
        row = *maybeRow;
      else if (auto maybeRow = findPivotRow({}, Direction::DOWN, column))
        row = *maybeRow;
      else {
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
    swapRows(unknown.pos, nRow - 1);
    // It is not strictly necessary to resize tableau, but for now we maintain
    // the invariant that the actual size of the tableau is (nRow, nCol).
    tableau.resize(nRow - 1, nCol);
    nRow--;
    rowVar.pop_back();
    con.pop_back();
  } else if (op == UndoOp::UNMARK_EMPTY) {
    empty = false;
  } else if (op == UndoOp::UNMARK_REDUNDANT) {
    assert(index.hasValue() &&
           "UNMARK_REDUNDANT undo entry must be accompanied by an index");
    assert(nRedundant != 0 && "No redundant constraints present.");

    Unknown &unknown = unknownFromIndex(*index);
    assert(unknown.ownsRow &&
           "Constraint to be unmarked as redundant must be a row");
    swapRows(unknown.pos, nRedundant - 1);
    unknown.redundant = false;
    nRedundant--;
  } else if (op == UndoOp::UNMARK_ZERO) {
    Unknown &unknown = unknownFromIndex(*index);
    if (!unknown.ownsRow) {
      assert(unknown.pos == liveColBegin - 1 &&
             "Column to be revived should be the last dead column");
      liveColBegin--;
    }
    unknown.zero = false;
  }
}

// Rollback to the specified snapshot.
//
// We undo all the log entries until the log size when the snapshot was taken
// is reached.
void Simplex::rollback(unsigned snapshot) {
  while (undoLog.size() > snapshot) {
    auto entry = undoLog.back();
    undoLog.pop_back();
    undoOp(entry.first, entry.second);
  }
}

llvm::Optional<Fraction<int64_t>>
Simplex::computeRowOptimum(Direction direction, unsigned row) {
  while (auto maybePivot = findPivot(row, direction)) {
    if (maybePivot->first == row)
      return {};
    pivot(*maybePivot);
  }
  return Fraction<int64_t>(tableau(row, 1), tableau(row, 0));
}

llvm::Optional<Fraction<int64_t>>
Simplex::computeOptimum(Direction direction, ArrayRef<int64_t> coeffs) {
  assert(!empty && "Tableau should not be empty");

  auto snap = getSnapshot();
  unsigned conIndex = addRow(coeffs);
  unsigned row = con[conIndex].pos;
  auto optimum = computeRowOptimum(direction, row);
  rollback(snap);
  return optimum;
}

template <typename T>
std::vector<T> concat(const std::vector<T> &v, const std::vector<T> &w) {
  auto result = v;
  result.insert(result.end(), w.begin(), w.end());
  return std::move(result);
}

// Make a tableau to represent a pair of points in the original tableau.
//
// The product constraints and variables are stored as: first A's, then B's.
//
// The product tableau has row layout:
//   A's redundant rows, B's redundant rows, A's other rows, B's other rows.
//
// It has column layout:
//   denominator, constant, A's columns, B's columns.
// TODO reconsider the following:
// TODO we don't need the dead columns or the redundant constraints. The caller
// only cares about duals of the new constraints that are added after returning
// from this function, so we can safely drop redundant constraints from the
// original simplex.
Simplex Simplex::makeProduct(const Simplex &A, const Simplex &B) {
  unsigned numVar = A.numberVariables() + B.numberVariables();
  unsigned numCon = A.numberConstraints() + B.numberConstraints();
  Simplex result(numVar);

  result.tableau.resize(numCon, numVar);
  result.nRedundant = A.nRedundant + B.nRedundant;
  result.liveColBegin = A.liveColBegin + B.liveColBegin;
  result.empty = A.empty || B.empty;

  auto indexFromBIndex = [&](int index) {
    return index >= 0 ? A.numberVariables() + index
                      : ~(A.numberConstraints() + ~index);
  };

  result.con = concat(A.con, B.con);
  result.var = concat(A.var, B.var);
  for (unsigned i = 2; i < A.liveColBegin; i++)
    result.colVar.push_back(A.colVar[i]);
  for (unsigned i = 2; i < B.liveColBegin; i++) {
    result.colVar.push_back(indexFromBIndex(B.colVar[i]));
    result.unknownFromIndex(result.colVar.back()).pos =
        result.colVar.size() - 1;
  }
  for (unsigned i = A.liveColBegin; i < A.nCol; i++) {
    result.colVar.push_back(A.colVar[i]);
    result.unknownFromIndex(result.colVar.back()).pos =
        result.colVar.size() - 1;
  }
  for (unsigned i = B.liveColBegin; i < B.nCol; i++) {
    result.colVar.push_back(indexFromBIndex(B.colVar[i]));
    result.unknownFromIndex(result.colVar.back()).pos =
        result.colVar.size() - 1;
  }

  auto appendRowFromA = [&](unsigned row) {
    for (unsigned col = 0; col < result.liveColBegin; col++)
      result.tableau(result.nRow, col) = A.tableau(row, col);
    unsigned offset = B.liveColBegin - 2;
    for (unsigned col = result.liveColBegin; col < A.nCol; col++)
      result.tableau(result.nRow, offset + col) = A.tableau(row, col);
    result.rowVar.push_back(A.rowVar[row]);
    result.unknownFromIndex(result.rowVar.back()).pos =
        result.rowVar.size() - 1;
    result.nRow++;
  };

  // Also fixes the corresponding entry in rowVar and var/con.
  auto appendRowFromB = [&](unsigned row) {
    result.tableau(result.nRow, 0) = A.tableau(row, 0);
    result.tableau(result.nRow, 1) = A.tableau(row, 1);
    unsigned offset = A.liveColBegin - 2;
    for (unsigned col = 2; col < B.liveColBegin; col++)
      result.tableau(result.nRow, offset + col) = B.tableau(row, col);

    offset = A.nCol - 2;
    for (unsigned col = B.liveColBegin; col < B.nCol; col++)
      result.tableau(result.nRow, offset + col) = B.tableau(row, col);
    result.rowVar.push_back(indexFromBIndex(B.rowVar[row]));
    result.unknownFromIndex(result.rowVar.back()).pos =
        result.rowVar.size() - 1;
    result.nRow++;
  };
  for (unsigned row = 0; row < A.nRedundant; row++)
    appendRowFromA(row);
  for (unsigned row = 0; row < B.nRedundant; row++)
    appendRowFromB(row);
  for (unsigned row = A.nRedundant; row < A.nRow; row++)
    appendRowFromA(row);
  for (unsigned row = B.nRedundant; row < B.nRow; row++)
    appendRowFromB(row);

  return result;
}

llvm::Optional<std::vector<int64_t>> Simplex::getSamplePointIfIntegral() const {
  if (empty)
    return {};

  std::vector<int64_t> sample;
  for (const Unknown &u : var) {
    if (!u.ownsRow)
      sample.push_back(0);
    else {
      if (tableau(u.pos, 1) % tableau(u.pos, 0) != 0)
        return {};
      sample.push_back(tableau(u.pos, 1) / tableau(u.pos, 0));
    }
  }
  return sample;
}

void Simplex::addFlatAffineConstraints(const FlatAffineConstraints &cs) {
  assert(cs.getNumIds() == numberVariables() &&
         "FlatAffineConstraints must have same dimensionality as simplex");
  for (unsigned i = 0; i < cs.getNumInequalities(); ++i)
    addInequality(cs.getInequality(i));
  for (unsigned i = 0; i < cs.getNumEqualities(); ++i)
    addEquality(cs.getEquality(i));
}

// Given a simplex for a polytope, construct a new simplex whose variables are
// identified with a pair of points (x, y) in the original polytope. Supports
// some operations needed for general basis reduction. In what follows, <x, y>
// denotes the dot product of the vectors x and y.
class GBRSimplex {
public:
  GBRSimplex(const Simplex &originalSimplex)
      : simplex(Simplex::makeProduct(originalSimplex, originalSimplex)),
        simplexConstraintOffset(simplex.numberConstraints()) {}

  // Add an equality <dir, x - y> = 0.
  void addEqualityForDirection(const std::vector<int64_t> &dir) {
    snapshotStack.push_back(simplex.getSnapshot());
    simplex.addEquality(getCoeffsForDirection(dir));
  }

  // Compute max(<dir, x - y>) and save the dual variables for only the
  // direction constraints to `dual`.
  Fraction<int64_t> computeWidthAndDuals(const std::vector<int64_t> &dir,
                                         std::vector<int64_t> &dual,
                                         int64_t &dualDenom) {
    unsigned snap = simplex.getSnapshot();
    unsigned conIndex = simplex.addRow(getCoeffsForDirection(dir));
    unsigned row = simplex.con[conIndex].pos;
    auto width = simplex.computeRowOptimum(Simplex::Direction::UP, row);
    assert(width && "Width should not be unbounded!");
    dualDenom = simplex.tableau(row, 0);
    dual.clear();
    // The increment is i += 2 because equalities are added as two inequalities,
    // one positive and one negative. We only want to process the positive ones.
    for (unsigned i = simplexConstraintOffset; i < conIndex; i += 2) {
      if (simplex.con[i].ownsRow)
        dual.push_back(0);
      else {
        // The dual variable is the negative of the row coefficient.
        dual.push_back(-simplex.tableau(row, simplex.con[i].pos));
      }
    }
    simplex.rollback(snap);
    return *width;
  }

  // Remove the last equality that was added through addEqualityForDirection.
  void removeLastEquality() {
    assert(!snapshotStack.empty() && "Snapshot stack is empty!");
    simplex.rollback(snapshotStack.back());
    snapshotStack.pop_back();
  }

private:
  // Returns coefficients for the expression <dir, x - y>.
  std::vector<int64_t> getCoeffsForDirection(ArrayRef<int64_t> dir) {
    assert(2 * dir.size() == simplex.numberVariables() &&
           "Direction vector has wrong dimensionality");
    std::vector<int64_t> coeffs;
    for (unsigned i = 0; i < dir.size(); i++)
      coeffs.emplace_back(dir[i]);
    for (unsigned i = 0; i < dir.size(); i++)
      coeffs.emplace_back(-dir[i]);
    coeffs.emplace_back(0); // constant term
    return coeffs;
  }

  Simplex simplex;
  // The first index of the equality constraints, the index immediately after
  // the last constraint in the initial product simplex.
  unsigned simplexConstraintOffset;
  // A stack of snapshots, used for rolling back.
  std::vector<unsigned> snapshotStack;
};

// Let b_{level}, b_{level + 1}, ... b_n be the current basis.
// Let F_i(v) = max <v, x - y> where x and y are points in the original polytope
// and <b_j, x - y> = 0 is satisfied for all j < i. (here <u, v> denotes the
// inner product)
//
// In every iteration, we first replace b_{i+1} with b_{i+1} + u*b_i, where u is
// the integer such that F_i(b_{i+1} + u*b_i) minimized. Let alpha be the dual
// variable associated with the constraint <b_i, x - y> = 0 when computing
// F_{i+1}(b_{i+1}). alpha must be the minimizing value of u, if it were allowed
// to be rational. Due to convexity, the minimizing integer value is either
// floor(alpha) or ceil(alpha), so we just need to check which of these gives a
// lower F_{i+1} value. If alpha turned out to be an integer, then u = alpha.
//
// Now if F_i(b_{i+1}) < eps * F_i(b_i), we swap b_i and (the new) b_{i + 1}
// and decrement i (unless i = level, in which case we stay at the same i).
// Otherwise, we increment i. We use eps = 0.75.
//
// In an iteration we need to compute:
//
// Some of the required F and alpha values may already be known. We cache the
// known values and reuse them if possible. In particular:
//
// When we set b_{i+1} to b_{i+1} + u*b_i, no F values are changed since we only
// added a multiple of b_i to b_{i+1}. In particular F_{i+1}(b_{i+1})
// = min F_i(b_{i+1} + alpha * b_i) so adding u*b_i to b_{i+1} does not
// change this. Also <b_i, x - y> = 0 and <b_{i+1}, x - y> = 0 already
// imply <b_{i+1} + u*b_i, x - y> = 0, so the constraints are unchanged.
//
// When we decrement i, we swap b_i and b_{i+1}. In the following paragraphs we
// always refer to the final vector b_{i+1} after updating). But note that when
// we come to the end of an iteration we always know F_i(b_{i+1}), so we just
// need to update the cached value to reflect this. However the value of
// F_{i+1}(b_i) is not already known, so if there was a stale value of F[i+1] in
// the cache we remove this. Moreover, the iteration after decrementing will
// want the dual variables from this computation so we cache this when we
// compute the minimizing value of u.
//
// If alpha turns out to be an integer, then we never compute F_i(b_{i+1}) in
// this iteration. But in this case, F_{i+1}(b_{i+1}) = F_{i+1}(b'_{i+1}) where
// b'_{i+1} is the vector before updating. Therefore, we can update the cache
// with this value. Furthermore, we can just inherit the dual variables from
// this computation.
//
// When incrementing i we do not make any changes to the basis so no
// invalidation occurs.
void Simplex::reduceBasis(Matrix<int64_t> &basis, unsigned level) {
  const Fraction<int64_t> epsilon(3, 4);

  if (level == basis.getNumRows() - 1)
    return;

  GBRSimplex gbrSimplex(*this);
  std::vector<Fraction<int64_t>> F;
  std::vector<int64_t> alpha;
  int64_t alphaDenom;
  auto findUAndGetFCandidate = [&](unsigned i) -> Fraction<int64_t> {
    assert(i < level + alpha.size() && "alpha_i is not known!");

    int64_t u = floorDiv(alpha[i - level], alphaDenom);
    basis.addToRow(i, i + 1, u);
    if (alpha[i - level] % alphaDenom != 0) {
      std::vector<int64_t> uAlpha[2];
      int64_t uAlphaDenom[2];
      Fraction<int64_t> F_i[2];

      // Initially u is floor(alpha) and basis reflects this.
      F_i[0] = gbrSimplex.computeWidthAndDuals(basis.getRow(i + 1), uAlpha[0],
                                               uAlphaDenom[0]);

      // Now try ceil(alpha), i.e. floor(alpha) + 1.
      ++u;
      basis.addToRow(i, i + 1, 1);
      F_i[1] = gbrSimplex.computeWidthAndDuals(basis.getRow(i + 1), uAlpha[1],
                                               uAlphaDenom[1]);

      int j = F_i[0] < F_i[1] ? 0 : 1;
      if (j == 0)
        // Subtract 1 to go from u = ceil(alpha) back to floor(alpha).
        basis.addToRow(i, i + 1, -1);

      alpha = std::move(uAlpha[j]);
      alphaDenom = uAlphaDenom[j];
      return F_i[j];
    }
    assert(i + 1 - level < F.size() && "F_{i+1} wasn't saved");
    // When alpha minimizes F_i(b_{i+1} + alpha*b_i), this is equal to
    // F_{i+1}(b_{i+1}).
    return F[i + 1 - level];
  };

  // In the ith iteration of the loop, gbrSimplex has constraints for directions
  // from `level` to i - 1.
  unsigned i = level;
  while (i < basis.getNumRows() - 1) {
    Fraction<int64_t> F_i_candidate; // F_i(b_{i+1} + u*b_i)
    if (i >= level + F.size()) {
      // We don't even know the value of F_i(b_i), so let's find that first.
      // We have to do this first since later we assume that F already contains
      // values up to and including i.

      assert((i == 0 || i - 1 < level + F.size()) &&
             "We are at level i but we don't know the value of F_{i-1}");

      // We don't actually use these duals at all, but it doesn't matter
      // because this case should only occur when i is level, and there are no
      // duals in that case anyway.
      assert(i == level && "This case should only occur when i == level");
      F.push_back(
          gbrSimplex.computeWidthAndDuals(basis.getRow(i), alpha, alphaDenom));
    }

    if (i >= level + alpha.size()) {
      assert(i + 1 >= level + F.size() && "We don't know alpha_i but we know "
                                          "F_{i+1}, this should never happen");
      // We don't know alpha for our level, so let's find it.
      gbrSimplex.addEqualityForDirection(basis.getRow(i));
      F.push_back(gbrSimplex.computeWidthAndDuals(basis.getRow(i + 1), alpha,
                                                  alphaDenom));
      gbrSimplex.removeLastEquality();
    }

    F_i_candidate = findUAndGetFCandidate(i);

    if (F_i_candidate < epsilon * F[i - level]) {
      basis.swapRows(i, i + 1);
      F[i - level] = F_i_candidate;
      // The values of F_{i+1}(b_{i+1}) and higher may change after the swap,
      // so we remove the cached values here.
      F.resize(i - level + 1);
      if (i == level) {
        // TODO (performance) isl seems to assume alpha is 0 in this case. Look
        // into this. For now we assume that alpha is not known and must be
        // recomputed.
        alpha.clear();
        continue;
      }

      gbrSimplex.removeLastEquality();
      i--;
      continue;
    }

    alpha.clear();
    gbrSimplex.addEqualityForDirection(basis.getRow(i));
    i++;
  }
}

// Try to find an integer sample point in the polytope.
//
// If such a point exists, this function returns it. Otherwise, it returns and
// empty llvm::Optional.
llvm::Optional<std::vector<int64_t>> Simplex::findIntegerSample() {
  if (empty)
    return {};

  unsigned nDims = var.size();
  Matrix<int64_t> basis = Matrix<int64_t>::getIdentityMatrix(nDims);
  return findIntegerSampleRecursively(basis, 0);
}

std::pair<int64_t, std::vector<int64_t>> Simplex::findRationalSample() const {
  int64_t denom = 1;
  for (const Unknown &u : var) {
    if (u.ownsRow)
      denom = lcm(denom, tableau(u.pos, 0));
  }
  std::vector<int64_t> sample;
  int64_t gcd = denom;
  for (const Unknown &u : var) {
    if (!u.ownsRow)
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

// Search for an integer sample point using a branch and bound algorithm.
//
// Each row in the basis matrix is a vector, and the set of basis vectors should
// span the space. Initially this is called with the identity matrix, i.e., the
// basis vectors are just the variables.
//
// In every level, a value is assigned to the level-th basis vector, as follows.
// Compute the minimum and maximum rational values of this direction.
// If only one integer point lies in this range, constrain the variable to
// have this value and recurse to the next variable.
//
// If the range has multiple values, perform general basis reduction via
// reduceBasis and then compute the bounds again. Now we can't do any better
// than this, so we just recurse on every integer value in this range.
//
// If the range contains no integer value, then of course the polytope is empty
// for the current assignment of the values in previous levels, so return to
// the previous level.
//
// If we reach the last level where all the variables have been assigned values
// already, then we simply return the current sample point if it is integral, or
// an empty llvm::Optional otherwise.
llvm::Optional<std::vector<int64_t>>
Simplex::findIntegerSampleRecursively(Matrix<int64_t> &basis, unsigned level) {
  if (level == basis.getNumRows())
    return getSamplePointIfIntegral();

  std::vector<int64_t> basisCoeffVector = basis.getRow(level);
  basisCoeffVector.emplace_back(0); // constant term

  auto getBounds = [&]() -> std::pair<int64_t, int64_t> {
    int64_t min_rounded_up;
    if (auto opt = computeOptimum(Direction::DOWN, basisCoeffVector))
      min_rounded_up = ceil(*opt);
    else
      llvm_unreachable("Tableau should not be unbounded");

    int64_t max_rounded_down;
    if (auto opt = computeOptimum(Direction::UP, basisCoeffVector))
      max_rounded_down = floor(*opt);
    else
      llvm_unreachable("Tableau should not be unbounded");

    return {min_rounded_up, max_rounded_down};
  };

  int64_t min_rounded_up, max_rounded_down;
  std::tie(min_rounded_up, max_rounded_down) = getBounds();

  // Heuristic: if the sample point is integral at this point, just return it.
  if (auto opt = getSamplePointIfIntegral())
    return *opt;

  if (min_rounded_up < max_rounded_down) {
    reduceBasis(basis, level);
    std::tie(min_rounded_up, max_rounded_down) = getBounds();
  }

  for (int64_t i = min_rounded_up; i <= max_rounded_down; ++i) {
    auto snapshot = getSnapshot();
    basisCoeffVector.back() = -i;
    // Add the constraint `basisCoeffVector = i`.
    addEquality(basisCoeffVector);
    if (auto opt = findIntegerSampleRecursively(basis, level + 1))
      return *opt;
    rollback(snapshot);
  }

  return {};
}

// The minimum of an unknown is obviously unbounded if it is a column variable
// and no constraint limits its value from below.
//
// The minimum of a row variable is not obvious because it depends on the
// boundedness of all referenced column variables.
//
// A column variable is bounded from below if there a exists a constraint for
// which the corresponding column coefficient is strictly positive and the row
// variable is non-negative (restricted).
inline bool Simplex::minIsObviouslyUnbounded(Unknown &unknown) const {
  // tableau.checkSparsity();
  if (unknown.ownsRow)
    return false;

  for (size_t i = nRedundant; i < nRow; i++) {
    if (unknownFromRow(i).restricted && tableau(i, unknown.pos) > 0)
      return false;
  }
  return true;
}

// The maximum of an unknown is obviously unbounded if it is a column variable
// and no constraint limits its value from above.
//
// The maximum of a row variable is not obvious because it depends on the
// boundedness of all referenced column variables.
//
// A column variable is surely unbounded from above if there does not exist a
// constraint for which the corresponding column coefficient is strictly
// negative and the row variable is non-negative (restricted).
inline bool Simplex::maxIsObviouslyUnbounded(Unknown &unknown) const {
  if (unknown.ownsRow)
    return false;

  for (unsigned row = nRedundant; row < nRow; row++) {
    if (tableau(row, unknown.pos) < 0 && unknownFromRow(row).restricted)
      return false;
  }
  return true;
}

// A row is obviously not constrained to be zero if
// - the tableau is rational and the constant term is not zero
// - the tableau is integer and the constant term is at least one (it is also
//   not zero if the constant term is at most negative one, but this is only
//   called for restricted rows, so it doesn't cost anything to be imprecise)
//
// This is because of the invariant that the sample value is always a valid
// point in the tableau (assuming the tableau is not empty).
inline bool Simplex::rowIsObviouslyNotZero(unsigned row) const {
  return tableau(row, 1) >= tableau(row, 0);
}

// A row is equal to zero if its constant term and all coefficients for live
// columns are equal to zero.
inline bool Simplex::rowIsObviouslyZero(unsigned row) const {
  if (tableau(row, 1) != 0)
    return false;
  for (unsigned col = liveColBegin; col < nCol; col++) {
    if (tableau(row, col) != 0)
      return false;
  }
  return true;
}

// An unknown is considered to be relevant if it is neither a redundant row nor
// a dead column
inline bool Simplex::unknownIsRelevant(Unknown &unknown) const {
  if (unknown.ownsRow && unknown.pos < nRedundant)
    return false;
  else if (!unknown.ownsRow && unknown.pos < liveColBegin)
    return false;
  return true;
}

// A row is obviously non integral if all of its non-dead column entries are
// zero and the constant term denominator is not divisible by the row
// denominator.
//
// If there are non-zero entries then integrality depends on the values of all
// referenced column variables.
inline bool Simplex::rowIsObviouslyNonIntegral(unsigned row) const {
  for (unsigned j = liveColBegin; j < nCol; j++) {
    if (tableau(row, j) != 0)
      return false;
  }
  return tableau(row, 1) % tableau(row, 0) != 0;
}

// Pivot the unknown to row position in the specified direction. If no direction
// is provided, both directions are allowed. The unknown is assumed to be
// bounded in the specified direction. If no direction is specified, the unknown
// is assumed to be unbounded in both directions.
//
// If the unknown is already in row position we need not do anything. Otherwise,
// we find a row to pivot to via findPivotRow and pivot to it.
// TODO currently doesn't support an optional direction
inline void Simplex::toRow(Unknown &unknown, Direction direction) {
  if (unknown.ownsRow)
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

// Compare the maximum value of the unknown to origin. Return +1 if the maximum
// value is greater than origin, 0 if they are equal, and -1 if it is less
// than origin.
//
// If the unknown is marked zero, then its maximum value is zero and we can
// return accordingly.
//
// If the maximum is obviously unbounded, we can return 1.
//
// Otherwise, we move the unknown up to a row and keep trying to find upward
// pivots until the unknown becomes unbounded or becomes maximised.
template <int origin>
int64_t Simplex::signOfMax(Unknown &u) {
  static_assert(origin >= 0, "");
  assert(!u.redundant && "signOfMax called for redundant unknown");

  if (maxIsObviouslyUnbounded(u))
    return 1;

  toRow(u, Direction::UP);

  // The only callsite where origin == 1 only cares whether it's negative or
  // not, so we can save some pivots by quitting early in this case. This is
  // needed for pivot-parity with isl.
  static_assert(origin == 0 || origin == 1, "");
  auto mustContinueLoop = [this](unsigned row) {
    return origin == 1 ? tableau(row, 1) < tableau(row, 0)
                       : tableau(row, 1) <= 0;
  };

  while (mustContinueLoop(u.pos)) {
    auto p = findPivot(u.pos, Direction::UP);
    if (!p) {
      // u is manifestly maximised
      return sign(tableau(u.pos, 1) - origin * tableau(u.pos, 0));
    } else if (p->first == u.pos) { // u is manifestly unbounded
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
  std::swap(colVar[i], colVar[j]);
  unknownFromColumn(i).pos = i;
  unknownFromColumn(j).pos = j;
}

// Remove the row from the tableau.
//
// If the row is not already the last one, swap it with the last row.
// Then decrement the row count, remove the constraint entry, and remove the
// entry in row_var.
inline void Simplex::dropRow(unsigned row) {
  // It is unclear why this restriction exists. Perhaps because bmaps outside
  // keep track of the number of equalities, and hence moving around constraints
  // would require updating them?
  assert(~rowVar[row] == int(con.size() - 1) &&
         "Row to be dropped must be the last constraint");

  if (row != nRow - 1)
    swapRows(row, nRow - 1);
  nRow--;
  rowVar.pop_back();
  con.pop_back();
}

inline bool Simplex::killCol(unsigned col) {
  Unknown &unknown = unknownFromColumn(col);
  unknown.zero = true;
  undoLog.emplace_back(UndoOp::UNMARK_ZERO, colVar[col]);
  if (col != liveColBegin)
    swapColumns(col, liveColBegin);
  liveColBegin++;
  // tableau.checkSparsity();
  return false;
}

inline int Simplex::indexFromUnknown(const Unknown &u) const {
  if (u.ownsRow)
    return rowVar[u.pos];
  else
    return colVar[u.pos];
}

// If temp_row is set, the last row is a temporary unknown and undo entries
// should not be written for this row.
inline void Simplex::closeRow(unsigned row, bool temp_row) {
  Unknown *u = &unknownFromRow(row);
  assert(u->restricted && "expected restricted variable\n");

  if (!u->zero && !temp_row) {
    // pushUndoEntryIfNeeded(UndoOp::UNMARK_ZERO, *u);
    undoLog.emplace_back(UndoOp::UNMARK_ZERO, indexFromUnknown(*u));
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
  if (!temp_row)
    markRedundant(u->pos);

  if (!empty)
    // Check if there are any vars that can't attain integral values.
    for (const Unknown &u : var)
      if (u.ownsRow && rowIsObviouslyNonIntegral(u.pos)) {
        markEmpty();
        break;
      }
}

inline void Simplex::extendConstraints(unsigned n_new) {
  if (con.capacity() < con.size() + n_new)
    con.reserve(con.size() + n_new);
  if (tableau.getNumRows() < nRow + n_new) {
    tableau.resize(nRow + n_new, tableau.getNumColumns());
    rowVar.reserve(nRow + n_new);
  }
}

// Given a constraint con >= 0, add another constraint -con >= 0 so that we cut
// our polytope to the hyperplane con = 0. Before the function returns the added
// constraint is removed again, but the effects on the other unknowns remain.
//
// If the constraint is in row position, add a new row with all the terms
// negated. If the constrain is in column position, add a row with a single
// -1 coefficient for that column and all zeroes in other columns.
//
// If our new constraint cannot achieve non-negative values, then the tableau is
// empty and we marked it as such. Otherwise, we know that its value must always
// be zero so we close the row and then drop it. Closing the row results in some
// columns being killed, so the effect of fixing it to be zero remains even
// after it is dropped.
inline void Simplex::cutToHyperplane(int con_index) {
  if (con[con_index].zero)
    return;
  assert(!con[con_index].redundant && con[con_index].restricted &&
         "expecting non-redundant non-negative variable");

  extendConstraints(1);
  rowVar.push_back(~con.size());
  con.emplace_back(true, false, nRow);
  Unknown &unknown = con[con_index];
  Unknown &temp_var = con.back();

  if (unknown.ownsRow) {
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

  int64_t sgn = signOfMax<0>(temp_var);
  if (sgn < 0) {
    assert(temp_var.ownsRow && "temp_var is in column position");
    dropRow(nRow - 1);
    markEmpty();
    return;
  }
  temp_var.restricted = true;
  assert(sgn <= 0 && "signOfMax is positive for the negated constraint");

  assert(temp_var.ownsRow && "temp_var is in column position");
  if (temp_var.pos != nRow - 1)
    tableau.swapRows(temp_var.pos, nRow - 1);

  closeRow(temp_var.pos, true);
  dropRow(temp_var.pos);
}

// Check for constraints that are constrained to be equal to zero. i.e. check
// for non-negative unknowns whose maximal value is
// - zero (for rational tableaus)
// - strictly less than one (for integer tableaus)
// Close unknowns with maximal value zero. In integer tableaus, if an unknown
// has a maximal value less than one, add a constraint to fix it to zero.
//
// We first mark unknowns which are restricted, not dead, not redundant, and
// which aren't obviously able to reach non-zero values.
//
// We iterate through the marked unknowns. If an unknown's maximal value (found
// via signOfMax<0>) is zero, then it is an equality, so we close the row. (the
// unknown is always in row position when signOfMax returns 0)
//
// Otherwise, if we have an integer tableau, we check if the unknown's maximal
// value is at least one. If not, then in an integer tableau it must be zero.
// We add another constraint fixing it to be zero via cutToHyperplane. Since
// this might have created new equalities, we call detectImplicitEqualities
// again to find these.
//
// TODO: consider changing the name, something with 'zero' is perhaps more
// indicative than equality. And 'detect' sounds more passive than what we're
// doing (adding constraints, closing rows).
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

void Simplex::printUnknown(llvm::raw_ostream &os, const Unknown &u) const {
  os << (u.ownsRow ? "r" : "c");
  os << u.pos;
  if (u.restricted)
    llvm::errs() << " [>=0]";
  if (u.redundant)
    llvm::errs() << " [R]";
}

void Simplex::print(llvm::raw_ostream &os) const {
  os << "Dumping Simplex, rows = " << nRow << ", columns = " << nCol
     << "\nnRedundant = " << nRedundant << "\n";
  if (empty)
    os << "Simplex marked empty!\n";
  os << "var: ";
  for (unsigned i = 0; i < var.size(); ++i) {
    if (i > 0)
      os << ", ";
    printUnknown(os, var[i]);
  }
  os << "\ncon: ";
  for (unsigned i = 0; i < con.size(); ++i) {
    if (i > 0)
      os << ", ";
    printUnknown(os, con[i]);
  }
  os << '\n';
  for (unsigned row = 0; row < nRow; ++row) {
    if (row > 0)
      os << ", ";
    os << "r" << row << ": " << rowVar[row];
  }
  os << '\n';
  os << "c0: denom, c1: const";
  for (unsigned col = 2; col < nCol; ++col)
    os << ", c" << col << ": " << colVar[col];
  os << '\n';
  for (unsigned row = 0; row < nRow; ++row) {
    for (unsigned col = 0; col < nCol; ++col)
      os << tableau(row, col) << '\t';
    os << '\n';
  }
  os << '\n';
}

void Simplex::dump() const { print(llvm::errs()); }

} // namespace mlir
