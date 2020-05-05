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
    : nRow(0), nCol(2), nRedundant(0), tableau(0, 2 + nVar), empty(false) {
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
  for (unsigned i = 0; i < constraints.getNumInequalities(); ++i)
    addInequality(constraints.getInequality(i));
  for (unsigned i = 0; i < constraints.getNumEqualities(); ++i)
    addEquality(constraints.getEquality(i));
}

const Simplex::Unknown &Simplex::unknownFromIndex(int index) const {
  assert(index != NULL_INDEX && "NULL_INDEX passed to unknownFromIndex");

  if (index >= 0)
    return var[index];
  else
    return con[~index];
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

  if (index >= 0)
    return var[index];
  else
    return con[~index];
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

  if (nRow >= tableau.getNumRows())
    tableau.resize(tableau.getNumRows() + 1, tableau.getNumColumns());

  rowVar.push_back(~con.size());
  nRow++;

  con.emplace_back(true, false, nRow - 1);

  tableau(nRow - 1, 0) = 1;
  tableau(nRow - 1, 1) = coeffs.back();
  for (unsigned col = 2; col < nCol; ++col)
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
    gcd = llvm::greatestCommonDivisor(gcd, tableau(row, col));
  }
  for (unsigned col = 0; col < nCol; ++col)
    tableau(row, col) /= gcd;
}

bool Simplex::signMatchesDirection(int64_t elem, Direction direction) const {
  assert(elem != 0 && "elem should not be 0");
  if (direction == Direction::UP)
    return elem > 0;
  else
    return elem < 0;
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
  for (unsigned j = 2; j < nCol; ++j) {
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
  assert((pivotRow >= nRedundant && pivotCol >= 2) &&
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

// Add an inequality to the tableau. If coeffs is c_0, c_1, ... c_n, where n
// is the curent number of variables, then the corresponding inequality is
// c_n + c_0*x_0 + c_1*x_1 + ... + c_{n-1}*x_{n-1} >= 0.
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
unsigned Simplex::getSnapshot() const {
  return undoLog.size();
}

void Simplex::undoOp(UndoOp op, llvm::Optional<int> index) {
  if (op == UndoOp::DEALLOCATE) {
    assert(index.hasValue() && "DEALLOCATE undo entry must be accompanied by an index");

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

    // Move this unknown to the last row and remove the last row from the tableau.
    swapRows(unknown.pos, nRow - 1);
    // It is not strictly necessary to resize tableau, but for now we maintain the invariant
    // that the actual size of the tableau is (nRow, nCol).
    tableau.resize(nRow - 1, nCol);
    nRow--;
    rowVar.pop_back();
    con.pop_back();
  } else if (op == UndoOp::UNMARK_EMPTY) {
    empty = false;
  } else if (op == UndoOp::UNMARK_REDUNDANT) {
    assert(index.hasValue() && "UNMARK_REDUNDANT undo entry must be accompanied by an index");
    assert(nRedundant != 0 && "No redundant constraints present.");

    Unknown &unknown = unknownFromIndex(*index);
    assert(unknown.ownsRow &&
           "Constraint to be unmarked as redundant must be a row");
    swapRows(unknown.pos, nRedundant - 1);
    unknown.redundant = false;
    nRedundant--;
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

llvm::Optional<Fraction<int64_t>> Simplex::computeOptimum(
    Direction direction, ArrayRef<int64_t> coeffs) {
  assert(!empty && "Tableau should not be empty");

  auto snap = getSnapshot();
  unsigned conIndex = addRow(coeffs);
  unsigned row = con[conIndex].pos;
  auto optimum = computeRowOptimum(direction, row);
  rollback(snap);
  return optimum;
}
void Simplex::dumpUnknown(const Unknown &u) const {
  llvm::errs() << (u.ownsRow ? "r" : "c");
  llvm::errs() << u.pos;
  if (u.restricted)
    llvm::errs() << " [>=0]";
  if (u.redundant)
    llvm::errs() << " [R]";
}

void Simplex::dump() const {
  llvm::errs() << "Dumping Simplex, rows = " << nRow << ", columns = " << nCol
               << "\nnRedundant = " << nRedundant << "\n";
  if (empty)
    llvm::errs() << "Simplex marked empty!\n";
  llvm::errs() << "var: ";
  for (unsigned i = 0; i < var.size(); ++i) {
    if (i > 0)
      llvm::errs() << ", ";
    dumpUnknown(var[i]);
  }
  llvm::errs() << "\ncon: ";
  for (unsigned i = 0; i < con.size(); ++i) {
    if (i > 0)
      llvm::errs() << ", ";
    dumpUnknown(con[i]);
  }
  llvm::errs() << '\n';
  for (unsigned row = 0; row < nRow; ++row) {
    if (row > 0)
      llvm::errs() << ", ";
    llvm::errs() << "r" << row << ": " << rowVar[row];
  }
  llvm::errs() << '\n';
  llvm::errs() << "c0: denom, c1: const";
  for (unsigned col = 2; col < nCol; ++col)
    llvm::errs() << ", c" << col << ": " << colVar[col];
  llvm::errs() << '\n';
  for (unsigned row = 0; row < nRow; ++row) {
    for (unsigned col = 0; col < nCol; ++col)
      llvm::errs() << tableau(row, col) << '\t';
    llvm::errs() << '\n';
  }
  llvm::errs() << '\n';
}

} // namespace mlir
