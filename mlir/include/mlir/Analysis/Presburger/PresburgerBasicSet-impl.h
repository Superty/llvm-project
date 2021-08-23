//===- PresburgerBasicSet.cpp - MLIR PresburgerBasicSet Class -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_BASIC_SET_IMPL_H
#define MLIR_ANALYSIS_PRESBURGER_BASIC_SET_IMPL_H

#include "mlir/Analysis/Presburger/ISLPrinter.h"
#include "mlir/Analysis/Presburger/LinearTransform.h"
#include "mlir/Analysis/Presburger/ParamLexSimplex.h"
#include "mlir/Analysis/Presburger/Printer.h"
#include "mlir/Analysis/Presburger/Constraint.h"
#include "mlir/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::analysis;
using namespace mlir::analysis::presburger;

template <typename Int>
template <typename OInt>
PresburgerBasicSet<Int>::PresburgerBasicSet(const PresburgerBasicSet<OInt> &o) :
  ineqs(convert<InequalityConstraint<Int>>(o.ineqs)),
  eqs(convert<EqualityConstraint<Int>>(o.eqs)),
  divs(convert<DivisionConstraint<Int>>(o.divs)),
  nDim(o.nDim), nParam(o.nParam), nExist(o.nExist) {}

template <typename Int>
PresburgerBasicSet<Int>::PresburgerBasicSet(
    unsigned oNDim, unsigned oNParam, unsigned oNExist,
    const ArrayRef<DivisionConstraint<Int>> oDivs)
    : nDim(oNDim), nParam(oNParam), nExist(oNExist) {
  divs = llvm::to_vector<8>(oDivs);
}

template <typename Int>
PresburgerBasicSet<Int>::PresburgerBasicSet(const FlatAffineConstraints &fac)
    : nDim(fac.getNumDimIds()), nParam(fac.getNumSymbolIds()),
      nExist(fac.getNumLocalIds()) {
  for (unsigned i = 0, e = fac.getNumInequalities(); i < e; ++i)
    addInequality(fac.getInequality(i));
  for (unsigned i = 0, e = fac.getNumEqualities(); i < e; ++i)
    addEquality(fac.getEquality(i));
}

template <typename Int>
void PresburgerBasicSet<Int>::addInequality(ArrayRef<Int> coeffs) {
  ineqs.emplace_back(coeffs);
}

template <typename Int>
void PresburgerBasicSet<Int>::removeLastInequality() { ineqs.pop_back(); }

template <typename Int>
void PresburgerBasicSet<Int>::removeLastEquality() { eqs.pop_back(); }

template <typename Int>
const InequalityConstraint<Int> &
PresburgerBasicSet<Int>::getInequality(unsigned i) const {
  return ineqs[i];
}

template <typename Int>
const EqualityConstraint<Int> &PresburgerBasicSet<Int>::getEquality(unsigned i) const {
  return eqs[i];
}

template <typename Int>
ArrayRef<InequalityConstraint<Int>> PresburgerBasicSet<Int>::getInequalities() const {
  return ineqs;
}

template <typename Int>
ArrayRef<EqualityConstraint<Int>> PresburgerBasicSet<Int>::getEqualities() const {
  return eqs;
}

template <typename Int>
ArrayRef<DivisionConstraint<Int>> PresburgerBasicSet<Int>::getDivisions() const {
  return divs;
}

template <typename Int>
void PresburgerBasicSet<Int>::removeLastDivision() {
  divs.pop_back();
  for (auto &ineq : ineqs)
    ineq.removeLastDimension();
  for (auto &eq : eqs)
    eq.removeLastDimension();
  for (auto &div : divs)
    div.removeLastDimension();
}

template <typename Int>
void PresburgerBasicSet<Int>::removeDivision(unsigned i) {
  unsigned divIdx = getDivOffset() + i;

  for (auto &ineq : ineqs)
    ineq.eraseDimensions(divIdx, 1);
  for (auto &eq : eqs)
    eq.eraseDimensions(divIdx, 1);
  for (auto &div : divs) {
    div.eraseDimensions(divIdx, 1);
  }

  divs.erase(divs.begin() + i);
}

template <typename Int>
void PresburgerBasicSet<Int>::addEquality(ArrayRef<Int> coeffs) {
  eqs.emplace_back(coeffs);
}

template <typename Int>
PresburgerBasicSet<Int> PresburgerBasicSet<Int>::makePlainBasicSet() const {
  PresburgerBasicSet plainBasicSet(getNumTotalDims(), 0, 0);
  plainBasicSet.ineqs = ineqs;
  plainBasicSet.eqs = eqs;
  for (const DivisionConstraint<Int> &div : divs) {
    plainBasicSet.ineqs.emplace_back(div.getInequalityLowerBound());
    plainBasicSet.ineqs.emplace_back(div.getInequalityUpperBound());
  }
  return plainBasicSet;
}

template <typename Int>
Optional<SmallVector<Int, 8>>
PresburgerBasicSet<Int>::findIntegerSampleRemoveEqs(bool onlyEmptiness) const {
  auto copy = *this;
  if (!ineqs.empty()) {
    Simplex<Int> simplex(copy);
    simplex.detectImplicitEqualities();
    copy.updateFromSimplex(simplex);
  }

  auto coeffMatrix = copy.coefficientMatrixFromEqs();
  LinearTransform<Int> U =
      LinearTransform<Int>::makeTransformToColumnEchelon(coeffMatrix);
  SmallVector<Int, 8> vals;
  vals.reserve(copy.getNumTotalDims());
  unsigned col = 0;
  for (unsigned row = 0, e = copy.eqs.size(); row < e; ++row) {
    if (col == copy.getNumTotalDims())
      break;
    const auto &coeffs = coeffMatrix.getRow(row);
    if (coeffs[col] == 0)
      continue;
    Int val = copy.eqs[row].getCoeffs().back();
    for (unsigned c = 0; c < col; ++c) {
      val -= vals[c] * coeffs[c];
    }
    if (val % coeffs[col] != 0)
      return {};
    vals.push_back(-val / coeffs[col]);
    col++;
  }

  if (copy.ineqs.empty()) {
    if (onlyEmptiness)
      return vals;
    // Pad with zeros.
    vals.resize(copy.getNumTotalDims());
    return U.preMultiplyColumn(vals);
  }

  copy.eqs.clear();
  PresburgerBasicSet T = U.postMultiplyBasicSet(copy);
  T.substitute(vals);
  return T.findIntegerSample(onlyEmptiness);
}

template <typename Int>
Optional<SmallVector<Int, 8>>
PresburgerBasicSet<Int>::findIntegerSample(bool onlyEmptiness) const {
  if (!isPlainBasicSet())
    return makePlainBasicSet().findIntegerSample();
  if (!eqs.empty())
    return findIntegerSampleRemoveEqs(onlyEmptiness);
  PresburgerBasicSet cone = makeRecessionCone();

  if (cone.getNumEqualities() == 0 && onlyEmptiness) {
    // No shifting of constraints can make a full-dimensional cone integer empty.
    // Therefore, if the recession cone of this basic set is a full-dimensional
    // cone, then the basic set is certainly non-empty. However, this is only
    // true when the cone is represented without any trivial constraints.
    // A trivial constraint can be contradictory on its own just by shifting,
    // for example 1 >= 0 is valid but -1 >= 0 is not valid, and similarly for
    // equalities. Hence, we check explicitly for this case.
    if (isTrivial() == -1)
      return {};
    return SmallVector<Int, 8>();
  }
  if (cone.getNumEqualities() < getNumTotalDims())
    return findSampleUnbounded(cone, onlyEmptiness);
  else
    return findSampleBounded(onlyEmptiness);
}

template <typename Int>
int PresburgerBasicSet<Int>::isTrivial() const {
  bool allTrivial = true;
  for (const auto& ineq : ineqs) {
    if (ineq.isTrivial()) {
      if (ineq.getConstant() < 0)
        return -1;
    }
    else {
      allTrivial = false;
    }
  }

  for (const auto& eq : eqs) {
    if (eq.isTrivial()) {
      if (eq.getConstant() != 0)
        return -1;
    }
    else {
      allTrivial = false;
    }
  }

  if (allTrivial)
    return +1;

  return 0;
}

template <typename Int>
bool PresburgerBasicSet<Int>::isIntegerEmpty() const {
  // dumpISL();
  if (ineqs.empty() && eqs.empty())
    return false;
  return !findIntegerSample(true);
}

template <typename Int>
Optional<std::pair<Int, SmallVector<Int, 8>>>
PresburgerBasicSet<Int>::findRationalSample() const {
  Simplex<Int> simplex(*this);
  if (simplex.isEmpty())
    return {};
  return simplex.findRationalSample();
}

// Returns a matrix of the constraint coefficients in the specified vector.
//
// This only makes a matrix of the coefficients! The constant terms are
// omitted.

template <typename Int>
Matrix<Int> PresburgerBasicSet<Int>::coefficientMatrixFromEqs() const {
  // TODO check if this works because of missing symbols
  Matrix<Int> result(getNumEqualities(), getNumTotalDims());
  for (unsigned i = 0; i < getNumEqualities(); ++i) {
    for (unsigned j = 0; j < getNumTotalDims(); ++j)
      result(i, j) = eqs[i].getCoeffs()[j];
  }
  return result;
}

template <typename Int>
bool PresburgerBasicSet<Int>::isPlainBasicSet() const {
  return nParam == 0 && nExist == 0 && divs.empty();
}

template <typename Int>
void PresburgerBasicSet<Int>::substitute(ArrayRef<Int> values) {
  assert(isPlainBasicSet());
  for (auto &ineq : ineqs)
    ineq.substitute(values);
  for (auto &eq : eqs)
    eq.substitute(values);
  // for (auto &div : divs)
  //   div.substitute(values);
  // if (values.size() >= nDim - divs.size()) {
  //   nExist = 0;
  //   divs = std::vector(
  //     divs.begin() + values.size() - (nDim - divs.size()), divs.end());
  // } else if (values.size() >= nDim - nExist - divs.size())
  //   nExist -= values.size() - (nDim - nExist - divs.size());
  nDim -= values.size();
}

// Find a sample in the basic set, which has some unbounded dimensions and whose
// recession cone is `cone`.
//
// We first change basis to one where the bounded directions are the first
// directions. To do this, observe that each of the equalities in the cone
// represent a bounded direction. Now, consider the matrix where every row is
// an equality and every column is a coordinate (and constant terms are
// omitted). Note that the transform that puts this matrix in column echelon
// form can be viewed as a transform that performs our required rotation.
//
// After rotating, we find a sample for the bounded dimensions and substitute
// this into the transformed set, producing a full-dimensional cone (not
// necessarily centred at origin). We obtain a sample from this using
// findSampleFullCone. The sample for the whole transformed set is the
// concatanation of the two samples.
//
// Let the initial transform be U. Let the constraints matrix be M. We have
// found a sample x satisfying the transformed constraint matrix MU. Therefore,
// Ux is a sample that satisfies M.
template <typename Int>
llvm::Optional<SmallVector<Int, 8>>
PresburgerBasicSet<Int>::findSampleUnbounded(PresburgerBasicSet &cone,
                                        bool onlyEmptiness) const {
  auto coeffMatrix = cone.coefficientMatrixFromEqs();
  LinearTransform<Int> U =
      LinearTransform<Int>::makeTransformToColumnEchelon(coeffMatrix);
  PresburgerBasicSet transformedSet = U.postMultiplyBasicSet(*this);

  auto maybeBoundedSample =
      transformedSet.findBoundedDimensionsSample(cone, onlyEmptiness);
  if (!maybeBoundedSample)
    return {};
  if (onlyEmptiness)
    return maybeBoundedSample;

  transformedSet.substitute(*maybeBoundedSample);

  auto maybeUnboundedSample = transformedSet.findSampleFullCone();
  if (!maybeUnboundedSample)
    return {};

  // TODO change to SmallVector!

  SmallVector<Int, 8> sample(*maybeBoundedSample);
  sample.insert(sample.end(), maybeUnboundedSample->begin(),
                maybeUnboundedSample->end());
  return U.preMultiplyColumn(std::move(sample));
}

// Find a sample in this basic set, which must be a full-dimensional cone
// (not necessarily centred at origin).
//
// We are going to shift the cone such that any rational point in it can be
// rounded up to obtain a valid integer point.
//
// Let each constraint of the cone be of the form <a, x> >= c. For every x that
// satisfies this, we want x rounded up to also satisfy this. It is enough to
// ensure that x + e also satisfies this for any e such that every coordinate is
// in [0, 1). So we want <a, x> + <a, e> >= c. This is satisfied if we satisfy
// the single constraint <a, x> + sum_{a_i < 0} a_i >= c.
// 
template <typename Int>
Optional<SmallVector<Int, 8>> PresburgerBasicSet<Int>::findSampleFullCone() {
  // NOTE isl instead makes a recession cone, shifts the cone to some rational
  // point in the initial set, and then does the following on the shifted cone.
  // It is unclear why we need to do all that since the current basic set is
  // already the required shifted cone.
  for (unsigned i = 0, e = getNumInequalities(); i < e; ++i) {
    Int shift = 0;
    for (unsigned j = 0, e = getNumTotalDims(); j < e; ++j) {
      Int coeff = ineqs[i].getCoeffs()[j];
      if (coeff < 0)
        shift += coeff;
    }
    // adapt the constant
    ineqs[i].shift(shift);
  }

  auto sample = findRationalSample();
  if (!sample)
    return {};
  // TODO: This is only guaranteed if simplify is present
  // assert(sample && "Shifted set became empty!");
  for (auto &value : sample->second)
    value = ceilDiv(value, sample->first);

  return sample->second;
}

// Project this basic set to its bounded dimensions. It is assumed that the
// unbounded dimensions occupy the last \p unboundedDims dimensions.
//
// We can simply drop the constraints which involve the unbounded dimensions.
// This is because no combination of constraints involving unbounded
// dimensions can produce a bound on a bounded dimension.
//
// Proof sketch: suppose we are able to obtain a combination of constraints
// involving unbounded constraints which is of the form <a, x> + c >= y,
// where y is a bounded direction and x are the remaining directions. If this
// gave us an upper bound u on y, then we have u >= <a, x> + c - y >= 0, which
// means that a linear combination of the unbounded dimensions was bounded
// which is impossible since we are working in a basis where all bounded
// directions lie in the span of the first `nDim - unboundedDims` directions.
// 
template <typename Int>
void PresburgerBasicSet<Int>::projectOutUnboundedDimensions(unsigned unboundedDims) {
  assert(isPlainBasicSet());
  unsigned remainingDims = getNumTotalDims() - unboundedDims;

  // TODO: support for symbols

  for (unsigned i = 0; i < getNumEqualities();) {
    bool nonZero = false;
    for (unsigned j = remainingDims, e = getNumTotalDims(); j < e; j++) {
      if (eqs[i].getCoeffs()[j] != 0) {
        nonZero = true;
        break;
      }
    }

    if (nonZero) {
      removeEquality(i);
      // We need to test the index i again.
      continue;
    }

    i++;
  }
  for (unsigned i = 0; i < getNumInequalities();) {
    bool nonZero = false;
    for (unsigned j = remainingDims, e = getNumTotalDims(); j < e; j++) {
      if (ineqs[i].getCoeffs()[j] != 0) {
        nonZero = true;
        break;
      }
    }

    if (nonZero) {
      removeInequality(i);
      // We need to test the index i again.
      continue;
    }

    i++;
  }

  for (auto &ineq : ineqs)
    ineq.eraseDimensions(remainingDims, unboundedDims);
  for (auto &eq : eqs)
    eq.eraseDimensions(remainingDims, unboundedDims);
  nDim = remainingDims;
}

template <typename Int>
Optional<SmallVector<Int, 8>>
PresburgerBasicSet<Int>::findBoundedDimensionsSample(const PresburgerBasicSet &cone,
                                                bool onlyEmptiness) const {
  assert(cone.isPlainBasicSet());
  PresburgerBasicSet boundedSet = *this;
  boundedSet.projectOutUnboundedDimensions(getNumTotalDims() -
                                           cone.getNumEqualities());
  return boundedSet.findSampleBounded(onlyEmptiness);
}

template <typename Int>
Optional<SmallVector<Int, 8>>
PresburgerBasicSet<Int>::findSampleBounded(bool onlyEmptiness) const {
  if (getNumTotalDims() == 0)
    return SmallVector<Int, 8>();
  return Simplex<Int>(*this).findIntegerSample();
}

// We shift all the constraints to the origin, then construct a simplex and
// detect implicit equalities. If a direction was intially both upper and lower
// bounded, then this operation forces it to be equal to zero, and this gets
// detected by simplex.

template <typename Int>
PresburgerBasicSet<Int> PresburgerBasicSet<Int>::makeRecessionCone() const {
  PresburgerBasicSet cone = *this;

  // TODO: check this
  for (unsigned r = 0, e = cone.getNumEqualities(); r < e; r++)
    cone.eqs[r].shiftToOrigin();

  for (unsigned r = 0, e = cone.getNumInequalities(); r < e; r++)
    cone.ineqs[r].shiftToOrigin();

  // NOTE isl does gauss here.

  Simplex<Int> simplex(cone);
  if (simplex.isEmpty()) {
    // TODO: empty flag for PresburgerBasicSet
    // cone.maybeIsEmpty = true;
    return cone;
  }

  // The call to detectRedundant can be removed if we gauss below.
  // Otherwise, this is needed to make it so that the number of equalities
  // accurately represents the number of bounded dimensions.
  simplex.detectRedundant();
  simplex.detectImplicitEqualities();
  cone.updateFromSimplex(simplex);

  // NOTE isl does gauss here.

  return cone;
}

template <typename Int>
void PresburgerBasicSet<Int>::removeInequality(unsigned i) {
  ineqs.erase(ineqs.begin() + i, ineqs.begin() + i + 1);
}

template <typename Int>
void PresburgerBasicSet<Int>::removeEquality(unsigned i) {
  eqs.erase(eqs.begin() + i, eqs.begin() + i + 1);
}

template <typename Int>
void PresburgerBasicSet<Int>::insertDimensions(unsigned pos, unsigned count) {
  if (count == 0)
    return;

  for (auto &ineq : ineqs)
    ineq.insertDimensions(pos, count);
  for (auto &eq : eqs)
    eq.insertDimensions(pos, count);
  for (auto &div : divs)
    div.insertDimensions(pos, count);
}

template <typename Int>
void PresburgerBasicSet<Int>::appendDivisionVariable(ArrayRef<Int> coeffs,
                                                Int denom) {
  assert(coeffs.size() == getNumTotalDims() + 1);
  divs.emplace_back(coeffs, denom, /*variable = */ getNumTotalDims());

  for (auto &ineq : ineqs)
    ineq.appendDimension();
  for (auto &eq : eqs)
    eq.appendDimension();
  for (auto &div : divs)
    div.appendDimension();
}

// TODO we can make these mutable arrays and move the divs in our only use case.

template <typename Int>
void PresburgerBasicSet<Int>::appendDivisionVariables(
    ArrayRef<DivisionConstraint<Int>> newDivs) {
#ifndef NDEBUG
  for (auto &div : newDivs)
    assert(div.getCoeffs().size() == getNumTotalDims() + newDivs.size() + 1);
#endif
  insertDimensions(nParam + nDim + nExist + divs.size(), newDivs.size());
  divs.insert(divs.end(), newDivs.begin(), newDivs.end());
}

template <typename Int>
void PresburgerBasicSet<Int>::prependDivisionVariables(
    ArrayRef<DivisionConstraint<Int>> newDivs) {
  insertDimensions(nParam + nDim + nExist, newDivs.size());
  divs.insert(divs.begin(), newDivs.begin(), newDivs.end());
}

template <typename Int>
void PresburgerBasicSet<Int>::prependExistentialDimensions(unsigned count) {
  insertDimensions(nParam + nDim, count);
  nExist += count;
}

template <typename Int>
void PresburgerBasicSet<Int>::appendExistentialDimensions(unsigned count) {
  insertDimensions(nParam + nDim + nExist, count);
  nExist += count;
}

template <typename Int>
void PresburgerBasicSet<Int>::toCommonSpace(PresburgerBasicSet &a,
                                       PresburgerBasicSet &b) {
  unsigned initialANExist = a.nExist;
  a.appendExistentialDimensions(b.nExist);
  b.prependExistentialDimensions(initialANExist);

  unsigned offset = a.nParam + a.nDim + a.nExist;
  SmallVector<DivisionConstraint<Int>, 8> aDivs = a.divs, bDivs = b.divs;
  for (DivisionConstraint<Int> &div : aDivs)
    div.insertDimensions(offset + aDivs.size(), bDivs.size());
  for (DivisionConstraint<Int> &div : bDivs)
    div.insertDimensions(offset, aDivs.size());

  a.appendDivisionVariables(bDivs);
  b.prependDivisionVariables(aDivs);
}

template <typename Int>
void PresburgerBasicSet<Int>::intersect(PresburgerBasicSet bs) {
  toCommonSpace(*this, bs);
  ineqs.insert(ineqs.end(), std::make_move_iterator(bs.ineqs.begin()),
               std::make_move_iterator(bs.ineqs.end()));
  eqs.insert(eqs.end(), std::make_move_iterator(bs.eqs.begin()),
             std::make_move_iterator(bs.eqs.end()));
}

template <typename Int>
void PresburgerBasicSet<Int>::updateFromSimplex(const Simplex<Int> &simplex) {
  if (simplex.isEmpty()) {
    // maybeIsEmpty = true;
    return;
  }

  unsigned simplexEqsOffset = getNumInequalities();
  for (unsigned i = 0, ineqsIndex = 0; i < simplexEqsOffset; ++i) {
    if (simplex.isMarkedRedundant(i)) {
      removeInequality(ineqsIndex);
      continue;
    }
    if (simplex.constraintIsEquality(i)) {
      addEquality(getInequality(ineqsIndex).getCoeffs());
      removeInequality(ineqsIndex);
      continue;
    }
    ++ineqsIndex;
  }

  assert((simplex.numConstraints() - simplexEqsOffset) % 2 == 0 &&
         "expecting simplex to contain two ineqs for each eq");

  for (unsigned i = simplexEqsOffset, eqsIndex = 0;
       i < simplex.numConstraints(); i += 2) {
    if (simplex.isMarkedRedundant(i) && simplex.isMarkedRedundant(i + 1)) {
      removeEquality(eqsIndex);
      continue;
    }
    ++eqsIndex;
  }
}

template <typename Int>
void PresburgerBasicSet<Int>::print(raw_ostream &os) const {
  printPresburgerBasicSet(os, *this);
}

template <typename Int>
void PresburgerBasicSet<Int>::dump() const {
  print(llvm::errs());
  llvm::errs() << '\n';
}

template <typename Int>
void PresburgerBasicSet<Int>::dumpCoeffs() const {
  llvm::errs() << "nDim = " << nDim << ", nSym = " << nParam
               << ", nExist = " << nExist << ", nDiv = " << divs.size() << "\n";
  llvm::errs() << "nTotalDims = " << getNumTotalDims() << "\n";
  llvm::errs() << "nIneqs = " << ineqs.size() << '\n';
  for (auto &ineq : ineqs) {
    ineq.dumpCoeffs();
  }
  llvm::errs() << "nEqs = " << eqs.size() << '\n';
  for (auto &eq : eqs) {
    eq.dumpCoeffs();
  }
  llvm::errs() << "nDivs = " << divs.size() << '\n';
  for (auto &div : divs) {
    div.dumpCoeffs();
  }
}

template <typename Int>
void PresburgerBasicSet<Int>::printISL(raw_ostream &os) const {
  printPresburgerBasicSetISL(os, *this);
}

template <typename Int>
void PresburgerBasicSet<Int>::dumpISL() const {
  printISL(llvm::errs());
  llvm::errs() << '\n';
}

// TODO: Improve flow
// TODO: Intersection of sets should have a simplify call at end
template <typename Int>
void PresburgerBasicSet<Int>::simplify(bool aggressive) {
  // Remove redundancy
  normalizeConstraints();
  if (nExist != 0 || divs.size() != 0)
    removeRedundantVars();

  // Divs should be normalized to be compared properly
  if (divs.size() != 0) {
    orderDivisions();
    normalizeDivisions();
    removeDuplicateDivs();
  }

  removeDuplicateConstraints();

  // Removal of duplicate divs may lead to duplicate constraints
  if (aggressive)
    removeRedundantConstraints();

  // Try to recover divisions
  recoverDivisionsFromInequalities();
  recoverDivisionsFromEqualities();

  // Recovering of divisions may cause unordered and non-normalized divisions
  orderDivisions();
  normalizeDivisions();

  // Remove constant divs
  removeDuplicateDivs();
  removeConstantDivs();

  // Remove trivial redundancy
  removeTriviallyRedundantConstraints();
  removeDuplicateConstraints();
}

template <typename Int>
void PresburgerBasicSet<Int>::removeRedundantConstraints() {
  Simplex<Int> simplex(*this);
  simplex.detectRedundant();
  this->updateFromSimplex(simplex);
}

template <typename Int>
void PresburgerBasicSet<Int>::swapDivisions(unsigned vari, unsigned varj) {
  // Swap the constraints
  std::swap(divs[vari], divs[varj]);
  DivisionConstraint<Int>::swapVariables(divs[vari], divs[varj]);

  unsigned divOffset = getDivOffset();

  // Swap the coefficents in every constraint
  for (EqualityConstraint<Int> &eq : eqs)
    eq.swapCoeffs(divOffset + vari, divOffset + varj);

  for (InequalityConstraint<Int> &ineq : ineqs)
    ineq.swapCoeffs(divOffset + vari, divOffset + varj);

  for (DivisionConstraint<Int> &div : divs)
    div.swapCoeffs(divOffset + vari, divOffset + varj);
}

template <typename Int>
unsigned PresburgerBasicSet<Int>::getDivOffset() {
  return nParam + nDim + nExist;
}

template <typename Int>
unsigned PresburgerBasicSet<Int>::getExistOffset() {
  return nParam + nDim;
}

template <typename Int>
void PresburgerBasicSet<Int>::normalizeDivisions() {
  unsigned divOffset = getDivOffset();

  for (unsigned divi = 0; divi < divs.size(); ++divi) {
    DivisionConstraint div = divs[divi];
    const ArrayRef<Int> coeffs = div.getCoeffs();
    Int denom = div.getDenominator();

    SmallVector<Int, 8> shiftCoeffs(coeffs.size(), 0),
        shiftResidue(coeffs.size(), 0);

    for (unsigned i = 0; i < coeffs.size(); ++i) {
      Int coeff = coeffs[i];
      Int newCoeff = coeff;

      // Shift the coefficent to be in the range (-d/2, d/2]
      if ((-denom >= (Int)2 * coeff) || ((Int)2 * coeff > denom)) {
        newCoeff = ((coeff % denom) + denom) % denom;
        if ((Int)2 * (coeff % denom) > denom)
          newCoeff -= denom;

        shiftCoeffs[i] = newCoeff - coeff;
        shiftResidue[i] = (coeff - newCoeff) / denom;
      }
    }

    // Get index of the current division
    unsigned divDimension = divOffset + divi;

    // Shift all constraints by the shifts calculated above
    for (EqualityConstraint<Int> &eq : eqs)
      eq.shiftCoeffs(shiftResidue, eq.getCoeffs()[divDimension]);

    for (InequalityConstraint<Int> &ineq : ineqs)
      ineq.shiftCoeffs(shiftResidue, ineq.getCoeffs()[divDimension]);

    // Ordering of divs ensures that a division is only dependent
    // on divs before it.
    for (unsigned i = divi + 1; i < divs.size(); ++i)
      divs[i].shiftCoeffs(shiftResidue, divs[i].getCoeffs()[divDimension]);

    // Shift the current division by the extra coefficients
    divs[divi].shiftCoeffs(shiftCoeffs, 1);
  }

  // Take out gcd
  for (DivisionConstraint<Int> &div : divs)
    div.removeCommonFactor();
}

template <typename Int>
void PresburgerBasicSet<Int>::orderDivisions() {
  unsigned divOffset = getDivOffset();
  unsigned nDivs = divs.size();

  for (unsigned i = 0; i < nDivs;) {
    const ArrayRef<Int> &coeffs = divs[i].getCoeffs();
    bool foundDependency = false;

    // Get the first division on which this division is dependent
    for (unsigned j = i + 1; j < nDivs; ++j) {
      if (coeffs[j + divOffset] != 0) {
        foundDependency = true;
        swapDivisions(i, j);
        break;
      }
    }

    if (!foundDependency)
      ++i;
  }
}

template <typename Int>
bool PresburgerBasicSet<Int>::redundantVar(unsigned var) {
  for (auto &con : eqs) {
    if (con.getCoeffs()[var] != 0)
      return false;
  }

  for (auto &con : ineqs) {
    if (con.getCoeffs()[var] != 0)
      return false;
  }

  for (auto &con : divs) {
    if (con.getCoeffs()[var] != 0)
      return false;
  }
  return true;
}

template <typename Int>
SmallVector<Int, 8>
PresburgerBasicSet<Int>::copyWithNonRedundant(std::vector<unsigned> &nrExists,
                                        std::vector<unsigned> &nrDiv,
                                         const ArrayRef<Int> &ogCoeffs) {
  SmallVector<Int, 8> newCoeffs;

  for (unsigned i = 0; i < nDim + nParam; ++i)
    newCoeffs.push_back(ogCoeffs[i]);
  for (unsigned i : nrExists)
    newCoeffs.push_back(ogCoeffs[i]);
  for (unsigned i : nrDiv)
    newCoeffs.push_back(ogCoeffs[i]);
  newCoeffs.push_back(ogCoeffs.back());

  return newCoeffs;
}

template <typename Int>
void PresburgerBasicSet<Int>::removeRedundantVars() {
  std::vector<unsigned> nonRedundantExist, nonRedundantDivs;

  // Check for redundant existentials
  unsigned existOffset = getExistOffset();

  for (unsigned i = 0; i < nExist; ++i) {
    unsigned var = existOffset + i;
    if (!redundantVar(var))
      nonRedundantExist.push_back(var);
  }

  // Check for redundant divisions
  unsigned divOffset = getDivOffset();

  for (unsigned i = 0; i < divs.size(); ++i) {
    unsigned var = divOffset + i;
    if (!redundantVar(var))
      nonRedundantDivs.push_back(var);
  }

  SmallVector<InequalityConstraint<Int>, 8> newIneqs;
  SmallVector<EqualityConstraint<Int>, 8> newEqs;
  SmallVector<DivisionConstraint<Int>, 8> newDivs;

  for (InequalityConstraint<Int> &ineq : ineqs) {
    SmallVector<Int, 8> coeffs = copyWithNonRedundant(
        nonRedundantExist, nonRedundantDivs, ineq.getCoeffs());

    newIneqs.push_back(InequalityConstraint<Int>(coeffs));
  }

  for (EqualityConstraint<Int> &eq : eqs) {
    SmallVector<Int, 8> coeffs = copyWithNonRedundant(
        nonRedundantExist, nonRedundantDivs, eq.getCoeffs());

    newEqs.push_back(EqualityConstraint<Int>(coeffs));
  }

  unsigned variableOffset = nDim + nParam + nonRedundantExist.size();
  for (unsigned i : nonRedundantDivs) {
    DivisionConstraint<Int> div = divs[i - divOffset];

    SmallVector<Int, 8> coeffs = copyWithNonRedundant(
        nonRedundantExist, nonRedundantDivs, div.getCoeffs());

    newDivs.push_back(DivisionConstraint<Int>(coeffs, div.getDenominator(),
                                         variableOffset + newDivs.size()));
  }

  // Assign new vectors
  ineqs = newIneqs;
  eqs = newEqs;
  divs = newDivs;

  // Change dimensions
  nExist = nonRedundantExist.size();
}

template <typename Int>
bool PresburgerBasicSet<Int>::alignDivs(PresburgerBasicSet<Int> &bs1,
                                        PresburgerBasicSet<Int> &bs2,
                                        bool preserve) {

  // Assert that bs1 has more divisions than bs2
  if (bs1.getNumDivs() < bs2.getNumDivs())
    return alignDivs(bs2, bs1, preserve);

  // Append extra existentials if preserve is not set
  if (preserve && bs1.nExist != bs2.nExist)
    return false;
  if (bs1.nExist > bs2.nExist) {
    bs2.appendExistentialDimensions(bs1.nExist - bs2.nExist);
  } else {
    bs1.appendExistentialDimensions(bs2.nExist - bs1.nExist);
  }

  // Add the extra divisions from bs1 to bs2
  unsigned extraDivs = bs1.divs.size() - bs2.divs.size();
  bs2.insertDimensions(bs2.getDivOffset() + bs2.getNumDivs(), extraDivs);
  for (unsigned i = 0; i < extraDivs; ++i) {
    DivisionConstraint<Int> &div = bs1.divs[bs2.getNumDivs()];
    bs2.divs.push_back(div);
  }

  // TODO: Does there exist a better way to match divs?
  // Loop over divs and find the equal ones
  // This requires that divs in divs1 are ordered, which was ensured in
  // simplify calls.
  //
  // Ordering ensures that divisions at index will not depend on any division
  // at index >= i. This allows us to check same divisions easier.
  //
  // Ordering makes swapping easier since if two divisions at i, j match (j >=
  // i), any divisions at index >= i cannot depend on these two divisions
  // This ensures that any further swap will not affect these divisions in any
  // way.
  for (unsigned i = 0; i < bs1.divs.size(); ++i) {
    bool foundMatch = false;
    for (unsigned j = i; j < bs2.divs.size(); ++j) {
      if (DivisionConstraint<Int>::sameDivision(bs1.divs[i], bs2.divs[j])) {
        foundMatch = true;
        if (i != j) {
          bs2.swapDivisions(i, j);
          break;
        }
      }
    }

    // Match not found, convert to existential if preserve is false
    // Convert by swapping this division with this with the first division,
    // removing this division, and incrementing nExist.
    //
    // This part leverages the order of variables in coefficients: Existentials,
    // Divisions, Constant
    if (!foundMatch) {

      // Cannot convert to existential if preserve is set
      if (preserve)
        return false;

      // Add division inequalties
      bs1.addInequality(bs1.divs[i].getInequalityUpperBound().getCoeffs());
      bs1.addInequality(bs1.divs[i].getInequalityLowerBound().getCoeffs());
      bs2.addInequality(bs2.divs[i].getInequalityUpperBound().getCoeffs());
      bs2.addInequality(bs2.divs[i].getInequalityLowerBound().getCoeffs());

      // Swap the divisions
      bs1.swapDivisions(0, i);
      bs2.swapDivisions(0, i);

      bs1.divs.erase(bs1.divs.begin());
      bs2.divs.erase(bs2.divs.begin());

      // Div deleted before this index
      i--;

      bs1.nExist++;
      bs2.nExist++;
    }
  }

  return true;
}

template <typename Int>
void PresburgerBasicSet<Int>::normalizeConstraints() {
  for (EqualityConstraint<Int> &eq : eqs)
    eq.normalizeCoeffs();
  for (InequalityConstraint<Int> &ineq : ineqs)
    ineq.normalizeCoeffs();
}

/// Creates div coefficents using coefficients "ineq" of a lower bound
/// inequality for the existential at index "varIdx"
template <typename Int>
SmallVector<Int, 8> createDivFromLowerBound(const ArrayRef<Int> &ineq,
                                                unsigned varIdx) {
  // Remove existential from coefficents
  SmallVector<Int, 8> newCoeffs(ineq.size());
  for (unsigned i = 0; i < ineq.size(); ++i)
    newCoeffs[i] = -ineq[i];
  newCoeffs[varIdx] = 0;

  // Add (d - 1) to coefficents where d is the denominator
  newCoeffs.back() += ineq[varIdx] - 1;

  return newCoeffs;
}

/// Return whether any division depends variable at index "varIdx"
template <typename Int>
bool divsDependOnExist(SmallVector<DivisionConstraint<Int>, 8> &divs, unsigned varIdx) {
  for (unsigned i = 0; i < divs.size(); ++i) {
    if (divs[i].getCoeffs()[varIdx] != 0)
      return true;
  }
  return false;
}

/// Return whether constraint "con" depends on existentials other than "exist"
template <typename Int>
bool coeffDependsOnExist(Constraint<Int> &con, unsigned exist,
                         unsigned existOffset, unsigned nExist) {

  const ArrayRef<Int> &coeffs = con.getCoeffs();
  for (unsigned i = 0; i < nExist; ++i) {
    if (i == exist)
      continue;

    if (coeffs[i + existOffset] != 0)
      return true;
  }

  return false;
}

template <typename Int>
void PresburgerBasicSet<Int>::recoverDivisionsFromInequalities() {
  for (unsigned k = 0; k < ineqs.size(); ++k) {
    for (unsigned l = k + 1; l < ineqs.size(); ++l) {
      const ArrayRef<Int> &coeffs1 = ineqs[k].getCoeffs();
      const ArrayRef<Int> &coeffs2 = ineqs[l].getCoeffs();

      bool oppositeCoeffs = true;
      for (unsigned i = 0; i < coeffs1.size() - 1; ++i) {
        if (coeffs1[i] != -coeffs2[i]) {
          oppositeCoeffs = false;
          break;
        }
      }

      if (!oppositeCoeffs)
        continue;

      // Sum of constants < 0 : Denominator cannot be zero
      // Sum of constants = 0 : Inequalities represent equalities
      // Sum of constants > 0 : Inequalities may be converted to divisions
      Int constantSum = coeffs1.back() + coeffs2.back();
      if (constantSum < 0)
        continue;
      if (constantSum == 0) {
        addEquality(coeffs1);

        // Order of remove of inequalities is important. 
        // Inequality with greater index should be removed first.
        removeInequality(l);
        removeInequality(k);

        // Decrement k to reflect deletion of inequality
        k--;
        continue;
      }

      unsigned existOffset = getExistOffset();
      for (unsigned exist = 0; exist < nExist; ++exist) {
        // Check if this existential can be recovered from these two divisions
        // 1. The exist should appear in the inequalities?
        // 2. coefficent of the exist in the inequalities should be strictly
        //    greater than sum of constants of inequalities?
        // 3. The inequalities should not depend on any other exist to prevent
        //    circular division definations
        // 4. Any other division should not be defined in terms of this
        //    exist to prevent circular division definations
        //
        unsigned existIdx = existOffset + exist;
        if (coeffs1[existIdx] == 0)
          continue;
        if (constantSum >= std::abs(coeffs1[existIdx]))
          continue;
        if (coeffDependsOnExist(ineqs[k], exist, existOffset, nExist))
          continue;
        if (divsDependOnExist(divs, existIdx)) 
          continue;

        // Convert existential to division based on lower bound inequality
        SmallVector<Int, 8> newCoeffs;
        if (coeffs1[existOffset + exist] > 0)
          newCoeffs = createDivFromLowerBound(coeffs1, existIdx);
        else
          newCoeffs = createDivFromLowerBound(coeffs2, existIdx);

        if (constantSum == std::abs(coeffs1[existIdx]) - 1) {
          removeInequality(l);
          removeInequality(k);
        }

        // Insert the new division at starting of divs
        DivisionConstraint<Int> newDiv(newCoeffs, std::abs(coeffs1[existIdx]),
                                  getDivOffset() - 1);
        divs.insert(divs.begin(), newDiv);

        // Swap first and last existential
        // The last existential is treated as a division after this change
        for (auto &con : eqs)
          con.swapCoeffs(exist + existOffset, existOffset + nExist - 1);
        for (auto &con : ineqs)
          con.swapCoeffs(exist + existOffset, existOffset + nExist - 1);
        for (auto &con : divs)
          con.swapCoeffs(exist + existOffset, existOffset + nExist - 1);

        // Try using these two inequalities again for some other existential
        l--;

        // Reduce number of existentials and repeat again
        nExist--;
        break;

        // TODO: One of these inequalities is not needed after finding this.
        //       Remove it.
        // TODO: Remove both of these inequalities if the division exactly
        // matches.
      }
    }
  }
}

/// Convert given equality constraint to a division
/// The equality constraints are of the form
///
///          f(x) + n e >= 0
///
/// The division obtained from this if of the form
///
///          e = [-f(x) / n]
///
/// Returns the coefficents for the new division
template <typename Int>
SmallVector<Int, 8> createDivisionFromEq(const ArrayRef<Int> &coeffs,
                                             unsigned varIdx) {
  SmallVector<Int, 8> newCoeffs(coeffs.size());
  for (unsigned i = 0; i < coeffs.size(); ++i)
    newCoeffs[i] = coeffs[i];

  if (coeffs[varIdx] > 0) {
    for (Int &coeff : newCoeffs)
      coeff = -coeff;
  }

  newCoeffs[varIdx] = 0;

  return newCoeffs;
}

// TODO: Are the division loop prevention conditions too strict?
template <typename Int>
void PresburgerBasicSet<Int>::recoverDivisionsFromEqualities() {
  for (unsigned k = 0; k < eqs.size(); ++k) {
    const ArrayRef<Int> &coeffs = eqs[k].getCoeffs();

    // Check if equality depends only on one exitential, and get that
    // existential
    unsigned exist = nExist;
    unsigned existOffset = getExistOffset();
    bool canConvert = true;
    for (unsigned i = 0; i < nExist; ++i) {
      if (coeffs[i + existOffset] != 0) {

        if (exist != nExist) {
          canConvert = false;
          break;
        } else {
          exist = i;
        }
      }
    }
    if (!canConvert || exist == nExist)
      continue;

    // Don't convert to division if any division depends on this existential
    if (coeffDependsOnExist(eqs[k], exist, existOffset, nExist))
      continue;

    SmallVector<Int, 8> newCoeffs =
        createDivisionFromEq(coeffs, exist + existOffset);

    // Convert equality to division
    DivisionConstraint<Int> newDiv(newCoeffs, std::abs(coeffs[exist + existOffset]),
                              getDivOffset() - 1);
    divs.insert(divs.begin(), newDiv);

    // Swap first and last existential
    // The last existential is treated as a division after this change
    for (auto &con : eqs)
      con.swapCoeffs(exist + existOffset, existOffset + nExist - 1);
    for (auto &con : ineqs)
      con.swapCoeffs(exist + existOffset, existOffset + nExist - 1);
    for (auto &con : divs)
      con.swapCoeffs(exist + existOffset, existOffset + nExist - 1);

    // Reduce number of existentials
    nExist--;

    // TODO: Check if equality can be removed
  }
}

// TODO: Should hashing be used to compare divs?
template <typename Int>
void PresburgerBasicSet<Int>::removeDuplicateDivs() {
  if (divs.size() < 2)
    return;

  // Using int instead of unsigned since while doing --i, it may overflow
  for (int i = divs.size() - 1; i >= 0; --i) {
    for (int j = i - 1; j >= 0; --j) {
      if (!DivisionConstraint<Int>::sameDivision(divs[i], divs[j]))
        continue;

      unsigned divOffset = getDivOffset();

      // Merge div i to j
      for (EqualityConstraint<Int> &con : eqs)
        con.shiftCoeff(divOffset + j, con.getCoeffs()[divOffset + i]);
      for (InequalityConstraint<Int> &con : ineqs)
        con.shiftCoeff(divOffset + j, con.getCoeffs()[divOffset + i]);
      for (DivisionConstraint<Int> &con : divs)
        con.shiftCoeff(divOffset + j, con.getCoeffs()[divOffset + i]);

      // Remove constraint
      for (EqualityConstraint<Int> &con : eqs)
        con.eraseDimensions(divOffset + i, 1);
      for (InequalityConstraint<Int> &con : ineqs)
        con.eraseDimensions(divOffset + i, 1);

      for (int div = 0; div < (int)divs.size(); ++div) {
        if (div != i)
          divs[div].eraseDimensions(divOffset + i, 1);
      }

      // Remove the division defination
      divs.erase(divs.begin() + i);

      // Move to next div
      break;
    }
  }
}

/// Checks if a constraint is trivially redundant given its coefficents. If the
/// boolean equality is set, treats the constraint as an equality else as an
/// inequality.
///
/// Returns :
/// 0 --> Constraint is not redundant
/// 1 --> Constraint is redundant
/// 2 --> Constraint is invalid
template <typename Int>
static int triviallyRedundantConstraint(const ArrayRef<Int> &coeffs,
                                        bool equality) {
  // Check if constraint is constant
  for (unsigned i = 0; i < coeffs.size() - 1; ++i) {
    if (coeffs[i] != 0)
      return 0;
  }

  // Constraint is constant, check if its true
  Int constCoeff = coeffs.back();

  if (equality) {
    if (constCoeff == 0)
      return 1;
    else
      return 2;
  } else {
    if (constCoeff >= 0)
      return 1;
    else
      return 2;
  }
}

template <typename Int>
void PresburgerBasicSet<Int>::removeTriviallyRedundantConstraints() {
  for (unsigned i = 0; i < eqs.size(); ++i) {
    int result = triviallyRedundantConstraint(eqs[i].getCoeffs(), true);

    if (result == 1) {
      removeEquality(i);
      --i;
    } else if (result == 2) {
      // Constraint system is invalid, Remove all equalities and inequalities
      // other than this constraint
      Constraint<Int> invalidCon = eqs[i];

      PresburgerBasicSet<Int> newSet(getNumDims(), getNumParams(),
                                     getNumExists(), getDivisions());
      newSet.addEquality(invalidCon.getCoeffs());

      *this = newSet;
      return;
    }
  }

  for (unsigned i = 0; i < ineqs.size(); ++i) {
    int result = triviallyRedundantConstraint(ineqs[i].getCoeffs(), false);

    if (result == 1) {
      removeInequality(i);
      --i;
    } else if (result == 2) {
      // Constraint system is invalid, Remove all equalities and inequalities
      // other than this constraint
      Constraint<Int> invalidCon = ineqs[i];

      PresburgerBasicSet<Int> newSet(getNumDims(), getNumParams(),
                                     getNumExists(), getDivisions());
      newSet.addInequality(invalidCon.getCoeffs());

      *this = newSet;
      return;
    }
  }
}

template <typename Int>
void PresburgerBasicSet<Int>::removeDuplicateConstraints() {
  for (unsigned k = 0; k < ineqs.size(); ++k) {
    for (unsigned l = k + 1; l < ineqs.size(); ++l) {
      if (Constraint<Int>::sameConstraint(ineqs[k], ineqs[l])) {
        removeInequality(l);
        l--;
      }
    }
  }

  for (unsigned k = 0; k < eqs.size(); ++k) {
    for (unsigned l = k + 1; l < eqs.size(); ++l) {
      if (Constraint<Int>::sameConstraint(eqs[k], eqs[l])) {
        removeEquality(l);
        l--;
      }
    }
  }
}

template <typename Int>
void PresburgerBasicSet<Int>::removeConstantDivs() {
  for (unsigned divi = 0; divi < divs.size(); ++divi) {
    auto &div = divs[divi];
    bool isConst = true;
    ArrayRef<Int> coeffs = div.getCoeffs();
    for (unsigned i = 0, e = coeffs.size() - 1; i < e; ++i) {
      if (coeffs[i] != 0) {
        isConst = false;
        break;
      }
    }

    // Convert division to constant if it is constant
    if (isConst) {
      Int constant = floorDiv(coeffs.back(), div.getDenominator());
      unsigned divIdx = getDivOffset() + divi;

      for (auto &con : ineqs) {
        Int coeff = con.getCoeffs()[divIdx];
        con.shift(constant * coeff);
      }

      for (auto &con : eqs) {
        Int coeff = con.getCoeffs()[divIdx];
        con.shift(constant * coeff);
      }

      for (auto &con : divs) {
        Int coeff = con.getCoeffs()[divIdx];
        con.shift(constant * coeff);
      }

      // Remove division
      removeDivision(divi);
      --divi;
    }
  }
}

#endif // MLIR_ANALYSIS_PRESBURGER_BASIC_SET_IMPL_H
