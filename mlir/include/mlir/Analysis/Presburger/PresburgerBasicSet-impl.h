//===- PresburgerBasicSet.cpp - MLIR PresburgerBasicSet Class -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/ISLPrinter.h"
#include "mlir/Analysis/Presburger/LinearTransform.h"
#include "mlir/Analysis/Presburger/ParamLexSimplex.h"
#include "mlir/Analysis/Presburger/Printer.h"
#include "mlir/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#ifndef MLIR_ANALYSIS_PRESBURGER_BASIC_SET_IMPL_H
#define MLIR_ANALYSIS_PRESBURGER_BASIC_SET_IMPL_H

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
  if (cone.getNumEqualities() == 0 && onlyEmptiness)
    return SmallVector<Int, 8>();
  if (cone.getNumEqualities() < getNumTotalDims())
    return findSampleUnbounded(cone, onlyEmptiness);
  else
    return findSampleBounded(onlyEmptiness);
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

#endif // MLIR_ANALYSIS_PRESBURGER_BASIC_SET_IMPL_H
