//===- PresburgerBasicSet.cpp - MLIR PresburgerBasicSet Class -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/ParamLexSimplex.h"
#include "mlir/Analysis/Presburger/LinearTransform.h"
#include "mlir/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::analysis;
using namespace mlir::analysis::presburger;

void PresburgerBasicSet::addInequality(ArrayRef<int64_t> coeffs) {
  ineqs.emplace_back(coeffs);
}

void PresburgerBasicSet::removeLastInequality() {
  ineqs.pop_back();
}

void PresburgerBasicSet::removeLastEquality() {
  eqs.pop_back();
}

const InequalityConstraint &PresburgerBasicSet::getInequality(unsigned i) const {
  return ineqs[i];
}
const EqualityConstraint &PresburgerBasicSet::getEquality(unsigned i) const {
  return eqs[i];
}
ArrayRef<InequalityConstraint> PresburgerBasicSet::getInequalities() const {
  return ineqs;
}
ArrayRef<EqualityConstraint> PresburgerBasicSet::getEqualities() const {
  return eqs;
}
ArrayRef<DivisionConstraint> PresburgerBasicSet::getDivisions() const {
  return divs;
}

void PresburgerBasicSet::removeLastDivision() {
  divs.pop_back();
  for (auto &ineq : ineqs)
    ineq.removeLastDimension();
  for (auto &eq : eqs)
    eq.removeLastDimension();
  for (auto &div : divs)
    div.removeLastDimension();
}

void PresburgerBasicSet::addEquality(ArrayRef<int64_t> coeffs) {
  eqs.emplace_back(coeffs);
}

PresburgerBasicSet PresburgerBasicSet::makePlainBasicSet() const {
  PresburgerBasicSet plainBasicSet(getNumTotalDims(), 0, 0);
  plainBasicSet.ineqs = ineqs;
  plainBasicSet.eqs = eqs;
  for (const DivisionConstraint &div : divs) {
    plainBasicSet.ineqs.emplace_back(div.getInequalityLowerBound());
    plainBasicSet.ineqs.emplace_back(div.getInequalityUpperBound());
  }
  return plainBasicSet;
}

Optional<SmallVector<int64_t, 8>>
PresburgerBasicSet::findIntegerSample() const {
  if (!isPlainBasicSet())
    return makePlainBasicSet().findIntegerSample();

  PresburgerBasicSet cone = makeRecessionCone();
  if (cone.getNumEqualities() < getNumTotalDims())
    return findSampleUnbounded(cone);
  else
    return findSampleBounded();
}

Optional<std::pair<int64_t, SmallVector<int64_t, 8>>>
PresburgerBasicSet::findRationalSample() const {
  Simplex simplex(*this);
  if (simplex.isEmpty())
    return {};
  return simplex.findRationalSample();
}

// Returns a matrix of the constraint coefficients in the specified vector.
//
// This only makes a matrix of the coefficients! The constant terms are
// omitted.
Matrix PresburgerBasicSet::coefficientMatrixFromEqs() const {
  // TODO check if this works because of missing symbols
  Matrix result(getNumEqualities(), getNumTotalDims());
  for (unsigned i = 0; i < getNumEqualities(); ++i) {
    for (unsigned j = 0; j < getNumTotalDims(); ++j)
      result(i, j) = eqs[i].getCoeffs()[j];
  }
  return result;
}

bool PresburgerBasicSet::isPlainBasicSet() const {
  return nParam == 0 && nExist == 0 && divs.empty();
}

void PresburgerBasicSet::substitute(ArrayRef<int64_t> values) {
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
llvm::Optional<SmallVector<int64_t, 8>>
PresburgerBasicSet::findSampleUnbounded(PresburgerBasicSet &cone) const {
  auto coeffMatrix = cone.coefficientMatrixFromEqs();
  LinearTransform U =
      LinearTransform::makeTransformToColumnEchelon(std::move(coeffMatrix));
  PresburgerBasicSet transformedSet = U.postMultiplyBasicSet(*this);

  auto maybeBoundedSample = transformedSet.findBoundedDimensionsSample(cone);
  if (!maybeBoundedSample)
    return {};

  transformedSet.substitute(*maybeBoundedSample);

  auto maybeUnboundedSample = transformedSet.findSampleFullCone();
  if (!maybeUnboundedSample)
    return {};

  // TODO change to SmallVector!

  SmallVector<int64_t, 8> sample(*maybeBoundedSample);
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
Optional<SmallVector<int64_t, 8>> PresburgerBasicSet::findSampleFullCone() {
  // NOTE isl instead makes a recession cone, shifts the cone to some rational
  // point in the initial set, and then does the following on the shifted cone.
  // It is unclear why we need to do all that since the current basic set is
  // already the required shifted cone.
  for (unsigned i = 0, e = getNumInequalities(); i < e; ++i) {
    int64_t shift = 0;
    for (unsigned j = 0, e = getNumTotalDims(); j < e; ++j) {
      int64_t coeff = ineqs[i].getCoeffs()[j];
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
void PresburgerBasicSet::projectOutUnboundedDimensions(
    unsigned unboundedDims) {
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
    ineq.eraseDimensions(getNumTotalDims() - unboundedDims, unboundedDims);
  for (auto &eq : eqs)
    eq.eraseDimensions(getNumTotalDims() - unboundedDims, unboundedDims);
}

Optional<SmallVector<int64_t, 8>>
PresburgerBasicSet::findBoundedDimensionsSample(
    const PresburgerBasicSet &cone) const {
  assert(cone.isPlainBasicSet());
  PresburgerBasicSet boundedSet = *this;
  boundedSet.projectOutUnboundedDimensions(getNumTotalDims() -
                                           cone.getNumEqualities());
  return boundedSet.findSampleBounded();
}

Optional<SmallVector<int64_t, 8>>
PresburgerBasicSet::findSampleBounded() const {
  if (getNumTotalDims() == 0)
    return SmallVector<int64_t, 8>();

  Simplex simplex(*this);
  if (simplex.isEmpty())
    return {};

  // NOTE possible optimization for equalities. We could transform the basis
  // into one where the equalities appear as the first directions, so that
  // in the basis search recursion these immediately get assigned their
  // values.
  return simplex.findIntegerSample();
}

// We shift all the constraints to the origin, then construct a simplex and
// detect implicit equalities. If a direction was intially both upper and lower
// bounded, then this operation forces it to be equal to zero, and this gets
// detected by simplex.
PresburgerBasicSet PresburgerBasicSet::makeRecessionCone() const {
  PresburgerBasicSet cone = *this;

  // TODO: check this
  for (unsigned r = 0, e = cone.getNumEqualities(); r < e; r++)
    cone.eqs[r].shiftToOrigin();

  for (unsigned r = 0, e = cone.getNumInequalities(); r < e; r++)
    cone.ineqs[r].shiftToOrigin();

  // NOTE isl does gauss here.

  Simplex simplex(cone);
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

void PresburgerBasicSet::removeInequality(unsigned i) {
  ineqs.erase(ineqs.begin() + i, ineqs.begin() + i + 1);
}

void PresburgerBasicSet::removeEquality(unsigned i) {
  eqs.erase(eqs.begin() + i, eqs.begin() + i + 1);
}

void PresburgerBasicSet::insertDimensions(unsigned pos, unsigned count) {
  for (auto &ineq : ineqs)
    ineq.insertDimensions(pos, count);
  for (auto &eq : eqs)
    eq.insertDimensions(pos, count);
  for (auto &div : divs)
    div.insertDimensions(pos, count);
}

void PresburgerBasicSet::appendDivisionVariable(ArrayRef<int64_t> coeffs, int64_t denom) {
  divs.emplace_back(coeffs, denom, /*variable = */getNumTotalDims());
  for (auto &ineq : ineqs)
    ineq.appendDimension();
  for (auto &eq : eqs)
    eq.appendDimension();
  for (auto &div : divs)
    div.appendDimension();
}

// TODO we can make these mutable arrays and move the divs in our only use case.
void PresburgerBasicSet::appendDivisionVariables(ArrayRef<DivisionConstraint> newDivs) {
  insertDimensions(nParam + nDim + nExist + divs.size(), newDivs.size());
  divs.insert(divs.end(), newDivs.begin(), newDivs.end());
}

void PresburgerBasicSet::prependDivisionVariables(ArrayRef<DivisionConstraint> newDivs) {
  insertDimensions(nParam + nDim + nExist, newDivs.size());
  divs.insert(divs.begin(), newDivs.begin(), newDivs.end());
}

void PresburgerBasicSet::prependExistentialDimensions(unsigned count) {
  insertDimensions(nParam + nDim, count);
  nExist += count;
}

void PresburgerBasicSet::appendExistentialDimensions(unsigned count) {
  insertDimensions(nParam + nDim + nExist, count);
  nExist += count;
}

void PresburgerBasicSet::toCommonSpace(PresburgerBasicSet &a, PresburgerBasicSet &b) {
  unsigned initialANExist = a.nExist;
  a.appendExistentialDimensions(b.nExist);
  b.prependExistentialDimensions(initialANExist);

  unsigned offset = a.nParam + a.nDim + a.nExist;
  SmallVector<DivisionConstraint, 8> aDivs = a.divs, bDivs = b.divs;
  for (DivisionConstraint &div : aDivs)
    div.insertDimensions(offset + aDivs.size(), bDivs.size());
  for (DivisionConstraint &div : bDivs)
    div.insertDimensions(offset, aDivs.size());

  a.appendDivisionVariables(bDivs);
  b.prependDivisionVariables(aDivs);
}

void PresburgerBasicSet::intersect(PresburgerBasicSet bs) {
  toCommonSpace(*this, bs);
  ineqs.insert(ineqs.end(), std::make_move_iterator(bs.ineqs.begin()), std::make_move_iterator(bs.ineqs.end()));
  eqs.insert(eqs.end(), std::make_move_iterator(bs.eqs.begin()), std::make_move_iterator(bs.eqs.end()));
}

void PresburgerBasicSet::updateFromSimplex(const Simplex &simplex) {
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

void PresburgerBasicSet::dump() const {
  // auto printName = [&](unsigned idx) {
  //   assert(idx < nDim && "Out of bounds index!");
  //   if (idx > nDim - nDiv) llvm::errs() << "d" << idx - (nDim - nDiv);
  //   else if (idx > nDim - nDiv - nParam) llvm::errs() << "p" << idx - (nDim - nDiv - nParam);
  //   else llvm::errs() << "x" << idx;
  // };

  // auto printExpr = [&](ArrayRef<int64_t> expr) {
  //   for (unsigned idx = 0; idx < expr.size() - 1; ++idx) {
  //     if (expr[idx] == 0) continue;
  //     if (expr[idx] == -1) llvm::errs() << "-";
  //     else if (expr[idx] != 1) llvm::errs() << expr[idx];
  //     printName(idx);
  //     llvm::errs() << " + ";
  //   }
  //   llvm::errs() << expr.back();
  // };

  // llvm::errs() << "nDim = " << nDim << ", nParam = " << nParam << ", nDiv = " << nDiv << '\n';
  // for (unsigned i = 0; i < nDiv; ++i) {
  //   llvm::errs() << "d" << i << " = [(";
  //   printExpr(divs[i].num);
  //   llvm::errs() << ")/" << divs[i].den << "], ";
  // }
  // llvm::errs() << '\n';
  // for (auto &ineq : ineqs) {
  //   printExpr(ineq);
  //   llvm::errs() << " >= 0 and ";
  // }
  // for (auto &eq : eqs) {
  //   printExpr(eq);
  //   llvm::errs() << " == 0 and ";
  // }
}
