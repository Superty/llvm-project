//===- PresburgerBasicSet.cpp - MLIR PresburgerBasicSet Class -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/PresburgerBasicSet.h"
#include "mlir/Analysis/Presburger/ISLPrinter.h"
#include "mlir/Analysis/Presburger/LinearTransform.h"
#include "mlir/Analysis/Presburger/ParamLexSimplex.h"
#include "mlir/Analysis/Presburger/Printer.h"
#include "mlir/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::analysis;
using namespace mlir::analysis::presburger;

PresburgerBasicSet::PresburgerBasicSet(unsigned oNDim, unsigned oNParam,
                                       unsigned oNExist,
                                       const ArrayRef<DivisionConstraint> oDivs)
    : nDim(oNDim), nParam(oNParam), nExist(oNExist) {
  divs = llvm::to_vector<8>(oDivs);
}

PresburgerBasicSet::PresburgerBasicSet(
    unsigned oNDim, unsigned oNParam, unsigned oNExist,
    const ArrayRef<InequalityConstraint> oIneqs,
    const ArrayRef<EqualityConstraint> oEqs,
    const ArrayRef<DivisionConstraint> oDivs)
    : nDim(oNDim), nParam(oNParam), nExist(oNExist) {
  ineqs = llvm::to_vector<8>(oIneqs);
  eqs = llvm::to_vector<8>(oEqs);
  divs = llvm::to_vector<8>(oDivs);
}

void PresburgerBasicSet::addInequality(ArrayRef<int64_t> coeffs) {
  ineqs.emplace_back(coeffs);
}

void PresburgerBasicSet::removeLastInequality() { ineqs.pop_back(); }

void PresburgerBasicSet::removeLastEquality() { eqs.pop_back(); }

const InequalityConstraint &
PresburgerBasicSet::getInequality(unsigned i) const {
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

void PresburgerBasicSet::removeDivision(unsigned i) {
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
    return findSampleUnbounded(cone, false);
  else
    return findSampleBounded();
}

bool PresburgerBasicSet::isIntegerEmpty() {
  // dumpISL();
  if (ineqs.empty() && eqs.empty())
    return false;
  if (!isPlainBasicSet())
    return makePlainBasicSet().isIntegerEmpty();

  PresburgerBasicSet cone = makeRecessionCone();
  if (cone.getNumEqualities() < getNumTotalDims())
    return !findSampleUnbounded(cone, true).hasValue();
  else
    return !findSampleBounded().hasValue();
}

Optional<std::pair<int64_t, SmallVector<int64_t, 8>>>
PresburgerBasicSet::findRationalSample() const {
  Simplex simplex(*this);
  if (simplex.isEmpty())
    return {};
  return simplex.findRationalSample();
}

// Returns a matrix of the constraint coefficients in the specified vector.
// If constantTerm is false, only makes matrix of coefficents, excluding
// constant
Matrix
PresburgerBasicSet::coefficientMatrixFromEqs(bool constantTerm) const {
  // TODO check if this works because of missing symbols
  Matrix result(getNumEqualities(), getNumTotalDims() + constantTerm);
  for (unsigned i = 0; i < getNumEqualities(); ++i) {
    for (unsigned j = 0; j < getNumTotalDims() + constantTerm; ++j)
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
PresburgerBasicSet::findSampleUnbounded(PresburgerBasicSet &cone,
                                        bool onlyEmptiness) const {
  auto coeffMatrix = cone.coefficientMatrixFromEqs();
  LinearTransform U =
      LinearTransform::makeTransformToColumnEchelon(std::move(coeffMatrix));
  PresburgerBasicSet transformedSet = U.postMultiplyBasicSet(*this);

  auto maybeBoundedSample = transformedSet.findBoundedDimensionsSample(cone);
  if (!maybeBoundedSample)
    return {};
  if (onlyEmptiness)
    return maybeBoundedSample;

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
void PresburgerBasicSet::projectOutUnboundedDimensions(unsigned unboundedDims) {
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
  if (count == 0)
    return;

  for (auto &ineq : ineqs)
    ineq.insertDimensions(pos, count);
  for (auto &eq : eqs)
    eq.insertDimensions(pos, count);
  for (auto &div : divs)
    div.insertDimensions(pos, count);
}

void PresburgerBasicSet::appendDivisionVariable(ArrayRef<int64_t> coeffs,
                                                int64_t denom) {
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
void PresburgerBasicSet::appendDivisionVariables(
    ArrayRef<DivisionConstraint> newDivs) {
  for (auto &div : newDivs)
    assert(div.getCoeffs().size() == getNumTotalDims() + newDivs.size() + 1);
  insertDimensions(nParam + nDim + nExist + divs.size(), newDivs.size());
  divs.insert(divs.end(), newDivs.begin(), newDivs.end());
}

void PresburgerBasicSet::prependDivisionVariables(
    ArrayRef<DivisionConstraint> newDivs) {
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

void PresburgerBasicSet::toCommonSpace(PresburgerBasicSet &a,
                                       PresburgerBasicSet &b) {
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
  assert(getNumDims() == bs.getNumDims() && "Dimensions do not match");
  assert(getNumParams() == bs.getNumParams() && "Symbols do not match");
  toCommonSpace(*this, bs);
  ineqs.insert(ineqs.end(), std::make_move_iterator(bs.ineqs.begin()),
               std::make_move_iterator(bs.ineqs.end()));
  eqs.insert(eqs.end(), std::make_move_iterator(bs.eqs.begin()),
             std::make_move_iterator(bs.eqs.end()));
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

void PresburgerBasicSet::print(raw_ostream &os) const {
  printPresburgerBasicSet(os, *this);
}

void PresburgerBasicSet::dump() const {
  print(llvm::errs());
  llvm::errs() << '\n';
}

void PresburgerBasicSet::dumpCoeffs() const {
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

void PresburgerBasicSet::printISL(raw_ostream &os) const {
  printPresburgerBasicSetISL(os, *this);
}

void PresburgerBasicSet::dumpISL() const {
  printISL(llvm::errs());
  llvm::errs() << '\n';
}
 
// TODO: Improve flow
void PresburgerBasicSet::simplify() {
  // Remove redundancy
  normalizeConstraints();
  removeRedundantVars();

  // Divs should be normilzed to be compared properly
  orderDivisions();
  normalizeDivisions();
  removeDuplicateDivs();

  // Removal of duplicate divs may lead to duplicate constraints
  removeDuplicateConstraints();
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

  // Use gauss elimination to eliminate constraints using equalities
  gaussEliminateEq();
  
  // Remove trivial redundancy
  removeTriviallyRedundantConstraints();
  removeDuplicateConstraints();
}

void PresburgerBasicSet::removeRedundantConstraints() {
  Simplex simplex(*this);
  simplex.detectRedundant();
  this->updateFromSimplex(simplex);
}

void PresburgerBasicSet::swapDivisions(unsigned vari, unsigned varj) {
  // Swap the constraints
  std::swap(divs[vari], divs[varj]);
  DivisionConstraint::swapVariables(divs[vari], divs[varj]);

  unsigned divOffset = getDivOffset();

  // Swap the coefficents in every constraint
  for (EqualityConstraint &eq : eqs)
    eq.swapCoeffs(divOffset + vari, divOffset + varj);

  for (InequalityConstraint &ineq : ineqs)
    ineq.swapCoeffs(divOffset + vari, divOffset + varj);

  for (DivisionConstraint &div : divs)
    div.swapCoeffs(divOffset + vari, divOffset + varj);
}

unsigned PresburgerBasicSet::getDivOffset() {
  return nParam + nDim + nExist;
}

unsigned PresburgerBasicSet::getExistOffset() {
  return nParam + nDim;
}

void PresburgerBasicSet::normalizeDivisions() {
  unsigned divOffset = getDivOffset();

  for (unsigned divi = 0; divi < divs.size(); ++divi) {
    DivisionConstraint div = divs[divi];
    const ArrayRef<int64_t> coeffs = div.getCoeffs();
    int64_t denom = div.getDenominator();

    SmallVector<int64_t, 8> shiftCoeffs(coeffs.size(), 0),
        shiftResidue(coeffs.size(), 0);

    for (unsigned i = 0; i < coeffs.size(); ++i) {
      int64_t coeff = coeffs[i];
      int64_t newCoeff = coeff;

      // Shift the coefficent to be in the range (-d/2, d/2]
      if ((-denom >= 2 * coeff) || (2 * coeff > denom)) {
        newCoeff = ((coeff % denom) + denom) % denom;
        if (2 * (coeff % denom) > denom)
          newCoeff -= denom;

        shiftCoeffs[i] = newCoeff - coeff;
        shiftResidue[i] = (coeff - newCoeff) / denom;
      }
    }

    // Get index of the current division
    unsigned divDimension = divOffset + divi;

    // Shift all constraints by the shifts calculated above
    for (EqualityConstraint &eq : eqs)
      eq.shiftCoeffs(shiftResidue, eq.getCoeffs()[divDimension]);

    for (InequalityConstraint &ineq : ineqs)
      ineq.shiftCoeffs(shiftResidue, ineq.getCoeffs()[divDimension]);

    // Ordering of divs ensures that a division is only dependent
    // on divs before it.
    for (unsigned i = divi + 1; i < divs.size(); ++i)
      divs[i].shiftCoeffs(shiftResidue, divs[i].getCoeffs()[divDimension]);

    // Shift the current division by the extra coefficients
    divs[divi].shiftCoeffs(shiftCoeffs, 1);
  }

  // Take out gcd
  for (DivisionConstraint &div : divs)
    div.removeCommonFactor();
}

void PresburgerBasicSet::orderDivisions() {
  unsigned divOffset = getDivOffset();
  unsigned nDivs = divs.size();

  for (unsigned i = 0; i < nDivs;) {
    const ArrayRef<int64_t> &coeffs = divs[i].getCoeffs();
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

bool PresburgerBasicSet::redundantVar(unsigned var) {
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

SmallVector<int64_t, 8>
PresburgerBasicSet::copyWithNonRedundant(std::vector<unsigned> &nrExists,
                                         std::vector<unsigned> &nrDiv,
                                         const ArrayRef<int64_t> &ogCoeffs) {
  SmallVector<int64_t, 8> newCoeffs;

  for (unsigned i = 0; i < nDim + nParam; ++i)
    newCoeffs.push_back(ogCoeffs[i]);
  for (unsigned i : nrExists)
    newCoeffs.push_back(ogCoeffs[i]);
  for (unsigned i : nrDiv)
    newCoeffs.push_back(ogCoeffs[i]);
  newCoeffs.push_back(ogCoeffs.back());

  return newCoeffs;
}

void PresburgerBasicSet::removeRedundantVars() {
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

  SmallVector<InequalityConstraint, 8> newIneqs;
  SmallVector<EqualityConstraint, 8> newEqs;
  SmallVector<DivisionConstraint, 8> newDivs;

  for (InequalityConstraint &ineq : ineqs) {
    SmallVector<int64_t, 8> coeffs = copyWithNonRedundant(
        nonRedundantExist, nonRedundantDivs, ineq.getCoeffs());

    newIneqs.push_back(InequalityConstraint(coeffs));
  }

  for (EqualityConstraint &eq : eqs) {
    SmallVector<int64_t, 8> coeffs = copyWithNonRedundant(
        nonRedundantExist, nonRedundantDivs, eq.getCoeffs());

    newEqs.push_back(EqualityConstraint(coeffs));
  }

  unsigned variableOffset = nDim + nParam + nonRedundantExist.size();
  for (unsigned i : nonRedundantDivs) {
    DivisionConstraint div = divs[i - divOffset];

    SmallVector<int64_t, 8> coeffs = copyWithNonRedundant(
        nonRedundantExist, nonRedundantDivs, div.getCoeffs());

    newDivs.push_back(DivisionConstraint(coeffs, div.getDenominator(),
                                         variableOffset + newDivs.size()));
  }

  // Assign new vectors
  ineqs = newIneqs;
  eqs = newEqs;
  divs = newDivs;

  // Change dimensions
  nExist = nonRedundantExist.size();
}

void PresburgerBasicSet::alignDivs(PresburgerBasicSet &bs1,
                                          PresburgerBasicSet &bs2) {

  // Assert that bs1 has more divisions than bs2
  if (bs1.getNumDivs() < bs2.getNumDivs()) {
    alignDivs(bs2, bs1);
    return;
  }

  // TODO: Is there a better stratergy than this ?
  // Append extra existentials
  if (bs1.nExist > bs2.nExist) {
    bs2.appendExistentialDimensions(bs1.nExist - bs2.nExist);
  } else {
    bs1.appendExistentialDimensions(bs2.nExist - bs1.nExist);
  }

  // TODO: Is there a better stratergy than this ? 
  // Add the extra divisions from bs1 to bs2
  unsigned extraDivs = bs1.divs.size() - bs2.divs.size();
  bs2.insertDimensions(bs2.getDivOffset() + bs2.getNumDivs(), extraDivs);
  for (unsigned i = 0; i < extraDivs; ++i) {
    DivisionConstraint &div = bs1.divs[bs2.getNumDivs()];
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
      if (DivisionConstraint::sameDivision(bs1.divs[i], bs2.divs[j])) {
        foundMatch = true;
        if (i != j) {
          bs2.swapDivisions(i, j);
          break;
        }
      }
    }

    // Match not found, convert to existential
    // Convert by swapping this division with this with the first division,
    // removing this division, and incrementing nExist.
    //
    // This part leverages the order of variables in coefficients: Existentials,
    // Divisions, Constant
    if (!foundMatch) {
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
}

void PresburgerBasicSet::normalizeConstraints() {
  for (EqualityConstraint &eq : eqs)
    eq.normalizeCoeffs();
  for (InequalityConstraint &ineq : ineqs)
    ineq.normalizeCoeffs();
}

/// Creates div coefficents using coefficients "ineq" of a lower bound
/// inequality for the existential at index "varIdx"
SmallVector<int64_t, 8> createDivFromLowerBound(const ArrayRef<int64_t> &ineq,
                                                unsigned varIdx) {
  // Remove existential from coefficents
  SmallVector<int64_t, 8> newCoeffs(ineq.size());
  for (unsigned i = 0; i < ineq.size(); ++i)
    newCoeffs[i] = -ineq[i];
  newCoeffs[varIdx] = 0;

  // Add (d - 1) to coefficents where d is the denominator
  newCoeffs.back() += ineq[varIdx] - 1;

  return newCoeffs;
}

/// Return whether any division depends variable at index "varIdx"
bool divsDependOnExist(SmallVector<DivisionConstraint, 8> &divs, unsigned varIdx) {
  for (unsigned i = 0; i < divs.size(); ++i) {
    if (divs[i].getCoeffs()[varIdx] != 0)
      return true;
  }
  return false;
}

/// Return whether constraint "con" depends on existentials other than "exist"
bool coeffDependsOnExist(Constraint &con, unsigned exist,
                         unsigned existOffset, unsigned nExist) {

  const ArrayRef<int64_t> &coeffs = con.getCoeffs();
  for (unsigned i = 0; i < nExist; ++i) {
    if (i == exist)
      continue;

    if (coeffs[i + existOffset] != 0)
      return true;
  }

  return false;
}

void PresburgerBasicSet::recoverDivisionsFromInequalities() {
  for (unsigned k = 0; k < ineqs.size(); ++k) {
    for (unsigned l = k + 1; l < ineqs.size(); ++l) {
      const ArrayRef<int64_t> &coeffs1 = ineqs[k].getCoeffs();
      const ArrayRef<int64_t> &coeffs2 = ineqs[l].getCoeffs();

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
      int64_t constantSum = coeffs1.back() + coeffs2.back();
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
        SmallVector<int64_t, 8> newCoeffs;
        if (coeffs1[existOffset + exist] > 0)
          newCoeffs = createDivFromLowerBound(coeffs1, existIdx);
        else
          newCoeffs = createDivFromLowerBound(coeffs2, existIdx);

        // Insert the new division at starting of divs
        DivisionConstraint newDiv(newCoeffs, std::abs(coeffs1[existIdx]),
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
SmallVector<int64_t, 8> createDivisionFromEq(const ArrayRef<int64_t> &coeffs,
                                             unsigned varIdx) {
  SmallVector<int64_t, 8> newCoeffs(coeffs.size());
  for (unsigned i = 0; i < coeffs.size(); ++i)
    newCoeffs[i] = coeffs[i];

  if (coeffs[varIdx] > 0) {
    for (int64_t &coeff : newCoeffs)
      coeff = -coeff;
  }

  newCoeffs[varIdx] = 0;

  return newCoeffs;
}

// TODO: Are the division loop prevention conditions too strict?
void PresburgerBasicSet::recoverDivisionsFromEqualities() {
  for (unsigned k = 0; k < eqs.size(); ++k) {
    const ArrayRef<int64_t> &coeffs = eqs[k].getCoeffs();

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

    SmallVector<int64_t, 8> newCoeffs =
        createDivisionFromEq(coeffs, exist + existOffset);

    // Convert equality to division
    DivisionConstraint newDiv(newCoeffs, std::abs(coeffs[exist + existOffset]),
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
void PresburgerBasicSet::removeDuplicateDivs() {
  if (divs.size() < 2)
    return;

  // Using int instead of unsigned since while doing --i, it may overflow
  for (int i = divs.size() - 1; i >= 0; --i) {
    for (int j = i - 1; j >= 0; --j) {
      if (!DivisionConstraint::sameDivision(divs[i], divs[j]))
        continue;

      unsigned divOffset = getDivOffset();

      // Merge div i to j
      for (EqualityConstraint &con : eqs)
        con.shiftCoeff(divOffset + j, con.getCoeffs()[divOffset + i]);
      for (InequalityConstraint &con : ineqs)
        con.shiftCoeff(divOffset + j, con.getCoeffs()[divOffset + i]);
      for (DivisionConstraint &con : divs)
        con.shiftCoeff(divOffset + j, con.getCoeffs()[divOffset + i]);

      // Remove constraint
      for (EqualityConstraint &con : eqs)
        con.eraseDimensions(divOffset + i, 1);
      for (InequalityConstraint &con : ineqs)
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

void PresburgerBasicSet::convertDimsToExists(unsigned l, unsigned r) {
  assert(l <= r && r <= getNumDims() &&
         "Cannot convert non dimensions to existentials");

  unsigned numMoved = r - l;
  nDim -= numMoved;
  nExist += numMoved;

  for (auto &ineq : ineqs) {
    Constraint old = ineq;
    ineq.eraseDimensions(l, numMoved);
    ineq.insertDimensions(getNumDims() + getNumParams(), numMoved);
    for (unsigned i = l; i < r; ++i)
      ineq.setCoeff(getExistOffset() + i - l, old.getCoeffs()[i]);
  }
  for (auto &eq : eqs) {
    Constraint old = eq;
    eq.eraseDimensions(l, numMoved);
    eq.insertDimensions(getNumDims() + getNumParams(), numMoved);
    for (unsigned i = l; i < r; ++i)
      eq.setCoeff(getExistOffset() + i - l, old.getCoeffs()[i]);
  }
  for (auto &div : divs) {
    Constraint old = div;
    div.eraseDimensions(l, numMoved);
    div.insertDimensions(getNumDims() + getNumParams(), numMoved);
    for (unsigned i = l; i < r; ++i)
      div.setCoeff(getExistOffset() + i - l, old.getCoeffs()[i]);
  }
}

void PresburgerBasicSet::removeConstantDivs() {
  for (unsigned divi = 0; divi < divs.size(); ++divi) {
    auto &div = divs[divi];
    bool isConst = true;
    ArrayRef<int64_t> coeffs = div.getCoeffs();
    for (unsigned i = 0, e = coeffs.size() - 1; i < e; ++i) {
      if (coeffs[i] != 0) {
        isConst = false;
        break;
      }
    }

    // Convert division to constant if it is constant
    if (isConst) {
      int64_t constant = floorDiv(coeffs.back(), div.getDenominator()); 
      unsigned divIdx = getDivOffset() + divi;

      for (auto &con: ineqs) {
        int64_t coeff = con.getCoeffs()[divIdx];
        con.shift(constant * coeff);
      }

      for (auto &con: eqs) {
        int64_t coeff = con.getCoeffs()[divIdx];
        con.shift(constant * coeff);
      }

     for (auto &con: divs) {
        int64_t coeff = con.getCoeffs()[divIdx];
        con.shift(constant * coeff);
      }

      // Remove division
      removeDivision(divi); 
      --divi;
    }
  }
}

// TODO: Implement gauss elimination of equalities to simplify all constraints.
void PresburgerBasicSet::gaussEliminateEq() {
}

/// Checks if a constraint is trivially redundant given its coefficents. If the
/// boolean equality is set, treats the constraint as an equality else as an
/// inequality.
///
/// Returns : 
/// 0 --> Constraint is not redundant
/// 1 --> Constraint is redundant
/// 2 --> Constraint is invalid
static int triviallyRedundantConstraint(const ArrayRef<int64_t> &coeffs,
                                         bool equality) {
  // Check if constraint is constant
  for (unsigned i = 0; i < coeffs.size() - 1; ++i) {
    if (coeffs[i] != 0)
      return 0;
  }

  // Constraint is constant, check if its true
  int64_t constCoeff = coeffs.back();

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

void PresburgerBasicSet::removeTriviallyRedundantConstraints() {
  for (unsigned i = 0; i < eqs.size(); ++i) {
    int result = triviallyRedundantConstraint(eqs[i].getCoeffs(), true);

    if (result == 1) {
      removeEquality(i);
      --i;
    } else if (result == 2) {
      // Constraint system is invalid, Remove all equalities and inequalities
      // other than this constraint
      Constraint invalidCon = eqs[i];

      PresburgerBasicSet newSet(getNumDims(), getNumParams(), getNumExists(),
                                getDivisions());
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
      Constraint invalidCon = ineqs[i];

      PresburgerBasicSet newSet(getNumDims(), getNumParams(), getNumExists(),
                                getDivisions());
      newSet.addInequality(invalidCon.getCoeffs());

      *this = newSet;
      return;
    }
  }
}

void PresburgerBasicSet::removeDuplicateConstraints() {
  for (unsigned k = 0; k < ineqs.size(); ++k) {
    for (unsigned l = k + 1; l < ineqs.size(); ++l) {
      if (Constraint::sameConstraint(ineqs[k], ineqs[l])) {
        removeInequality(l);
        l--;
      }
    }
  }

  for (unsigned k = 0; k < eqs.size(); ++k) {
    for (unsigned l = k + 1; l < eqs.size(); ++l) {
      if (Constraint::sameConstraint(eqs[k], eqs[l])) {
        removeEquality(l);
        l--;
      }
    }
  }
}
