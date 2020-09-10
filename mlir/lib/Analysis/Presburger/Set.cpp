//===- Set.cpp - MLIR PresburgerSet Class ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Set.h"
#include "mlir/Analysis/Presburger/Simplex.h"

using namespace mlir;

PresburgerSet::PresburgerSet(FlatAffineConstraints fac)
    : nDim(fac.getNumDimIds()), nSym(fac.getNumSymbolIds()) {
  addFlatAffineConstraints(fac);
}

unsigned PresburgerSet::getNumFACs() const {
  return flatAffineConstraints.size();
}

unsigned PresburgerSet::getNumDims() const { return nDim; }

unsigned PresburgerSet::getNumSyms() const { return nSym; }

ArrayRef<FlatAffineConstraints>
PresburgerSet::getAllFlatAffineConstraints() const {
  return flatAffineConstraints;
}

const FlatAffineConstraints &
PresburgerSet::getFlatAffineConstraints(unsigned index) const {
  assert(index < flatAffineConstraints.size() && "index out of bounds!");
  return flatAffineConstraints[index];
}

/// Assert that the FlatAffineConstraints and PresburgerSet live in
/// compatible spaces.
static void assertDimensionsCompatible(const FlatAffineConstraints &fac,
                                       const PresburgerSet &set) {
  assert(fac.getNumDimIds() == set.getNumDims() &&
         "Number of dimensions of the FlatAffineConstraints and PresburgerSet"
         "do not match!");
  assert(fac.getNumSymbolIds() == set.getNumSyms() &&
         "Number of symbols of the FlatAffineConstraints and PresburgerSet"
         "do not match!");
}

/// Assert that the two PresburgerSets live in compatible spaces.
static void assertDimensionsCompatible(const PresburgerSet &setA,
                                       const PresburgerSet &setB) {
  assert(setA.getNumDims() == setB.getNumDims() &&
         "Number of dimensions of the PresburgerSets do not match!");
  assert(setA.getNumSyms() == setB.getNumSyms() &&
         "Number of symbols of the PresburgerSets do not match!");
}

/// Add a FAC to the union.
void PresburgerSet::addFlatAffineConstraints(const FlatAffineConstraints &fac) {
  assertDimensionsCompatible(fac, *this);
  flatAffineConstraints.push_back(fac);
}

/// Union the current set with the given set.
///
/// This is accomplished by simply adding all the FACs of the given set to the
/// current set.
void PresburgerSet::unionSet(const PresburgerSet &set) {
  assertDimensionsCompatible(set, *this);
  for (const FlatAffineConstraints &fac : set.flatAffineConstraints)
    addFlatAffineConstraints(std::move(fac));
}

/// A point is contained in the union iff any of the parts contain the point.
bool PresburgerSet::containsPoint(ArrayRef<int64_t> point) const {
  for (const FlatAffineConstraints &fac : flatAffineConstraints) {
    if (fac.containsPoint(point))
      return true;
  }
  return false;
}

PresburgerSet PresburgerSet::makeUniverse(unsigned nDim, unsigned nSym) {
  PresburgerSet result(nDim, nSym);
  result.addFlatAffineConstraints(FlatAffineConstraints(nDim, nSym));
  return result;
}

// Compute the intersection of the two sets.
//
// We directly compute (S_1 or S_2 ...) and (T_1 or T_2 ...)
// as (S_1 and T_1) or (S_1 and T_2) or ...
void PresburgerSet::intersectSet(const PresburgerSet &set) {
  assertDimensionsCompatible(set, *this);

  PresburgerSet result(nDim, nSym);
  for (const FlatAffineConstraints &csA : flatAffineConstraints) {
    for (const FlatAffineConstraints &csB : set.flatAffineConstraints) {
      FlatAffineConstraints intersection(csA);
      intersection.append(csB);
      if (!intersection.isEmpty())
        result.addFlatAffineConstraints(std::move(intersection));
    }
  }
  *this = std::move(result);
}

/// Returns
static SmallVector<int64_t, 8> negatedCoeffs(ArrayRef<int64_t> coeffs) {
  SmallVector<int64_t, 8> negatedCoeffs;
  for (int64_t coeff : coeffs)
    negatedCoeffs.emplace_back(-coeff);
  return negatedCoeffs;
}

/// Return the complement of the given inequality.
///
/// The complement of a_1 x_1 + ... + a_n x_ + c >= 0 is
/// a_1 x_1 + ... + a_n x_ + c < 0, i.e., -a_1 x_1 - ... - a_n x_ - c - 1 >= 0.
static SmallVector<int64_t, 8> complementIneq(ArrayRef<int64_t> ineq) {
  SmallVector<int64_t, 8> coeffs;
  for (int64_t coeff : ineq)
    coeffs.emplace_back(-coeff);
  --coeffs.back();
  return coeffs;
}

/// Return the set difference b - s and accumulate the result into `result`.
/// `simplex` must correspond to b.
///
/// In the following, U denotes union, /\ denotes intersection, - denotes set
/// subtraction and ~ denotes complement.
/// Let b be the basic set and s = (U_i s_i) be the set. We want b - (U_i s_i).
///
/// Let s_i = /\_j s_ij. To compute b - s_i = b /\ ~s_i, we partition s_i based
/// on the first violated constraint:
/// ~s_i = (~s_i1) U (s_i1 /\ ~s_i2) U (s_i1 /\ s_i2 /\ ~s_i3) U ...
/// And the required result is (b /\ ~s_i1) U (b /\ s_i1 /\ ~s_i2) U ...
/// We recurse by subtracting U_{j > i} S_j from each of these parts and
/// returning the union of the results.
///
/// As a heuristic, we try adding all the constraints and check if simplex
/// says that the intersection is empty. Also, in the process we find out that
/// some constraints are redundant, which we then ignore.
static void subtractRecursively(FlatAffineConstraints &b, Simplex &simplex,
                                const PresburgerSet &s, unsigned i,
                                PresburgerSet &result) {
  if (i == s.getNumFACs()) {
    result.addFlatAffineConstraints(b);
    return;
  }
  const FlatAffineConstraints &sI = s.getFlatAffineConstraints(i);
  unsigned initialSnap = simplex.getSnapshot();
  unsigned offset = simplex.numConstraints();
  simplex.addFlatAffineConstraints(sI);

  if (simplex.isEmpty()) {
    /// b /\ s_i is empty, so b - s_i = b. We move directly to i + 1.
    simplex.rollback(initialSnap);
    subtractRecursively(b, simplex, s, i + 1, result);
    return;
  }

  simplex.detectRedundant();
  SmallVector<bool, 8> isMarkedRedundant;
  for (unsigned j = 0; j < 2 * sI.getNumEqualities() + sI.getNumInequalities();
       j++)
    isMarkedRedundant.push_back(simplex.isMarkedRedundant(offset + j));

  simplex.rollback(initialSnap);

  auto recurseWithInequality = [&, i](ArrayRef<int64_t> ineq) {
    size_t snap = simplex.getSnapshot();
    b.addInequality(ineq);
    simplex.addInequality(ineq);
    subtractRecursively(b, simplex, s, i + 1, result);
    b.removeInequality(b.getNumInequalities() - 1);
    simplex.rollback(snap);
  };

  auto processInequality = [&](ArrayRef<int64_t> ineq) {
    recurseWithInequality(complementIneq(ineq));
    b.addInequality(ineq);
    simplex.addInequality(ineq);
  };

  unsigned originalNumIneqs = b.getNumInequalities();
  unsigned originalNumEqs = b.getNumEqualities();

  for (unsigned j = 0; j < sI.getNumInequalities(); j++) {
    if (isMarkedRedundant[j])
      continue;
    processInequality(sI.getInequality(j));
  }

  offset = sI.getNumInequalities();
  for (unsigned j = 0, e = sI.getNumEqualities(); j < e; ++j) {
    const ArrayRef<int64_t> &coeffs = sI.getEquality(j);
    // Same as the above loop for inequalities, done once each for the positive
    // and negative inequalities.
    if (!isMarkedRedundant[offset + 2 * j])
      processInequality(coeffs);
    if (!isMarkedRedundant[offset + 2 * j + 1])
      processInequality(negatedCoeffs(coeffs));
  }

  for (unsigned i = b.getNumInequalities(); i > originalNumIneqs; --i)
    b.removeInequality(i - 1);

  for (unsigned i = b.getNumEqualities(); i > originalNumEqs; --i)
    b.removeEquality(i - 1);

  simplex.rollback(initialSnap);
}

/// Returns the set difference fac - set.
PresburgerSet PresburgerSet::subtract(FlatAffineConstraints &fac,
                                      const PresburgerSet &set) {
  assertDimensionsCompatible(fac, set);
  if (fac.isEmptyByGCDTest())
    return PresburgerSet(fac.getNumDimIds(), fac.getNumSymbolIds());

  PresburgerSet result(fac.getNumDimIds(), fac.getNumSymbolIds());
  Simplex simplex(fac);
  subtractRecursively(fac, simplex, set, 0, result);
  return result;
}

PresburgerSet PresburgerSet::subtract(FlatAffineConstraints &&fac,
                                      const PresburgerSet &set) {
  FlatAffineConstraints lvalue(fac);
  return subtract(fac, set);
}

PresburgerSet PresburgerSet::complement(const PresburgerSet &set) {
  // The complement of S is the universe of all points, minus S.
  return subtract(FlatAffineConstraints(set.getNumDims(), set.getNumSyms()),
                  set);
}

/// Subtracts the set from the current set.
///
/// We compute (U_i t_i) - (U_i set_i) as U_i (t_i - U_i set_i).
void PresburgerSet::subtract(const PresburgerSet &set) {
  assertDimensionsCompatible(set, *this);
  PresburgerSet result(nDim, nSym);
  for (FlatAffineConstraints &c : flatAffineConstraints)
    result.unionSet(subtract(c, set));
  *this = result;
}

/// Return true if all the sets in the union are known to be integer empty,
/// false otherwise.
bool PresburgerSet::isIntegerEmpty() const {
  assert(nSym == 0 && "findIntegerSample is intended for non-symbolic sets");
  // The set is empty iff all of the disjuncts are empty.
  for (const FlatAffineConstraints &fac : flatAffineConstraints) {
    if (!fac.isIntegerEmpty())
      return false;
  }
  return true;
}

bool PresburgerSet::findIntegerSample(SmallVectorImpl<int64_t> &sample) {
  assert(nSym == 0 && "findIntegerSample is intended for non-symbolic sets");
  // A sample exists iff any of the disjuncts containts a sample.
  for (FlatAffineConstraints &fac : flatAffineConstraints) {
    if (Optional<SmallVector<int64_t, 8>> opt = fac.findIntegerSample()) {
      sample = std::move(*opt);
      return true;
    }
  }
  return false;
}

void PresburgerSet::print(raw_ostream &os) const {
  os << getNumFACs() << " FlatAffineConstraints:\n";
  for (const FlatAffineConstraints &fac : flatAffineConstraints) {
    fac.print(os);
    os << '\n';
  }
}

void PresburgerSet::dump() const { print(llvm::errs()); }
