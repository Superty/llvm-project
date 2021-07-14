#ifndef MLIR_ANALYSIS_PRESBURGER_SET_IMPL_H
#define MLIR_ANALYSIS_PRESBURGER_SET_IMPL_H

#include "mlir/Analysis/Presburger/Set.h"
#include "mlir/Analysis/Presburger/Constraint.h"
#include "mlir/Analysis/Presburger/ISLPrinter.h"
#include "mlir/Analysis/Presburger/ParamLexSimplex.h"
#include "mlir/Analysis/Presburger/Printer.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Analysis/Presburger/Coalesce-impl.h"


// TODO should we change this to a storage type?
using namespace mlir;
using namespace analysis::presburger;

template <typename Int>
PresburgerSet<Int>::PresburgerSet(PresburgerBasicSet<Int> cs)
    : nDim(cs.getNumDims()), nSym(cs.getNumParams()), markedEmpty(false) {
  addBasicSet(cs);
}

template <typename Int>
template <typename OInt>
PresburgerSet<Int>::PresburgerSet(const PresburgerSet<OInt> &o) : nDim(o.nDim), nSym(o.nSym), basicSets(convert<PresburgerBasicSet<Int>>(o.basicSets)), markedEmpty(o.markedEmpty) {
  if (o.maybeSample)
    *maybeSample = convert<Int>(*o.maybeSample);
}

template <typename Int>
unsigned PresburgerSet<Int>::getNumBasicSets() const { return basicSets.size(); }

template <typename Int>
unsigned PresburgerSet<Int>::getNumDims() const { return nDim; }

template <typename Int>
unsigned PresburgerSet<Int>::getNumSyms() const { return nSym; }

template <typename Int>
bool PresburgerSet<Int>::isMarkedEmpty() const {
  return markedEmpty || basicSets.empty();
}

template <typename Int>
bool PresburgerSet<Int>::isUniverse() const {
  if (markedEmpty || basicSets.empty())
    return false;
  for (const PresburgerBasicSet<Int> &bs : basicSets) {
    if (bs.getNumInequalities() == 0 && bs.getNumEqualities() == 0)
      return true;
  }
  return false;
}

template <typename Int>
const SmallVector<PresburgerBasicSet<Int>, 4> &PresburgerSet<Int>::getBasicSets() const {
  return basicSets;
}

// This is only used to check assertions
template <typename Int>
static void assertDimensionsCompatible(PresburgerBasicSet<Int> cs,
                                       PresburgerSet<Int> set) {
  assert(cs.getNumDims() == set.getNumDims() &&
         cs.getNumParams() == set.getNumSyms() &&
         "Dimensionalities of PresburgerBasicSet<Int> and PresburgerSet<Int> do not "
         "match");
}

template <typename Int>
static void assertDimensionsCompatible(PresburgerSet<Int> set1, PresburgerSet<Int> set2) {
  assert(set1.getNumDims() == set2.getNumDims() &&
         set1.getNumSyms() == set2.getNumSyms() &&
         "Dimensionalities of PresburgerSet<Int>s do not match");
}

template <typename Int>
void PresburgerSet<Int>::addBasicSet(const PresburgerBasicSet<Int> &cs) {
  assertDimensionsCompatible(cs, *this);

  markedEmpty = false;
  basicSets.push_back(cs);
}
template <typename Int>
void PresburgerSet<Int>::reserveBasicSets(unsigned count) {
  basicSets.reserve(count);
}

template <typename Int>
void PresburgerSet<Int>::addBasicSet(PresburgerBasicSet<Int> &&cs) {
  assertDimensionsCompatible(cs, *this);

  markedEmpty = false;
  basicSets.push_back(std::move(cs));
}

template <typename Int>
void PresburgerSet<Int>::unionSet(const PresburgerSet<Int> &set) {
  assertDimensionsCompatible(set, *this);

  if (basicSets.empty()) {
    basicSets = set.basicSets;
    return;
  }

  reserveBasicSets(basicSets.size() + set.basicSets.size());
  for (const PresburgerBasicSet<Int> &cs : set.basicSets)
    addBasicSet(cs);
}

template <typename Int>
void PresburgerSet<Int>::unionSet(PresburgerSet<Int> &&set) {
  assertDimensionsCompatible(set, *this);

  basicSets.reserve(basicSets.size() + set.basicSets.size());
  for (PresburgerBasicSet<Int> &cs : set.basicSets)
    addBasicSet(std::move(cs));
}

// Compute the intersection of the two sets.
//
// We directly compute (S_1 or S_2 ...) and (T_1 or T_2 ...)
// as (S_1 and T_1) or (S_1 and T_2) or ...
template <typename Int>
void PresburgerSet<Int>::intersectSet(const PresburgerSet<Int> &set) {
  assertDimensionsCompatible(set, *this);

  if (markedEmpty)
    return;
  if (set.markedEmpty) {
    markedEmpty = true;
    return;
  }

  if (set.isUniverse())
    return;
  if (isUniverse()) {
    *this = set;
    return;
  }

  PresburgerSet<Int> result(nDim, nSym, true);
  result.reserveBasicSets(basicSets.size() * set.basicSets.size());
  for (const PresburgerBasicSet<Int> &cs1 : basicSets) {
    for (const PresburgerBasicSet<Int> &cs2 : set.basicSets) {
      PresburgerBasicSet<Int> intersection(cs1);
      intersection.intersect(cs2);
      result.addBasicSet(std::move(intersection));
    }
  }
  *this = std::move(result);
}

template <typename Int>
PresburgerSet<Int> PresburgerSet<Int>::makeEmptySet(unsigned nDim, unsigned nSym) {
  PresburgerSet<Int> result(nDim, nSym, true);
  return result;
}

/// Return `coeffs` with all the elements negated.
template <typename Int>
static SmallVector<Int, 8>
getNegatedCoeffs(ArrayRef<Int> coeffs) {
  SmallVector<Int, 8> negatedCoeffs;
  negatedCoeffs.reserve(coeffs.size());
  for (Int coeff : coeffs)
    negatedCoeffs.emplace_back(-coeff);
  return negatedCoeffs;
}

/// Return the complement of the given inequality.
///
/// The complement of a_1 x_1 + ... + a_n x_ + c >= 0 is
/// a_1 x_1 + ... + a_n x_ + c < 0, i.e., -a_1 x_1 - ... - a_n x_ - c - 1 >= 0.
template <typename Int>
static SmallVector<Int, 8>
getComplementIneq(ArrayRef<Int> ineq) {
  SmallVector<Int, 8> coeffs;
  coeffs.reserve(ineq.size());
  for (Int coeff : ineq)
    coeffs.emplace_back(-coeff);
  coeffs.back() -= 1;
  return coeffs;
}

// Return the set difference B - S and accumulate the result into `result`.
// `simplex` must correspond to B.
//
// In the following, U denotes union, /\ denotes intersection, - denotes set
// subtraction and ~ denotes complement.
// Let B be the basic set and S = (U_i S_i) be the set. We want B - (U_i S_i).
//
// Let S_i = /\_j S_ij. To compute B - S_i = B /\ ~S_i, we partition S_i based
// on the first violated constraint:
// ~S_i = (~S_i1) U (S_i1 /\ ~S_i2) U (S_i1 /\ S_i2 /\ ~S_i3) U ...
// And the required result is (B /\ ~S_i1) U (B /\ S_i1 /\ ~S_i2) U ...
// We recurse by subtracting U_{j > i} S_j from each of these parts and
// returning the union of the results.
//
// As a heuristic, we try adding all the constraints and check if simplex
// says that the intersection is empty. Also, in the process we find out that
// some constraints are redundant, which we then ignore.
template <typename Int>
void subtractRecursively(PresburgerBasicSet<Int> &b, Simplex<Int> &simplex,
                         const PresburgerSet<Int> &s, unsigned i,
                         PresburgerSet<Int> &result) {
  if (i == s.getNumBasicSets()) {
    // PresburgerBasicSet<Int> BCopy = B;
    // BCopy.simplify();
    result.addBasicSet(b);
    return;
  }
  PresburgerBasicSet<Int> oldB = b;
  PresburgerBasicSet<Int> sI = s.getBasicSets()[i];

  auto initialSnapshot = simplex.getSnapshot();
  unsigned numSIDivs = sI.getNumDivs();
  PresburgerBasicSet<Int>::toCommonSpace(b, sI);
  for (unsigned j = 0; j < numSIDivs; ++j)
    simplex.addVariable();

  for (unsigned j = sI.getNumDivs() - numSIDivs, e = sI.getNumDivs(); j < e;
       ++j) {
    const DivisionConstraint<Int> &div = sI.getDivisions()[j];
    simplex.addInequality(div.getInequalityLowerBound().getCoeffs());
    simplex.addInequality(div.getInequalityUpperBound().getCoeffs());
  }

  unsigned offset = simplex.numConstraints();
  auto snapshot = simplex.getSnapshot();
  simplex.addBasicSet(sI);

  if (simplex.isEmpty()) {
    simplex.rollback(snapshot);
    subtractRecursively(b, simplex, s, i + 1, result);
    simplex.rollback(initialSnapshot);
    b = oldB;
    return;
  }

  simplex.detectRedundant();
  SmallVector<bool, 8> isMarkedRedundant;
  for (unsigned j = 0; j < 2 * sI.getNumEqualities() + sI.getNumInequalities();
       j++)
    isMarkedRedundant.push_back(simplex.isMarkedRedundant(offset + j));

  simplex.rollback(snapshot);
  // Recurse with the part b ^ ~ineq. Note that b is modified throughout
  // subtractRecursively. At the time this function is called, the current b is
  // actually equal to b ^ s_i1 ^ s_i2 ^ ... ^ s_ij, and ineq is the next
  // inequality, s_{i,j+1}. This function recurses into the next level i + 1
  // with the part b ^ s_i1 ^ s_i2 ^ ... ^ s_ij ^ ~s_{i,j+1}.
  auto recurseWithInequality = [&, i](ArrayRef<Int> ineq) {
    size_t snapshot = simplex.getSnapshot();
    b.addInequality(ineq);
    simplex.addInequality(ineq);
    subtractRecursively(b, simplex, s, i + 1, result);
    b.removeInequality(b.getNumInequalities() - 1);
    simplex.rollback(snapshot);
  };

  // For each inequality ineq, we first recurse with the part where ineq
  // is not satisfied, and then add the ineq to b and simplex because
  // ineq must be satisfied by all later parts.
  auto processInequality = [&](ArrayRef<Int> ineq) {
    recurseWithInequality(getComplementIneq(ineq));
    b.addInequality(ineq);
    simplex.addInequality(ineq);
  };

  // processInequality appends some additional constraints to b. We want to
  // rollback b to its initial state before returning, which we will do by
  // removing all constraints beyond the original number of inequalities
  // and equalities, so we store these counts first.
  unsigned originalNumIneqs = b.getNumInequalities();
  unsigned originalNumEqs = b.getNumEqualities();

  for (unsigned j = 0, e = sI.getNumInequalities(); j < e; j++) {
    if (isMarkedRedundant[j])
      continue;
    processInequality(sI.getInequality(j).getCoeffs());
  }

  offset = sI.getNumInequalities();
  for (unsigned j = 0, e = sI.getNumEqualities(); j < e; ++j) {
    const ArrayRef<Int> &coeffs = sI.getEquality(j).getCoeffs();
    // Same as the above loop for inequalities, done once each for the positive
    // and negative inequalities that make up this equality.
    if (!isMarkedRedundant[offset + 2 * j])
      processInequality(coeffs);
    if (!isMarkedRedundant[offset + 2 * j + 1])
      processInequality(getNegatedCoeffs(coeffs));
  }

  // Rollback b and simplex to their initial states.
  for (unsigned i = b.getNumInequalities(); i > originalNumIneqs; --i)
    b.removeInequality(i - 1);

  for (unsigned i = b.getNumEqualities(); i > originalNumEqs; --i)
    b.removeEquality(i - 1);

  simplex.rollback(initialSnapshot);
  b = oldB;
}

template <typename Int>
PresburgerSet<Int> /* why does the formatter want to merge these words?? */
PresburgerSet<Int>::eliminateExistentials(const PresburgerBasicSet<Int> &bs) {
  ParamLexSimplex<Int> paramLexSimplex(bs.getNumTotalDims(),
                                  bs.getNumParams() + bs.getNumDims());
  for (const auto &div : bs.getDivisions()) {
    // The division variables must be in the same order they are stored in the
    // basic set.
    paramLexSimplex.addInequality(div.getInequalityLowerBound().getCoeffs());
    paramLexSimplex.addInequality(div.getInequalityUpperBound().getCoeffs());
  }
  for (const auto &ineq : bs.getInequalities()) {
    paramLexSimplex.addInequality(ineq.getCoeffs());
  }
  for (const auto &eq : bs.getEqualities()) {
    paramLexSimplex.addEquality(eq.getCoeffs());
  }

  PresburgerSet<Int> result(bs.getNumDims(), bs.getNumParams());
  for (auto &b : paramLexSimplex.findParamLexmin().domain) {
    b.nParam = bs.nParam;
    b.nDim = bs.nDim;
    result.addBasicSet(std::move(b));
  }

  // Coalesce with preserve so as to not add additional existentials
  return coalesce(result, true);
}

template <typename Int>
PresburgerSet<Int> PresburgerSet<Int>::eliminateExistentials(const PresburgerSet<Int> &set) {
  PresburgerSet<Int> unquantifiedSet(set.getNumDims(), set.getNumSyms());
  for (const auto &bs : set.getBasicSets()) {
    if (bs.getNumExists() == 0) {
      unquantifiedSet.addBasicSet(bs);
    } else {
      unquantifiedSet.unionSet(PresburgerSet<Int>::eliminateExistentials(bs));
    }
  }
  return unquantifiedSet;
}

template <typename Int>
PresburgerSet<Int> PresburgerSet<Int>::eliminateExistentials(PresburgerSet<Int> &&set) {
  PresburgerSet<Int> unquantifiedSet(set.getNumDims(), set.getNumSyms());
  bool clean = true;
  for (auto &bs : set.basicSets) {
    if (bs.getNumExists() != 0) {
      clean = false;
      break;
    }
  }
  if (clean)
    return std::move(set);

  for (auto &bs : set.basicSets) {
    if (bs.getNumExists() == 0) {
      unquantifiedSet.addBasicSet(std::move(bs));
    } else {
      unquantifiedSet.unionSet(PresburgerSet<Int>::eliminateExistentials(bs));
    }
  }
  return unquantifiedSet;
}

// Returns the set difference c - set.
template <typename Int>
PresburgerSet<Int> PresburgerSet<Int>::subtract(PresburgerBasicSet<Int> cs,
                                      const PresburgerSet<Int> &set) {
  assertDimensionsCompatible(cs, set);

  if (set.isUniverse())
    return PresburgerSet<Int>::makeEmptySet(set.getNumDims(), set.getNumSyms());
  if (set.isMarkedEmpty())
    return PresburgerSet<Int>(cs);

  Simplex<Int> simplex(cs);
  PresburgerSet<Int> result(set.getNumDims(), set.getNumSyms());
  subtractRecursively(cs, simplex, eliminateExistentials(set), 0, result);
  return result;
}

template <typename Int>
PresburgerSet<Int> PresburgerSet<Int>::complement(const PresburgerSet<Int> &set) {
  auto res = subtract(PresburgerBasicSet<Int>(set.getNumDims(), set.getNumSyms(), 0),
                  set);
  return res;
}

// Subtracts the set S from the current set.
//
// We compute (U_i T_i) - (U_i S_i) as U_i (T_i - U_i S_i).
template <typename Int>
void PresburgerSet<Int>::subtract(const PresburgerSet<Int> &set) {
  assertDimensionsCompatible(set, *this);

  if (markedEmpty)
    return;
  if (set.isMarkedEmpty())
    return;
  if (set.isUniverse()) {
    markedEmpty = true;
    return;
  }
  if (isUniverse()) {
    *this = PresburgerSet<Int>::complement(set);
    return;
  }

  PresburgerSet<Int> result = PresburgerSet<Int>::makeEmptySet(nDim, nSym);
  for (const PresburgerBasicSet<Int> &c : basicSets)
    result.unionSet(subtract(c, set));
  *this = result;
}

template <typename Int>
bool PresburgerSet<Int>::equal(const PresburgerSet<Int> &s, const PresburgerSet<Int> &t) {
  // TODO we cannot assert here, as equal is used by other functionality that
  // otherwise breaks here
  // assert(s.getNumSyms() + t.getNumSyms() == 0 &&
  //       "operations on sets with symbols are not yet supported");

  assertDimensionsCompatible(s, t);
  PresburgerSet<Int> sCopy = s, tCopy = t;
  sCopy.subtract(std::move(t));
  if (!sCopy.isIntegerEmpty())
    return false;
  tCopy.subtract(std::move(s));
  if (!tCopy.isIntegerEmpty())
    return false;
  return true;
}

template <typename Int>
Optional<SmallVector<Int, 8>> PresburgerSet<Int>::findIntegerSample() {
  if (maybeSample)
    return maybeSample;
  if (markedEmpty)
    return {};
  if (isUniverse())
    return SmallVector<Int, 8>(nDim, 0);

  for (PresburgerBasicSet<Int> &cs : basicSets) {
    if (auto opt = cs.findIntegerSample()) {
      maybeSample = SmallVector<Int, 8>();

      for (Int v : opt.getValue())
        maybeSample->push_back(v);

      return maybeSample;
    }
  }
  return {};
}

template <typename Int>
bool PresburgerSet<Int>::isIntegerEmpty() const {
  if (markedEmpty)
    return true;
  for (const PresburgerBasicSet<Int> &bs : basicSets) {
    if (!bs.isIntegerEmpty())
      return false;
  }
  return true;
}

template <typename Int>
llvm::Optional<SmallVector<Int, 8>>
PresburgerSet<Int>::maybeGetCachedSample() const {
  if (isUniverse())
    return SmallVector<Int, 8>(nDim, 0);
  return maybeSample;
}

// TODO refactor and rewrite after discussion with the others
template <typename Int>
void PresburgerSet<Int>::printISL(raw_ostream &os) const {
  printPresburgerSetISL(os, *this);
}

template <typename Int>
void PresburgerSet<Int>::dumpISL() const {
  printISL(llvm::errs());
  llvm::errs() << '\n';
}

template <typename Int>
void PresburgerSet<Int>::print(raw_ostream &os) const {
  printPresburgerSet(os, *this);
}

template <typename Int>
void PresburgerSet<Int>::dump() const {
  print(llvm::errs());
  llvm::errs() << '\n';
}

template <typename Int>
void PresburgerSet<Int>::dumpCoeffs() const {
  llvm::errs() << "nBasicSets = " << basicSets.size() << '\n';
  for (auto &basicSet : basicSets) {
    basicSet.dumpCoeffs();
    llvm::errs() << "\n\n";
  }
}

template <typename Int>
llvm::hash_code PresburgerSet<Int>::hash_value() const {
  // TODO how should we hash PresburgerBasicSet<Int> without having access to
  // private vars?
  return llvm::hash_combine(nDim, nSym);
}

template <typename Int>
void PresburgerSet<Int>::simplify(bool aggressive) {
  for (auto &set : basicSets) {
    set.simplify(aggressive);
  } 
}

#endif // MLIR_ANALYSIS_PRESBURGER_SET_IMPL_H
