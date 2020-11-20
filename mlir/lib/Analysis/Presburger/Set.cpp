#include "mlir/Analysis/Presburger/Set.h"
#include "mlir/Analysis/Presburger/Constraint.h"
#include "mlir/Analysis/Presburger/Printer.h"
#include "mlir/Analysis/Presburger/ISLPrinter.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Analysis/Presburger/ParamLexSimplex.h"
#include "llvm/ADT/SmallVector.h"

// TODO should we change this to a storage type?
using namespace mlir;
using namespace analysis::presburger;

PresburgerSet::PresburgerSet(PresburgerBasicSet cs)
    : nDim(cs.getNumDims()), nSym(cs.getNumParams()), markedEmpty(false) {
  addBasicSet(cs);
}

unsigned PresburgerSet::getNumBasicSets() const {
  return basicSets.size();
}

unsigned PresburgerSet::getNumDims() const { return nDim; }

unsigned PresburgerSet::getNumSyms() const { return nSym; }

bool PresburgerSet::isMarkedEmpty() const { return markedEmpty || basicSets.empty(); }

bool PresburgerSet::isUniverse() const {
  if (markedEmpty || basicSets.empty())
    return false;
  for (const PresburgerBasicSet &bs : basicSets) {
    if (bs.getNumInequalities() == 0 && bs.getNumEqualities() == 0)
      return true;
  }
  return false;
}

const SmallVector<PresburgerBasicSet, 4> &
PresburgerSet::getBasicSets() const {
  return basicSets;
}

// This is only used to check assertions
static void assertDimensionsCompatible(PresburgerBasicSet cs,
                                       PresburgerSet set) {
  assert(cs.getNumDims() == set.getNumDims() &&
         cs.getNumParams() == set.getNumSyms() &&
         "Dimensionalities of PresburgerBasicSet and PresburgerSet do not "
         "match");
}

static void assertDimensionsCompatible(PresburgerSet set1, PresburgerSet set2) {
  assert(set1.getNumDims() == set2.getNumDims() &&
         set1.getNumSyms() == set2.getNumSyms() &&
         "Dimensionalities of PresburgerSets do not match");
}

void PresburgerSet::addBasicSet(PresburgerBasicSet cs) {
  assertDimensionsCompatible(cs, *this);
  markedEmpty = false;
  basicSets.push_back(cs);
}

void PresburgerSet::unionSet(PresburgerSet set) {
  alignParams(*this, set);
  assertDimensionsCompatible(set, *this);
  for (PresburgerBasicSet &cs : set.basicSets)
    addBasicSet(std::move(cs));
}

// Compute the intersection of the two sets.
//
// We directly compute (S_1 or S_2 ...) and (T_1 or T_2 ...)
// as (S_1 and T_1) or (S_1 and T_2) or ...
void PresburgerSet::intersectSet(PresburgerSet set) {
  alignParams(*this, set);
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

  PresburgerSet result(nDim, nSym, true, set.getParamNames());
  for (const PresburgerBasicSet &cs1 : basicSets) {
    for (const PresburgerBasicSet &cs2 : set.basicSets) {
      PresburgerBasicSet intersection(cs1);
      intersection.intersect(cs2);
      result.addBasicSet(std::move(intersection));
    }
  }
  *this = std::move(result);
}

PresburgerSet PresburgerSet::makeEmptySet(unsigned nDim, unsigned nSym, ArrayRef<std::string> oParamNames) {
  PresburgerSet result(nDim, nSym, true, oParamNames);
  return result;
}

/// Return `coeffs` with all the elements negated.
static SmallVector<int64_t, 8> getNegatedCoeffs(ArrayRef<int64_t> coeffs) {
  SmallVector<int64_t, 8> negatedCoeffs;
  negatedCoeffs.reserve(coeffs.size());
  for (int64_t coeff : coeffs)
    negatedCoeffs.emplace_back(-coeff);
  return negatedCoeffs;
}

/// Return the complement of the given inequality.
///
/// The complement of a_1 x_1 + ... + a_n x_ + c >= 0 is
/// a_1 x_1 + ... + a_n x_ + c < 0, i.e., -a_1 x_1 - ... - a_n x_ - c - 1 >= 0.
static SmallVector<int64_t, 8> getComplementIneq(ArrayRef<int64_t> ineq) {
  SmallVector<int64_t, 8> coeffs;
  coeffs.reserve(ineq.size());
  for (int64_t coeff : ineq)
    coeffs.emplace_back(-coeff);
  --coeffs.back();
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
void subtractRecursively(PresburgerBasicSet &b, Simplex &simplex,
                         const PresburgerSet &s, unsigned i,
                         PresburgerSet &result) {
  if (i == s.getNumBasicSets()) {
    // PresburgerBasicSet BCopy = B;
    // BCopy.simplify();
    result.addBasicSet(b);
    return;
  }
  PresburgerBasicSet oldB = b;
  PresburgerBasicSet sI = s.getBasicSets()[i];

  auto initialSnapshot = simplex.getSnapshot();
  unsigned numSIDivs = sI.getNumDivs();
  PresburgerBasicSet::toCommonSpace(b, sI);
  for (unsigned j = 0; j < numSIDivs; ++j)
    simplex.addVariable();

  for (unsigned j = sI.getNumDivs() - numSIDivs, e = sI.getNumDivs(); j < e; ++j) {
    const DivisionConstraint &div = sI.getDivisions()[j];
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
  auto recurseWithInequality = [&, i](ArrayRef<int64_t> ineq) {
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
  auto processInequality = [&](ArrayRef<int64_t> ineq) {
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
    const ArrayRef<int64_t> &coeffs = sI.getEquality(j).getCoeffs();
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

PresburgerSet PresburgerSet::eliminateExistentials(const PresburgerBasicSet &bs) {
  ParamLexSimplex paramLexSimplex(bs.getNumTotalDims(), bs.getNumParams() + bs.getNumDims());
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

  PresburgerSet result(bs.getNumDims(), bs.getNumParams());
  for (auto &b : paramLexSimplex.findParamLexmin().domain) {
    b.nParam = bs.nParam;
    b.nDim = bs.nDim; 
    result.addBasicSet(b);
  }
  return result;
}

void PresburgerSet::insertParametricDimensions(unsigned pos, unsigned count) {
  for (auto &bs : basicSets)
    bs.insertParametricDimensions(pos, count);
  nSym += count;
}

void PresburgerSet::swapDimensions(unsigned i, unsigned j) {
  for (auto &bs : basicSets)
    bs.swapDimensions(i, j);
}

void PresburgerSet::alignParams(PresburgerSet &s, PresburgerSet &t) {
  auto &sl = s.paramNames;
  auto &tl = t.paramNames;
  if (sl == tl)
    return;
  assert(sl.size() == s.nSym && tl.size() == t.nSym);

  unsigned nCommon = 0;
  for (unsigned i = 0; i < sl.size(); ++i) {
    for (unsigned j = 0; j < tl.size(); ++j) {
      if (sl[i] == tl[j]) {
        s.swapDimensions(s.nDim + nCommon, s.nDim + i);
        t.swapDimensions(t.nDim + nCommon, t.nDim + j);
        std::swap(sl[nCommon], sl[i]);
        std::swap(tl[nCommon], tl[j]);
        nCommon++;
        break;
      }
    }
  }

  sl.insert(sl.end(),             tl.begin() + nCommon, tl.begin() + t.nSym);
  tl.insert(tl.begin() + nCommon, sl.begin() + nCommon, sl.begin() + s.nSym);
  unsigned total = sl.size();
  s.insertParametricDimensions(s.nSym,  total - s.nSym);
  t.insertParametricDimensions(nCommon, total - t.nSym);
}

void PresburgerSet::dumpParamNames() const {
  for (auto name : paramNames) {
    llvm::errs() << name << ", ";
  }
  llvm::errs() << '\n';
}

PresburgerSet PresburgerSet::eliminateExistentials(const PresburgerSet &set) {
  PresburgerSet unquantifiedSet(set.getNumDims(), set.getNumSyms(), set.getParamNames());
  for (const auto &bs : set.getBasicSets()) {
    if (bs.getNumExists() == 0) {
      unquantifiedSet.addBasicSet(bs);
    } else {
      auto res = PresburgerSet::eliminateExistentials(bs);
      res.paramNames = set.paramNames;
      unquantifiedSet.unionSet(std::move(res));
    }
  }
  return unquantifiedSet;
}

// Returns the set difference c - set.
PresburgerSet PresburgerSet::subtract(PresburgerBasicSet cs,
                                      const PresburgerSet &set) {
  assertDimensionsCompatible(cs, set);

  if (set.isUniverse())
    return PresburgerSet::makeEmptySet(set.getNumDims(), set.getNumSyms(), set.getParamNames());
  if (set.isMarkedEmpty())
    return PresburgerSet(cs);

  Simplex simplex(cs);
  PresburgerSet result(set.getNumDims(), set.getNumSyms(), set.getParamNames());
  subtractRecursively(cs, simplex, eliminateExistentials(set), 0, result);
  return result;
}

PresburgerSet PresburgerSet::complement(const PresburgerSet &set) {
  return subtract(PresburgerBasicSet(set.getNumDims(), set.getNumSyms(), 0),
                  set);
}

// Subtracts the set S from the current set.
//
// We compute (U_i T_i) - (U_i S_i) as U_i (T_i - U_i S_i).
void PresburgerSet::subtract(PresburgerSet set) {
  alignParams(*this, set);
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
    *this = PresburgerSet::complement(set);
    return;
  }

  PresburgerSet result = PresburgerSet::makeEmptySet(nDim, nSym, set.getParamNames());
  for (PresburgerBasicSet &c : basicSets)
    result.unionSet(subtract(std::move(c), set));
  *this = std::move(result);
}

bool PresburgerSet::equal(PresburgerSet s, PresburgerSet t) {
  // TODO we cannot assert here, as equal is used by other functionality that
  // otherwise breaks here
  // assert(s.getNumSyms() + t.getNumSyms() == 0 &&
  //       "operations on sets with symbols are not yet supported");

  alignParams(s, t);
  assertDimensionsCompatible(s, t);

  PresburgerSet sCopy = s, tCopy = t;
  sCopy.subtract(std::move(t));
  if (!sCopy.isIntegerEmpty())
    return false;
  tCopy.subtract(std::move(s));
  if (!tCopy.isIntegerEmpty())
    return false;
  return true;
}

Optional<SmallVector<int64_t, 8>> PresburgerSet::findIntegerSample() {
  if (maybeSample)
    return maybeSample;
  if (markedEmpty)
    return {};
  if (isUniverse())
    return SmallVector<int64_t, 8>(nDim, 0);

  for (PresburgerBasicSet &cs : basicSets) {
    if (auto opt = cs.findIntegerSample()) {
      maybeSample = SmallVector<int64_t, 8>();

      for (int64_t v : opt.getValue())
        maybeSample->push_back(v);

      return maybeSample;
    }
  }
  return {};
}

bool PresburgerSet::isIntegerEmpty() {
  if (markedEmpty)
    return true;
  for (PresburgerBasicSet &bs : basicSets) {
    if (!bs.isIntegerEmpty())
      return false;
  }
  return true;
}

llvm::Optional<SmallVector<int64_t, 8>>
PresburgerSet::maybeGetCachedSample() const {
  if (isUniverse())
    return SmallVector<int64_t, 8>(nDim, 0);
  return maybeSample;
}

// TODO refactor and rewrite after discussion with the others
void PresburgerSet::printISL(raw_ostream &os) const {
  printPresburgerSetISL(os, *this);
}

void PresburgerSet::dumpISL() const { printISL(llvm::errs()); llvm::errs() << '\n'; }

void PresburgerSet::print(raw_ostream &os) const {
  printPresburgerSet(os, *this);
}

void PresburgerSet::dump() const { print(llvm::errs()); llvm::errs() << '\n'; }

void PresburgerSet::dumpCoeffs() const {
  llvm::errs() << "nBasicSets = " << basicSets.size() << '\n';
  for (auto &basicSet : basicSets) {
    basicSet.dumpCoeffs();
    llvm::errs() << "\n\n";
  }
}

llvm::hash_code PresburgerSet::hash_value() const {
  // TODO how should we hash PresburgerBasicSet without having access to
  // private vars?
  return llvm::hash_combine(nDim, nSym);
}