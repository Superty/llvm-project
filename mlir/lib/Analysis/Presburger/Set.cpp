#include "mlir/Analysis/Presburger/Set.h"
#include "mlir/Analysis/Presburger/Simplex.h"

// TODO should we change this to a storage type?
using namespace mlir;

PresburgerSet::PresburgerSet(FlatAffineConstraints cs)
    : nDim(cs.getNumDimIds()), nSym(cs.getNumSymbolIds()), markedEmpty(false) {
  addFlatAffineConstraints(cs);
}

unsigned PresburgerSet::getNumBasicSets() const {
  return flatAffineConstraints.size();
}

unsigned PresburgerSet::getNumDims() const { return nDim; }

unsigned PresburgerSet::getNumSyms() const { return nSym; }

bool PresburgerSet::isMarkedEmpty() const { return markedEmpty; }

bool PresburgerSet::isUniverse() const {
  return flatAffineConstraints.size() == 0 && !markedEmpty;
}

const SmallVector<FlatAffineConstraints, 4> &
PresburgerSet::getFlatAffineConstraints() const {
  return flatAffineConstraints;
}

// This is only used to check assertions
static void assertDimensionsCompatible(FlatAffineConstraints cs,
                                       PresburgerSet set) {
  assert(cs.getNumDimIds() == set.getNumDims() &&
         cs.getNumSymbolIds() == set.getNumSyms() &&
         "Dimensionalities of FlatAffineConstraints and PresburgerSet do not "
         "match");
}

static void assertDimensionsCompatible(PresburgerSet set1, PresburgerSet set2) {
  assert(set1.getNumDims() == set2.getNumDims() &&
         set1.getNumSyms() == set2.getNumSyms() &&
         "Dimensionalities of PresburgerSets do not match");
}

void PresburgerSet::addFlatAffineConstraints(FlatAffineConstraints cs) {
  assertDimensionsCompatible(cs, *this);

  markedEmpty = false;
  flatAffineConstraints.push_back(cs);
}

void PresburgerSet::unionSet(const PresburgerSet &set) {
  assertDimensionsCompatible(set, *this);

  for (const FlatAffineConstraints &cs : set.flatAffineConstraints)
    addFlatAffineConstraints(std::move(cs));
}

// Compute the intersection of the two sets.
//
// We directly compute (S_1 or S_2 ...) and (T_1 or T_2 ...)
// as (S_1 and T_1) or (S_1 and T_2) or ...
void PresburgerSet::intersectSet(const PresburgerSet &set) {
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

  PresburgerSet result(nDim, nSym, true);
  for (const FlatAffineConstraints &cs1 : flatAffineConstraints) {
    for (const FlatAffineConstraints &cs2 : set.flatAffineConstraints) {
      FlatAffineConstraints intersection(cs1);
      intersection.append(cs2);
      if (!intersection.isEmpty())
        result.addFlatAffineConstraints(std::move(intersection));
    }
  }
  *this = std::move(result);
}

PresburgerSet PresburgerSet::makeEmptySet(unsigned nDim, unsigned nSym) {
  PresburgerSet result(nDim, nSym, true);
  return result;
}

SmallVector<int64_t, 8> inequalityFromEquality(const ArrayRef<int64_t> &eq,
                                               bool negated, bool strict) {
  SmallVector<int64_t, 8> coeffs;
  for (auto coeff : eq)
    coeffs.emplace_back(negated ? -coeff : coeff);

  // The constant is at the end
  if (strict)
    --coeffs.back();

  return coeffs;
}

SmallVector<int64_t, 8> complementIneq(const ArrayRef<int64_t> &ineq) {
  return inequalityFromEquality(ineq, true, true);
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
void subtractRecursively(FlatAffineConstraints &b, Simplex &simplex,
                         const PresburgerSet &s, unsigned i,
                         PresburgerSet &result) {
  if (i == s.getNumBasicSets()) {
    // FlatAffineConstraints BCopy = B;
    // BCopy.simplify();
    result.addFlatAffineConstraints(b);
    return;
  }
  const FlatAffineConstraints &sI = s.getFlatAffineConstraints()[i];
  auto initialSnap = simplex.getSnapshot();
  unsigned offset = simplex.numConstraints();
  simplex.addFlatAffineConstraints(sI);

  if (simplex.isEmpty()) {
    simplex.rollback(initialSnap);
    subtractRecursively(b, simplex, s, i + 1, result);
    return;
  }

  SmallVector<bool, 8> isMarkedRedundant;
  for (unsigned j = 0; j < 2 * sI.getNumEqualities() + sI.getNumInequalities();
       j++)
    isMarkedRedundant.push_back(simplex.isMarkedRedundant(offset + j));

  simplex.rollback(initialSnap);
  // TODO benchmark does it make a lot of difference if we always_inline this?
  auto addInequality = [&](const ArrayRef<int64_t> &ineq) {
    b.addInequality(ineq);
    simplex.addInequality(ineq);
  };

  auto recurseWithInequality = [&, i](const ArrayRef<int64_t> &ineq) {
    size_t snap = simplex.getSnapshot();

    addInequality(ineq);

    subtractRecursively(b, simplex, s, i + 1, result);

    b.removeInequality(b.getNumInequalities() - 1);
    simplex.rollback(snap);
  };

  unsigned originalNumIneqs = b.getNumInequalities();
  unsigned originalNumEqs = b.getNumEqualities();

  for (unsigned j = 0, e = sI.getNumEqualities(); j < e; ++j) {
    // The first inequality is positive and the second is negative, of which
    // we need the complements (strict negative and strict positive).
    const auto &eq = sI.getEquality(j);
    if (!isMarkedRedundant[2 * j]) {
      recurseWithInequality(inequalityFromEquality(eq, true, true));
      if (isMarkedRedundant[2 * j + 1]) {
        addInequality(inequalityFromEquality(eq, false, false));
        continue;
      }
    }
    if (!isMarkedRedundant[2 * j + 1]) {
      recurseWithInequality(inequalityFromEquality(eq, false, true));
      if (isMarkedRedundant[2 * j]) {
        addInequality(inequalityFromEquality(eq, true, false));
        continue;
      }
    }

    b.addEquality(eq);
    simplex.addEquality(eq);
  }

  offset = 2 * sI.getNumEqualities();
  for (unsigned j = 0; j < sI.getNumInequalities(); j++) {
    if (isMarkedRedundant[offset + j])
      continue;
    const auto &ineq = sI.getInequality(j);

    recurseWithInequality(complementIneq(ineq));
    addInequality(ineq);
  }

  for (unsigned i = b.getNumInequalities(); i > originalNumIneqs; --i)
    b.removeInequality(i - 1);

  for (unsigned i = b.getNumEqualities(); i > originalNumEqs; --i)
    b.removeEquality(i - 1);

  // TODO benchmark technically we can probably drop this as the caller will
  // rollback. See if it makes much of a difference. Only the last rollback
  // would be eliminated by this.
  simplex.rollback(initialSnap);
}

// Returns the set difference c - set.
PresburgerSet PresburgerSet::subtract(FlatAffineConstraints cs,
                                      const PresburgerSet &set) {
  assertDimensionsCompatible(cs, set);
  if (cs.isEmptyByGCDTest())
    return PresburgerSet::makeEmptySet(cs.getNumDimIds(), cs.getNumSymbolIds());

  if (set.isUniverse())
    return PresburgerSet::makeEmptySet(set.getNumDims(), set.getNumSyms());
  if (set.isMarkedEmpty())
    return PresburgerSet(cs);

  PresburgerSet result(cs.getNumDimIds(), cs.getNumSymbolIds());
  Simplex simplex(cs);
  subtractRecursively(cs, simplex, set, 0, result);
  return result;
}

PresburgerSet PresburgerSet::complement(const PresburgerSet &set) {
  return subtract(FlatAffineConstraints(set.getNumDims(), set.getNumSyms()),
                  set);
}

// Subtracts the set S from the current set.
//
// We compute (U_i T_i) - (U_i S_i) as U_i (T_i - U_i S_i).
void PresburgerSet::subtract(const PresburgerSet &set) {
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

  PresburgerSet result = PresburgerSet::makeEmptySet(nDim, nSym);
  for (const FlatAffineConstraints &c : flatAffineConstraints)
    result.unionSet(subtract(c, set));
  *this = result;
}

bool PresburgerSet::equal(const PresburgerSet &s, const PresburgerSet &t) {
  // TODO we cannot assert here, as equal is used by other functionality that
  // otherwise breaks here
  // assert(s.getNumSyms() + t.getNumSyms() == 0 &&
  //       "operations on sets with symbols are not yet supported");
  if (s.getNumSyms() + t.getNumSyms() != 0)
    return false;
  if (s.getNumDims() != t.getNumDims())
    return false;

  PresburgerSet sCopy = s, tCopy = t;
  sCopy.subtract(t);
  tCopy.subtract(s);
  return !sCopy.findIntegerSample() && !tCopy.findIntegerSample();
}

Optional<SmallVector<int64_t, 8>> PresburgerSet::findIntegerSample() {
  assert(nSym == 0 && "sampling on sets with symbols is not yet supported");
  if (maybeSample)
    return maybeSample;
  if (markedEmpty)
    return {};
  if (isUniverse())
    return SmallVector<int64_t, 8>(nDim, 0);

  for (FlatAffineConstraints &cs : flatAffineConstraints) {
    if (auto opt = cs.findIntegerSample()) {
      maybeSample = SmallVector<int64_t, 8>();

      for (int64_t v : opt.getValue())
        maybeSample->push_back(v);

      return maybeSample;
    }
  }
  return {};
}

llvm::Optional<SmallVector<int64_t, 8>>
PresburgerSet::maybeGetCachedSample() const {
  if (isUniverse())
    return SmallVector<int64_t, 8>(nDim, 0);
  return maybeSample;
}

// TODO refactor and rewrite after discussion with the others
void PresburgerSet::print(raw_ostream &os) const {
  printVariableList(os);
  os << " : ";
  printConstraints(os);
}

void PresburgerSet::printConstraints(raw_ostream &os) const {
  if (markedEmpty) {
    os << "(1 = 0)";
    return;
  }
  os << "(";
  bool fst = true;
  for (auto &c : flatAffineConstraints) {
    if (fst)
      fst = false;
    else
      os << " or ";
    printFlatAffineConstraints(os, c);
  }
  os << ")";
}

void PresburgerSet::dump() const { print(llvm::errs()); }

void PresburgerSet::printVar(raw_ostream &os, int64_t val, unsigned i,
                             unsigned &countNonZero) const {
  bool isConst = i >= getNumDims() + getNumSyms();
  if (val == 0) {
    return;
  } else if (val > 0) {
    if (countNonZero > 0) {
      os << " + ";
    }
    if (val > 1 || isConst)
      os << val;
  } else {
    if (countNonZero > 0) {
      os << " - ";
      if (val != -1 || isConst)
        os << -val;
    } else {
      if (val == -1 && !isConst)
        os << "-";
      else
        os << val;
    }
  }

  if (i < getNumDims()) {
    os << 'd' << i;
  } else if (i < getNumDims() + getNumSyms()) {
    os << 's' << (i - getNumDims());
  }
  countNonZero++;
}

void PresburgerSet::printFlatAffineConstraints(raw_ostream &os,
                                               FlatAffineConstraints cs) const {
  for (unsigned i = 0, e = cs.getNumEqualities(); i < e; ++i) {
    if (i != 0)
      os << " and ";
    unsigned countNonZero = 0;
    for (unsigned j = 0, f = cs.getNumCols(); j < f; ++j) {
      printVar(os, cs.atEq(i, j), j, countNonZero);
    }
    os << " = 0";
  }
  if (cs.getNumEqualities() > 0 && cs.getNumInequalities() > 0)
    os << " and ";
  for (unsigned i = 0, e = cs.getNumInequalities(); i < e; ++i) {
    if (i != 0)
      os << " and ";
    unsigned countNonZero = 0;
    for (unsigned j = 0, f = cs.getNumCols(); j < f; ++j) {
      printVar(os, cs.atIneq(i, j), j, countNonZero);
    }
    os << " >= 0";
  }
}

void PresburgerSet::printVariableList(raw_ostream &os) const {
  os << "(";
  for (unsigned i = 0; i < getNumDims(); i++)
    os << (i != 0 ? ", " : "") << 'd' << i;
  os << ")";

  if (getNumDims() > 0) {
    os << "[";
    for (unsigned i = 0; i < getNumSyms(); i++)
      os << (i != 0 ? ", " : "") << 's' << i;
    os << "]";
  }
}

llvm::hash_code PresburgerSet::hash_value() const {
  // TODO how should we hash FlatAffineConstraints without having access to
  // private vars?
  return llvm::hash_combine(nDim, nSym);
}
