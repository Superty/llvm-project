#include "mlir/Dialect/Presburger/Set.h"
#include "mlir/Analysis/Presburger/Simplex.h"

// TODO should we change this to a storage type?
using namespace mlir;
using namespace mlir::presburger;

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

void PresburgerSet::addFlatAffineConstraints(FlatAffineConstraints cs) {
  assert(cs.getNumDimIds() == nDim && cs.getNumSymbolIds() == nSym &&
         "Cannot add FlatAffineConstraints having different dimensionality");

  if (cs.isEmptyByGCDTest())
    return;

  markedEmpty = false;
  flatAffineConstraints.push_back(cs);
}

void PresburgerSet::unionSet(const PresburgerSet &set) {
  assert(set.getNumDims() == nDim && set.getNumSyms() == nSym &&
         "Cannot union Presburger sets having different dimensionality");

  for (const FlatAffineConstraints &cs : set.flatAffineConstraints)
    addFlatAffineConstraints(std::move(cs));
}

// Compute the intersection of the two sets.
//
// We directly compute (S_1 or S_2 ...) and (T_1 or T_2 ...)
// as (S_1 and T_1) or (S_1 and T_2) or ...
void PresburgerSet::intersectSet(const PresburgerSet &set) {
  assert(set.getNumDims() == nDim && set.getNumSyms() == nSym &&
         "Cannot intersect Presburger sets having different dimensionality");

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
// TODO reimplement this heuristic:
// As a heuristic, we try adding all the constraints and check if simplex
// says that the intersection is empty. Also, in the process we find out that
// some constraints are redundant, which we then ignore.
void subtractRecursively(FlatAffineConstraints &B, Simplex &simplex,
                         const PresburgerSet &S, unsigned i,
                         PresburgerSet &result) {
  if (i == S.getNumBasicSets()) {
    FlatAffineConstraints BCopy = B;
    // BCopy.simplify();
    result.addFlatAffineConstraints(std::move(BCopy));
    return;
  }
  const FlatAffineConstraints &S_i = S.getFlatAffineConstraints()[i];
  auto initialSnap = simplex.getSnapshot();
  // unsigned offset = simplex.numberConstraints();
  // simplex.addBasicSetAsInequalities(set_i);
  simplex.addFlatAffineConstraints(S_i);

  if (simplex.isEmpty()) {
    simplex.rollback(initialSnap);
    subtractRecursively(B, simplex, S, i + 1, result);
    return;
  }

  /*std::vector<bool> isMarkedRedundant;
  for (size_t j = 0; j < 2 * S_i.getNumEqualities() + S_i.getNumInequalities();
       j++)
    isMarkedRedundant.push_back(simplex.isMarkedRedundant(offset + j));
  */
  simplex.rollback(initialSnap);
  // TODO benchmark does it make a lot of difference if we always_inline this?
  auto addInequalityFromEquality = [&](const ArrayRef<int64_t> &eq,
                                       bool negated, bool strict) {
    SmallVector<int64_t, 64> coeffs;
    for (auto coeff : eq)
      coeffs.emplace_back(negated ? -coeff : coeff);

    // The constant is at the end
    if (strict)
      --coeffs[eq.size() - 1];

    B.addInequality(coeffs);
    simplex.addInequality(coeffs);
  };
  auto recurseWithInequalityFromEquality = [&, i](const ArrayRef<int64_t> &eq,
                                                  bool negated, bool strict) {
    size_t snap = simplex.getSnapshot();
    addInequalityFromEquality(eq, negated, strict);

    subtractRecursively(B, simplex, S, i + 1, result);

    // TODO check if this removes the right inequality
    B.removeInequality(B.getNumInequalities() - 1);
    simplex.rollback(snap);
  };

  size_t addedIneqs = 0;

  for (size_t j = 0; j < S_i.getNumEqualities(); j++) {
    // The first inequality is positive and the second is negative, of which
    // we need the complements (strict negative and strict positive).
    // TODO reimplement the heuristics
    const auto &eq = S_i.getEquality(j);
    recurseWithInequalityFromEquality(eq, true, true);
    recurseWithInequalityFromEquality(eq, false, true);

    addInequalityFromEquality(eq, false, false);
    addInequalityFromEquality(eq, true, false);
    addedIneqs += 2;
  }

  // offset = 2 * S_i.getNumEqualities();
  for (size_t j = 0; j < S_i.getNumInequalities(); j++) {
    /*if (isMarkedRedundant[offset + j])
      continue;*/
    const auto &ineq = S_i.getInequality(j);

    size_t snap = simplex.getSnapshot();

    SmallVector<int64_t, 64> complement;
    for (auto coeff : ineq)
      complement.emplace_back(-coeff);

    // The constant is at the end
    --complement[ineq.size() - 1];

    B.addInequality(complement);
    simplex.addInequality(complement);
    subtractRecursively(B, simplex, S, i + 1, result);
    B.removeInequality(B.getNumInequalities() - 1);
    simplex.rollback(snap);

    B.addInequality(ineq);
    addedIneqs++;
    simplex.addInequality(ineq);
  }

  for (size_t i = 0; i < addedIneqs; i++)
    B.removeInequality(B.getNumInequalities() - 1);

  // TODO benchmark technically we can probably drop this as the caller will
  // rollback. See if it makes much of a difference. Only the last rollback
  // would be eliminated by this.
  simplex.rollback(initialSnap);
}

// Returns the set difference B - S.
PresburgerSet PresburgerSet::subtract(FlatAffineConstraints c,
                                      const PresburgerSet &set) {
  assert(set.getNumDims() == c.getNumDimIds() &&
         set.getNumSyms() == c.getNumSymbolIds() &&
         "Sets to be subtracted have different dimensionality");
  if (c.isEmptyByGCDTest())
    return PresburgerSet::makeEmptySet(c.getNumDimIds(), c.getNumSymbolIds());

  if (set.isUniverse())
    return PresburgerSet::makeEmptySet(set.getNumDims(), set.getNumSyms());
  if (set.isMarkedEmpty())
    return PresburgerSet(set.getNumDims(), set.getNumSyms());

  PresburgerSet result(c.getNumDimIds());
  Simplex simplex(c);
  subtractRecursively(c, simplex, set, 0, result);
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
  assert(set.getNumDims() == nDim && set.getNumSyms() == nSym &&
         "Sets to be subtracted have different dimensionality");
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
  // TODO implemenmt this
  return false;
}

// TODO refactor and rewrite after discussion with the others
void PresburgerSet::print(raw_ostream &os) const {
  printVariableList(os);
  if (markedEmpty) {
    // TODO dicuss what we want to print in the empty case
    os << " : (1 = 0)";
    return;
  }
  os << " : (";
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
