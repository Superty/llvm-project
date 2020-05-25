#include "mlir/Dialect/Presburger/Set.h"

// TODO should be change this to a storage type?
using namespace mlir;
using namespace mlir::presburger;

unsigned PresburgerSet::getNumBasicSets() const {
  return flatAffineConstraints.size();
}

unsigned PresburgerSet::getNumDims() const { return nDim; }

unsigned PresburgerSet::getNumSyms() const { return nSym; }

bool PresburgerSet::isMarkedEmpty() const { return markedEmpty; }

bool PresburgerSet::isUniverse() const {
  return flatAffineConstraints.size() == 0;
}

const SmallVector<FlatAffineConstraints, 4> &
PresburgerSet::getFlatAffineConstrains() const {
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

bool PresburgerSet::equal(const PresburgerSet &s, const PresburgerSet &t) {
  // TODO implemenmt this
  return false;
}

// TODO refactor and rewrite after discussion with the others
void PresburgerSet::print(raw_ostream &os) const {
  printVariableList(os);
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
