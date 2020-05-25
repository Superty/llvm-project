#ifndef PRESBURGER_SET_H
#define PRESBURGER_SET_H

#include "mlir/Analysis/AffineStructures.h"

namespace mlir {
namespace presburger {

class PresburgerSet {
public:
  PresburgerSet(unsigned nDim = 0, unsigned nSym = 0, bool markedEmpty = false)
      : nDim(nDim), nSym(nSym), markedEmpty(markedEmpty) {}

  unsigned getNumBasicSets() const;
  unsigned getNumDims() const;
  unsigned getNumSyms() const;
  const SmallVector<FlatAffineConstraints, 4> &getFlatAffineConstrains() const;
  void addFlatAffineConstraints(FlatAffineConstraints cs);
  void unionSet(const PresburgerSet &set);
  void intersectSet(const PresburgerSet &set);
  static bool equal(const PresburgerSet &s, const PresburgerSet &t);
  void print(raw_ostream &os) const;
  llvm::hash_code hash_value() const;
  bool isMarkedEmpty() const;
  bool isUniverse() const;

  /*std::optional<std::vector<INT>> findSample();
  static Set complement(const Set &set);
  static Set makeEmptySet(unsigned nDim);
  /// Checks if the two sets are equal. The provided sets must have the same
  /// dimensionality.
  static Set subtract(BasicSet B, const Set &S);
  void subtract(const Set &set);
  void dump() const;
  bool containsPoint(const std::vector<INT> &values) const;
  std::optional<std::vector<INT>> maybeGetCachedSample() const;
*/
private:
  unsigned nDim;
  unsigned nSym;
  SmallVector<FlatAffineConstraints, 4> flatAffineConstraints;
  // This is NOT just cached information about the constraints in basicSets.
  // If this is set to true, then the set is empty, irrespective of the state
  // of basicSets.
  bool markedEmpty;
  // std::optional<std::vector<INT>> maybeSample;
  void printFlatAffineConstraints(raw_ostream &os,
                                  FlatAffineConstraints cs) const;
  void printVariableList(raw_ostream &os) const;
  void printVar(raw_ostream &os, int64_t var, unsigned i,
                unsigned &countNonZero) const;
};

} // namespace presburger
} // namespace mlir

#endif // PRESBURGER_SET_H
