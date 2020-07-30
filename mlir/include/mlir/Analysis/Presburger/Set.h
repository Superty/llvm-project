#ifndef MLIR_ANALYSIS_PRESBURGER_SET_H
#define MLIR_ANALYSIS_PRESBURGER_SET_H

#include "mlir/Analysis/AffineStructures.h"

namespace mlir {
namespace analysis {
namespace presburger {

class PresburgerSet {
public:
  PresburgerSet(unsigned nDim = 0, unsigned nSym = 0, bool markedEmpty = false)
      : nDim(nDim), nSym(nSym), markedEmpty(markedEmpty) {}
  PresburgerSet(FlatAffineConstraints cs);

  unsigned getNumBasicSets() const;
  unsigned getNumDims() const;
  unsigned getNumSyms() const;
  const SmallVector<FlatAffineConstraints, 4> &getFlatAffineConstraints() const;
  void addFlatAffineConstraints(FlatAffineConstraints cs);
  void unionSet(const PresburgerSet &set);
  void intersectSet(const PresburgerSet &set);
  static bool equal(const PresburgerSet &s, const PresburgerSet &t);
  void print(raw_ostream &os) const;
  void printVariableList(raw_ostream &os) const;
  void printConstraints(raw_ostream &os) const;
  void dump() const;
  llvm::hash_code hash_value() const;
  bool isMarkedEmpty() const;
  bool isUniverse() const;

  static PresburgerSet makeEmptySet(unsigned nDim, unsigned nSym);
  static PresburgerSet complement(const PresburgerSet &set);
  void subtract(const PresburgerSet &set);
  static PresburgerSet subtract(FlatAffineConstraints c,
                                const PresburgerSet &set);

  llvm::Optional<SmallVector<int64_t, 8>> findIntegerSample();
  // bool containsPoint(const std::vector<INT> &values) const;
  llvm::Optional<SmallVector<int64_t, 8>> maybeGetCachedSample() const;

private:
  unsigned nDim;
  unsigned nSym;
  SmallVector<FlatAffineConstraints, 4> flatAffineConstraints;
  // This is NOT just cached information about the constraints in basicSets.
  // If this is set to true, then the set is empty, irrespective of the state
  // of basicSets.
  bool markedEmpty;
  Optional<SmallVector<int64_t, 8>> maybeSample;
};

} // namespace presburger
} // namespace analysis
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_SET_H
