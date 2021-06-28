#ifndef MLIR_ANALYSIS_PRESBURGER_SET_H
#define MLIR_ANALYSIS_PRESBURGER_SET_H

#include "mlir/Analysis/Presburger/PresburgerBasicSet.h"
#include "mlir/Analysis/Presburger/Coalesce.h"
#include <variant>

namespace mlir {
namespace analysis {
namespace presburger {

template <typename Int>
class PresburgerSet {
public:
  PresburgerSet(unsigned nDim = 0, unsigned nSym = 0, bool markedEmpty = false)
      : nDim(nDim), nSym(nSym), markedEmpty(markedEmpty) {}
  PresburgerSet(PresburgerBasicSet<Int> cs);
  template <typename OInt>
  PresburgerSet(const PresburgerSet<OInt> &oSet);

  unsigned getNumBasicSets() const;
  unsigned getNumDims() const;
  unsigned getNumSyms() const;
  static PresburgerSet eliminateExistentials(const PresburgerBasicSet<Int> &bs);
  static PresburgerSet eliminateExistentials(const PresburgerSet &set);
  static PresburgerSet eliminateExistentials(PresburgerSet &&set);
  const SmallVector<PresburgerBasicSet<Int>, 4> &getBasicSets() const;
  void addBasicSet(const PresburgerBasicSet<Int> &cs);
  void addBasicSet(PresburgerBasicSet<Int> &&cs);
  void unionSet(const PresburgerSet &set);
  void unionSet(PresburgerSet &&set);
  void intersectSet(const PresburgerSet &set);
  void reserveBasicSets(unsigned count);
  static bool equal(const PresburgerSet &s, const PresburgerSet &t);
  void print(raw_ostream &os) const;
  void dump() const;
  void dumpCoeffs() const;
  void printISL(raw_ostream &os) const;
  void dumpISL() const;
  void printVariableList(raw_ostream &os) const;
  void printConstraints(raw_ostream &os) const;
  llvm::hash_code hash_value() const;
  bool isMarkedEmpty() const;
  bool isUniverse() const;

  static PresburgerSet makeEmptySet(unsigned nDim, unsigned nSym);
  static PresburgerSet complement(const PresburgerSet &set);
  void subtract(const PresburgerSet &set);
  static PresburgerSet subtract(PresburgerBasicSet<Int> c, const PresburgerSet &set);

  llvm::Optional<SmallVector<Int, 8>> findIntegerSample();
  bool isIntegerEmpty() const;
  // bool containsPoint(const std::vector<INT> &values) const;
  llvm::Optional<SmallVector<Int, 8>> maybeGetCachedSample() const;

  /// Simplfies each DivisionConstraint in each PresburgerBasicSet
  void simplify(bool aggresive = true);

  template <typename OInt>
  friend class PresburgerSet;

private:
  unsigned nDim;
  unsigned nSym;
  SmallVector<PresburgerBasicSet<Int>, 4> basicSets;
  // This is NOT just cached information about the constraints in basicSets.
  // If this is set to true, then the set is empty, irrespective of the state
  // of basicSets.
  bool markedEmpty;
  Optional<SmallVector<Int, 8>> maybeSample;
  void printBasicSet(raw_ostream &os, PresburgerBasicSet<Int> cs) const;
  void printVar(raw_ostream &os, Int var, unsigned i,
                unsigned &countNonZero) const;
};

using DialectSet = PresburgerSet<DefaultInt>;
} // namespace presburger
} // namespace analysis
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_SET_H
