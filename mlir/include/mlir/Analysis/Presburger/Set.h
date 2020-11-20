#ifndef MLIR_ANALYSIS_PRESBURGER_SET_H
#define MLIR_ANALYSIS_PRESBURGER_SET_H

#include "mlir/Analysis/Presburger/PresburgerBasicSet.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace analysis {
namespace presburger {

class PresburgerSet {
public:
  PresburgerSet(unsigned nDim = 0, unsigned nSym = 0, bool markedEmpty = false, ArrayRef<std::string> oParamNames = {})
      : nDim(nDim), nSym(nSym), markedEmpty(markedEmpty), paramNames(oParamNames.begin(), oParamNames.end()) {}
  PresburgerSet(PresburgerBasicSet cs);
  PresburgerSet(unsigned nDim, unsigned nSym, llvm::SmallVector<std::string, 8> oParamNames)
      : nDim(nDim), nSym(nSym), markedEmpty(false), paramNames(std::move(oParamNames)) {}
  PresburgerSet(unsigned nDim, unsigned nSym, ArrayRef<std::string> oParamNames)
      : nDim(nDim), nSym(nSym), markedEmpty(false), paramNames(oParamNames.begin(), oParamNames.end()) {}

  unsigned getNumBasicSets() const;
  unsigned getNumDims() const;
  unsigned getNumSyms() const;
  static PresburgerSet eliminateExistentials(const PresburgerBasicSet &bs);
  static PresburgerSet eliminateExistentials(const PresburgerSet &set);
  const SmallVector<PresburgerBasicSet, 4> &getBasicSets() const;
  void addBasicSet(PresburgerBasicSet cs);
  void unionSet(PresburgerSet set);
  void intersectSet(PresburgerSet set);
  static bool equal(PresburgerSet s, PresburgerSet t);
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

  ArrayRef<std::string> getParamNames() const { return paramNames; }

  void dumpParamNames() const;

  static void alignParams(PresburgerSet &s, PresburgerSet &t);

  static PresburgerSet makeEmptySet(unsigned nDim, unsigned nSym, ArrayRef<std::string> oParamNames = {});
  static PresburgerSet complement(const PresburgerSet &set);
  void subtract(PresburgerSet set);
  static PresburgerSet subtract(PresburgerBasicSet c,
                                const PresburgerSet &set);

  llvm::Optional<SmallVector<int64_t, 8>> findIntegerSample();
  bool isIntegerEmpty();
  // bool containsPoint(const std::vector<INT> &values) const;
  llvm::Optional<SmallVector<int64_t, 8>> maybeGetCachedSample() const;

private:
  void insertParametricDimensions(unsigned pos, unsigned count);
  void swapDimensions(unsigned i, unsigned j);


  unsigned nDim;
  unsigned nSym;
  SmallVector<PresburgerBasicSet, 4> basicSets;
  // This is NOT just cached information about the constraints in basicSets.
  // If this is set to true, then the set is empty, irrespective of the state
  // of basicSets.
  bool markedEmpty;
  Optional<SmallVector<int64_t, 8>> maybeSample;
  void printBasicSet(raw_ostream &os, PresburgerBasicSet cs) const;
  void printVar(raw_ostream &os, int64_t var, unsigned i,
                unsigned &countNonZero) const;
  llvm::SmallVector<std::string, 8> paramNames;
};

} // namespace presburger
} // namespace analysis
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_SET_H
