#include "mlir/Analysis/Presburger/Coalesce.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Analysis/PresburgerSet.h"

using namespace mlir;

/// eliminates FlatAffineConstraints, that are fully contained within other
/// FlatAffineConstraints, from the representation of set
PresburgerSet mlir::coalesce(PresburgerSet &set) {
  PresburgerSet newSet =
      PresburgerSet::getEmptySet(set.getNumDims(), set.getNumSyms());
  ArrayRef<FlatAffineConstraints> basicSetVector =
      set.getAllFlatAffineConstraints();
  SmallVector<bool, 4> marked(set.getNumFACs());

  for (unsigned i = 0; i < basicSetVector.size(); i++) {
    if (marked[i])
      continue;
    FlatAffineConstraints bs1 = basicSetVector[i];

    unsigned numIneq = bs1.getNumInequalities();
    unsigned numEq = bs1.getNumEqualities();
    // check if bs1 is contained in any basicSet
    for (unsigned j = 0; j < basicSetVector.size(); j++) {
      if (j == i || marked[j])
        continue;
      FlatAffineConstraints bs2 = basicSetVector[j];
      Simplex simplex(bs2);

      bool contained = true;
      for (unsigned i = 0; i < numIneq; ++i) {
        contained &= simplex.isRedundant(bs1.getInequality(i));
      }
      for (unsigned i = 0; i < numEq; ++i) {
        contained &= simplex.isRedundant(bs1.getEquality(i));
      }

      if (contained) {
        marked[j] = true;
      }
    }

    if (!marked[i]) {
      newSet.unionFACInPlace(bs1);
    }
  }

  return newSet;
}
