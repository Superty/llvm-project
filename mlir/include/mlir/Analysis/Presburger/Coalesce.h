#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Dialect/Presburger/Set.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
using namespace mlir::presburger;

// coalesce a set according to the paper.
//
PresburgerSet coalesce(PresburgerSet &set);

void dump(SmallVectorImpl<int64_t> &cons);

// compare two constraints and gives true, even if they are multiples of each
// other
bool sameConstraint(ArrayRef<int64_t> c1, ArrayRef<int64_t> c2);

// add eq as two inequalities to ineq
void addAsIneq(ArrayRef<SmallVector<int64_t, 8>> eq,
               SmallVectorImpl<SmallVector<int64_t, 8>> &target);

// compute wrapping
Optional<SmallVector<int64_t, 8>> wrapping(FlatAffineConstraints &bs,
                                           SmallVectorImpl<int64_t> &valid,
                                           SmallVectorImpl<int64_t> &invalid);

// combine to constraints with the ratio
SmallVector<int64_t, 8> combineConstraint(ArrayRef<int64_t> c1,
                                          ArrayRef<int64_t> c2,
                                          Fraction<int64_t> ratio);

// return whether the facet of ineq, a constraint of bs, is contained within a
// polytope that has cut constraints cut
bool containedFacet(ArrayRef<int64_t> ineq, FlatAffineConstraints &bs,
                    SmallVector<SmallVector<int64_t, 8>, 8> &cut);

} // namespace mlir
