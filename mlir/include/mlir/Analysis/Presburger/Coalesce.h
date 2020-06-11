#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Dialect/Presburger/Set.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"

namespace mlir {
using namespace mlir::presburger;

// coalesce a set according to the paper.
//
PresburgerSet coalesce(PresburgerSet &set);

void dump(SmallVector<int64_t, 8> &cons);

// compare two constraints and gives true, even if they are multiples of each
// other
bool sameConstraint(SmallVector<int64_t, 8> c1, SmallVector<int64_t, 8> c2);

// add eq as two inequalities to ineq
void addAsIneq(SmallVector<SmallVector<int64_t, 8>, 8> &eq, SmallVector<SmallVector<int64_t, 8>, 8> &target);

// compute wrapping
Optional<SmallVector<int64_t, 8>> wrapping(FlatAffineConstraints bs, SmallVector<int64_t, 8> valid,
                                 SmallVector<int64_t, 8> invalid);

// combine to constraints with the ratio
SmallVector<int64_t, 8> combineConstraint(SmallVector<int64_t, 8> c1, SmallVector<int64_t, 8> c2, Fraction<int64_t> ratio);

// return whether the facet of ineq, a constraint of bs, is contained within a
// polytope that has cut constraints cut
bool containedFacet(SmallVector<int64_t, 8> &ineq, FlatAffineConstraints &bs,
                  SmallVector<SmallVector<int64_t, 8>, 8> &cut);

} // namespace mlir

