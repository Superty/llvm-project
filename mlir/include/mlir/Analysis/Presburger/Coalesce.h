#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Presburger/Set.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {

// coalesces a set according to the "integer set coalescing" by sven
// verdoolaege.
//
// Coalescing task two convex BasicSets and tries to figure out, whether the
// convex hull of those two BasicSets is the same integer set as the union of
// those two BasicSets and if so, tries to come up with a BasicSet corresponding
// to this convex hull.
PresburgerSet coalesce(PresburgerSet &set);

void dump(const ArrayRef<int64_t> cons);

// compare two constraints and gives true, even if they are stretched
bool sameConstraint(ArrayRef<int64_t> c1, ArrayRef<int64_t> c2);

// tries to find a constraint, that is a linear combination of valid and
// invalid, and is touches bs/is REDUNDANT in the most limited way possible
Optional<SmallVector<int64_t, 8>> wrapping(const FlatAffineConstraints &bs,
                                           SmallVectorImpl<int64_t> &valid,
                                           SmallVectorImpl<int64_t> &invalid);

// combine two constraints c1 >= 0 and c2 >= 0 with the ratio n/d as -n c1 + d
// c2 >= 0
SmallVector<int64_t, 8>
combineConstraint(ArrayRef<int64_t> c1, ArrayRef<int64_t> c2, Fraction &ratio);

// takes a BasicSet bs, a constraint ineq of that basicSet and the vector cut of
// constraints, that were typed as cutting bs. Computes wether the part of ineq,
// that lies within bs, is redundant for all constraints of cut
bool containedFacet(ArrayRef<int64_t> ineq, const FlatAffineConstraints &bs,
                    const SmallVector<ArrayRef<int64_t>, 8> &cut);

} // namespace mlir
