#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Presburger/Set.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {

/// coalesces a set according to the "integer set coalescing" by sven
/// verdoolaege.
///
/// Coalescing takes two convex BasicSets and tries to figure out, whether the
/// convex hull of those two BasicSets is the same integer set as the union of
/// those two BasicSets and if so, tries to come up with a BasicSet
/// corresponding to this convex hull.
///
/// If preserve is true, then does not add additional existentials in attempt to
/// coalesce.
template <typename Int>
PresburgerSet<Int> coalesce(PresburgerSet<Int> &set,
                            bool preserve = false);

template <typename Int>
void dump(ArrayRef<Int> cons);

/// compare two constraints and give true if they are equal. Can also handle
/// cases in which for some integers a and b, c1 = a/b * c2. This is the same
/// constraint but stretched, which doesn't influence it's hyperplane.
template <typename Int>
bool sameConstraint(ArrayRef<Int> c1, ArrayRef<Int> c2);

/// rotates invalid around valid, until it becomes redundant. It does this by
/// adding the smallest multiple of valid to invalid, such that the result is
/// redundant.
template <typename Int>
Optional<SmallVector<Int, 8>>
wrapping(const PresburgerBasicSet<Int> &bs, SmallVectorImpl<Int> &valid,
         SmallVectorImpl<Int> &invalid);

/// combine two constraints c1 >= 0 and c2 >= 0 with the ratio n/d as -n c1 + d
/// c2 >= 0
template <typename Int>
SmallVector<Int, 8> combineConstraint(ArrayRef<Int> c1,
                                              ArrayRef<Int> c2,
                                              Fraction<Int> &ratio);

/// takes a BasicSet bs, a constraint ineq of that basicSet and the vector cut
/// of constraints, that were typed as cutting bs. Computes wether the part of
/// bs, that satisfies ineq with equality, is redundant for all constraints of
/// cut
template <typename Int>
bool containedFacet(ArrayRef<Int> ineq, const PresburgerBasicSet<Int> &bs,
                    ArrayRef<ArrayRef<Int>> cut);


} // namespace mlir
