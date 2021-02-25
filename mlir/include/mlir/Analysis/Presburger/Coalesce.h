#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Analysis/PresburgerSet.h"

namespace mlir {

/// coalesces a set according to the "integer set coalescing" by sven
/// verdoolaege.
///
/// Coalescing takes two convex BasicSets and tries to figure out, whether the
/// convex hull of those two BasicSets is the same integer set as the union of
/// those two BasicSets and if so, tries to come up with a BasicSet
/// corresponding to this convex hull.
PresburgerSet coalesce(PresburgerSet &set);

} // namespace mlir
