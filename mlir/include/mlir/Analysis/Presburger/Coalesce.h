#include "mlir/Analysis/Presburger/Coalesce.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Dialect/Presburger/Set.h"


namespace mlir {
using namespace mlir::presburger;

  // coalesce a set according to the paper.
  //
  PresburgerSet coalesce(PresburgerSet &set);

  // compare two constraints and gives true, even if they are multiples of each
  // other
  /*bool sameConstraint(Constraint c1, Constraint c2);
  */
  // add eq as two inequalities to ineq
  void addAsIneq(SmallVector<ArrayRef<int64_t>, 64> eq, SmallVector<ArrayRef<int64_t>, 64> &target);
  /*
  // compute wrapping
  std::optional<ArrayRef<int64_t> wrapping(BasicSet bs, Constraint valid,
                                   Constraint invalid);

  // combine to constraints with the ratio
  Constraint combineConstraint(Constraint c1, Constraint c2, SafeRational ratio);

  // return whether the facet of ineq, a constraint of bs, is contained within a
  // polytope that has cut constraints cut
  bool containedFacet(Constraint &ineq, BasicSet &bs,
                    std::vector<ArrayRef<int64_t> &cut);
  */
} // namespace mlir

