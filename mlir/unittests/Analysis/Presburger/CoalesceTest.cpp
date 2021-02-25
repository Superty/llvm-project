#include "mlir/Analysis/Presburger/Coalesce.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {

/// Construct a FlatAffineConstraints from a set of inequality and
/// equality constraints.
static FlatAffineConstraints
makeFACFromConstraints(unsigned dims, ArrayRef<SmallVector<int64_t, 4>> ineqs,
                       ArrayRef<SmallVector<int64_t, 4>> eqs) {
  FlatAffineConstraints fac(ineqs.size(), eqs.size(), dims + 1, dims);
  for (const SmallVector<int64_t, 4> &eq : eqs)
    fac.addEquality(eq);
  for (const SmallVector<int64_t, 4> &ineq : ineqs)
    fac.addInequality(ineq);
  return fac;
}

static FlatAffineConstraints
makeFACFromIneqs(unsigned dims, ArrayRef<SmallVector<int64_t, 4>> ineqs) {
  return makeFACFromConstraints(dims, ineqs, {});
}

static PresburgerSet makeSetFromFACs(unsigned dims,
                                     ArrayRef<FlatAffineConstraints> facs) {
  PresburgerSet set = PresburgerSet::getEmptySet(dims);
  for (const FlatAffineConstraints &fac : facs)
    set.unionFACInPlace(fac);
  return set;
}

void expectCoalesce(size_t expectedNumBasicSets, PresburgerSet set) {
  PresburgerSet newSet = coalesce(set);
  set.dump();
  newSet.dump();
  EXPECT_TRUE(set.isEqual(newSet));
  EXPECT_TRUE(expectedNumBasicSets == newSet.getNumFACs());
}

TEST(CoalesceTest, containedOneDim) {
  PresburgerSet set =
      makeSetFromFACs(1, {
                             makeFACFromIneqs(1, {{1, 0},    // x >= 0.
                                                  {-1, 4}}), // x <= 4.
                             makeFACFromIneqs(1, {{1, -1},   // x >= 1.
                                                  {-1, 2}}), // x <= 2.
                         });
  expectCoalesce(1, set);
}

TEST(CoalesceTest, separateOneDim) {
  PresburgerSet set =
      makeSetFromFACs(1, {
                             makeFACFromIneqs(1, {{1, 0},    // x >= 0.
                                                  {-1, 2}}), // x <= 2.
                             makeFACFromIneqs(1, {{1, -3},   // x >= 3.
                                                  {-1, 4}}), // x <= 4.
                         });
  expectCoalesce(2, set);
}

TEST(CoalesceTest, separateTwoDim) {
  PresburgerSet set =
      makeSetFromFACs(2, {
                             makeFACFromIneqs(2, {{1, 0, 0},    // x >= 0.
                                                  {-1, 0, 3},   // x <= 3.
                                                  {0, 1, 0},    // y >= 0
                                                  {0, -1, 1}}), // y <= 1.
                             makeFACFromIneqs(2, {{1, 0, 0},    // x >= 0.
                                                  {-1, 0, 3},   // x <= 3.
                                                  {0, 1, -2},   // y >= 2.
                                                  {0, -1, 3}}), // y <= 3
                         });
  expectCoalesce(2, set);
}

TEST(CoalesceTest, containedTwoDim) {
  PresburgerSet set =
      makeSetFromFACs(2, {
                             makeFACFromIneqs(2, {{1, 0, 0},    // x >= 0.
                                                  {-1, 0, 3},   // x <= 3.
                                                  {0, 1, 0},    // y >= 0
                                                  {0, -1, 3}}), // y <= 3.
                             makeFACFromIneqs(2, {{1, 0, 0},    // x >= 0.
                                                  {-1, 0, 3},   // x <= 3.
                                                  {0, 1, -2},   // y >= 2.
                                                  {0, -1, 3}}), // y <= 3
                         });
  expectCoalesce(1, set);
}

} // namespace mlir
