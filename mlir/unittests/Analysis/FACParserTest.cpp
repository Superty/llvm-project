#include "./ParserUtil.h"
#include "mlir/Analysis/PresburgerSet.h"

#include <gtest/gtest.h>

namespace mlir {

/// Construct a FlatAffineConstraints from a set of inequality and
/// equality constraints.
static FlatAffineConstraints
makeFACFromConstraints(unsigned ids, ArrayRef<SmallVector<int64_t, 4>> ineqs,
                       ArrayRef<SmallVector<int64_t, 4>> eqs,
                       unsigned syms = 0) {
  FlatAffineConstraints fac(ineqs.size(), eqs.size(), ids + 1, ids - syms, syms,
                            /*numLocals=*/0);
  for (const auto &eq : eqs)
    fac.addEquality(eq);
  for (const auto &ineq : ineqs)
    fac.addInequality(ineq);
  return fac;
}

static bool facEquality(FlatAffineConstraints &fac1,
                        FlatAffineConstraints &fac2) {
  return PresburgerSet(fac1).isEqual(PresburgerSet(fac2));
}

TEST(ParseFACTest, simple) {
  FailureOr<FlatAffineConstraints> fac =
      parseFAC("(x)[] : (7 * x >= 0, -7 * x + 5 >= 0)");
  EXPECT_TRUE(succeeded(fac));

  FlatAffineConstraints ex = makeFACFromConstraints(1, {{7, 0}, {-7, 5}}, {});
  EXPECT_TRUE(facEquality(ex, *fac));
}

} // namespace mlir
