#include "mlir/Analysis/PresburgerSet.h"
#include "mlir/Parser.h"

#include <gtest/gtest.h>

namespace mlir {

/// Construct a FlatAffineConstraints from a set of inequality, equality, and
/// division onstraints.
static FlatAffineConstraints makeFACFromConstraints(
    unsigned dims, unsigned syms, ArrayRef<SmallVector<int64_t, 4>> ineqs,
    ArrayRef<SmallVector<int64_t, 4>> eqs = {},
    ArrayRef<std::pair<SmallVector<int64_t, 4>, int64_t>> divs = {}) {
  FlatAffineConstraints fac(ineqs.size(), eqs.size(), dims + syms + 1, dims,
                            syms, 0);
  for (const auto &div : divs) {
    fac.addLocalFloorDiv(div.first, div.second);
  }
  for (const auto &eq : eqs)
    fac.addEquality(eq);
  for (const auto &ineq : ineqs)
    fac.addInequality(ineq);
  return fac;
}

TEST(ParseFACTest, InvalidInputTest) {
  MLIRContext context;
  FailureOr<FlatAffineConstraints> fac;

  fac = parseFlatAffineConstraints("(x)", &context);
  EXPECT_TRUE(failed(fac))
      << "should not accept stings with no constraint list";

  fac = parseFlatAffineConstraints("(x)[] : ())", &context);
  EXPECT_TRUE(failed(fac))
      << "should not accept stings that contain remaining characters";

  fac = parseFlatAffineConstraints("(x)[] : (x - >= 0)", &context);
  EXPECT_TRUE(failed(fac))
      << "should not accept stings that contain incomplete constraints";

  fac = parseFlatAffineConstraints("(x)[] : (y == 0)", &context);
  EXPECT_TRUE(failed(fac))
      << "should not accept stings that contain unkown identifiers";

  fac = parseFlatAffineConstraints("(x, x) : (2 * x >= 0)", &context);
  EXPECT_TRUE(failed(fac))
      << "should not accept stings that contain repeated identifier names";

  fac = parseFlatAffineConstraints("(x)[x] : (2 * x >= 0)", &context);
  EXPECT_TRUE(failed(fac))
      << "should not accept stings that contain repeated identifier names";

  fac = parseFlatAffineConstraints("(x) : (2 * x + 9223372036854775808 >= 0)",
                                   &context);
  EXPECT_TRUE(failed(fac)) << "should not accept stings with integer literals "
                              "that do not fit into int64_t";
}

static bool facEquality(FlatAffineConstraints &fac1,
                        FlatAffineConstraints &fac2) {
  return PresburgerSet(fac1).isEqual(PresburgerSet(fac2));
}

static bool parseAndCompare(StringRef str, FlatAffineConstraints ex) {
  MLIRContext context;
  FailureOr<FlatAffineConstraints> fac =
      parseFlatAffineConstraints(str, &context);

  EXPECT_TRUE(succeeded(fac));

  return facEquality(ex, *fac);
}

TEST(ParseFACTest, ParseAndCompareTest) {
  // simple ineq
  EXPECT_TRUE(parseAndCompare("(x)[] : (x >= 0)",
                              makeFACFromConstraints(1, 0, {{1, 0}})));

  // simple eq
  EXPECT_TRUE(parseAndCompare("(x)[] : (x == 0)",
                              makeFACFromConstraints(1, 0, {}, {{1, 0}})));

  // multiple constraints
  EXPECT_TRUE(parseAndCompare("(x)[] : (7 * x >= 0, -7 * x + 5 >= 0)",
                              makeFACFromConstraints(1, 0, {{7, 0}, {-7, 5}})));

  // multiple dimensions
  EXPECT_TRUE(parseAndCompare("(x,y,z)[] : (x + y - z >= 0)",
                              makeFACFromConstraints(3, 0, {{1, 1, -1, 0}})));

  // dimensions and symbols
  EXPECT_TRUE(
      parseAndCompare("(x,y,z)[a,b] : (x + y - z + 2 * a - 15 * b >= 0)",
                      makeFACFromConstraints(3, 2, {{1, 1, -1, 2, -15, 0}})));

  // only symbols
  EXPECT_TRUE(parseAndCompare("()[a] : (2 * a - 4 == 0)",
                              makeFACFromConstraints(0, 1, {}, {{2, -4}})));

  // simple floordiv
  EXPECT_TRUE(parseAndCompare(
      "(x, y) : (y - 3 * ((x + y - 13) floordiv 3) - 42 == 0)",
      makeFACFromConstraints(2, 0, {}, {{0, 1, -3, -42}}, {{{1, 1, -13}, 3}})));

  // multiple floordiv
  EXPECT_TRUE(parseAndCompare(
      "(x, y) : (y - x floordiv 3 - y floordiv 2 == 0)",
      makeFACFromConstraints(2, 0, {}, {{0, 1, -1, -1, 0}},
                             {{{1, 0, 0}, 3}, {{0, 1, 0, 0}, 2}})));

  // nested floordiv
  EXPECT_TRUE(parseAndCompare(
      "(x, y) : (y - (x + y floordiv 2) floordiv 3 == 0)",
      makeFACFromConstraints(2, 0, {}, {{0, 1, 0, -1, 0}},
                             {{{0, 1, 0}, 2}, {{1, 0, 1, 0}, 3}})));
}

} // namespace mlir
