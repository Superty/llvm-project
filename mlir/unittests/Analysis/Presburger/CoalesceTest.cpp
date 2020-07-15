#include "mlir/Analysis/Presburger/Coalesce.h"
#include "mlir/Dialect/Presburger/Parser.h"
#include <fstream>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iostream>
#include <string>

using namespace mlir;
using namespace mlir::presburger;

PresburgerSet setFromString(StringRef string) {
  ErrorCallback callback = [](SMLoc loc, const Twine &message) {
    // This is a hack to make the Parser compile
    llvm_unreachable("Parser Callback");
    MLIRContext context;
    return mlir::emitError(UnknownLoc::get(&context), message);
  };
  PresburgerSetParser parser(string, callback);
  PresburgerSet res;
  parser.parsePresburgerSet(res);
  return res;
}

void expectContainedFacet(bool expected, SmallVector<int64_t, 8> &ineq,
                          FlatAffineConstraints &bs,
                          SmallVector<ArrayRef<int64_t>, 8> &cut) {
  EXPECT_TRUE(expected == containedFacet(ineq, bs, cut));
}

TEST(CoalesceTest, containedFacet1) {
  PresburgerSet set =
      setFromString("(x, y) : (x >= 0 and x <= 3 and y >= 0 and y <= 3)");
  FlatAffineConstraints bs = set.getFlatAffineConstraints()[0];
  SmallVector<int64_t, 8> ineq = {1, 0, 0};
  SmallVector<int64_t, 8> cutConstraint1 = {1, 0, 2};
  SmallVector<ArrayRef<int64_t>, 8> cutting = {cutConstraint1};
  expectContainedFacet(true, ineq, bs, cutting);
}

TEST(CoalesceTest, containedFacet2) {
  PresburgerSet set =
      setFromString("(x, y) : (x >= 0 and x <= 3 and y >= 0 and y <= 3)");
  FlatAffineConstraints bs = set.getFlatAffineConstraints()[0];
  SmallVector<int64_t, 8> ineq = {1, 0, 0};
  SmallVector<int64_t, 8> cutConstraint1 = {1, 0, 2};
  SmallVector<int64_t, 8> cutConstraint2 = {1, 0, 1};
  SmallVector<ArrayRef<int64_t>, 8> cutting = {cutConstraint1, cutConstraint2};
  expectContainedFacet(true, ineq, bs, cutting);
}

TEST(CoalesceTest, containedFacet3) {
  PresburgerSet set =
      setFromString("(x, y) : (x >= 0 and x <= 3 and y >= 0 and y <= 3)");
  FlatAffineConstraints bs = set.getFlatAffineConstraints()[0];
  SmallVector<int64_t, 8> ineq = {1, 0, 0};
  SmallVector<int64_t, 8> cutConstraint1 = {1, 0, -5};
  SmallVector<ArrayRef<int64_t>, 8> cutting = {cutConstraint1};
  expectContainedFacet(false, ineq, bs, cutting);
}

TEST(CoalesceTest, containedFacet4) {
  PresburgerSet set =
      setFromString("(x, y) : (x >= 0 and x <= 3 and y >= 0 and y <= 3)");
  FlatAffineConstraints bs = set.getFlatAffineConstraints()[0];
  SmallVector<int64_t, 8> ineq = {1, 0, 0};
  SmallVector<int64_t, 8> cutConstraint1 = {0, 1, 2};
  SmallVector<int64_t, 8> cutConstraint2 = {1, 0, -5};
  SmallVector<ArrayRef<int64_t>, 8> cutting = {cutConstraint1, cutConstraint2};
  expectContainedFacet(false, ineq, bs, cutting);
}

void expectWrapping(Optional<SmallVector<int64_t, 8>> expected,
                    FlatAffineConstraints bs, SmallVector<int64_t, 8> valid,
                    SmallVector<int64_t, 8> invalid) {
  if (!expected) {
    EXPECT_FALSE(wrapping(bs, valid, invalid).hasValue());
  } else {
    auto result = wrapping(bs, valid, invalid);
    EXPECT_TRUE(result.hasValue());
    if (!result) {
      EXPECT_TRUE(false);
    } else {
      EXPECT_TRUE(sameConstraint(result.getValue(), expected.getValue()));
    }
  }
}

TEST(CoalesceTest, wrapping) {
  PresburgerSet set1 = setFromString("(x,y) : (x = y and y <= 4 and y >= 0)");
  FlatAffineConstraints bs1 = set1.getFlatAffineConstraints()[0];
  SmallVector<int64_t, 8> valid1 = {-1, 1, 1};
  SmallVector<int64_t, 8> invalid1 = {0, -1, 3};
  SmallVector<int64_t, 8> expected1 = {-1, 0, 4};
  expectWrapping(expected1, bs1, valid1, invalid1);
  PresburgerSet set2 = setFromString("(x,y) : (2x = 3y and x >= 0 and x <= 6)");
  FlatAffineConstraints bs2 = set2.getFlatAffineConstraints()[0];
  SmallVector<int64_t, 8> valid2 = {-2, 3, 1};
  SmallVector<int64_t, 8> invalid2 = {0, 1, -1};
  SmallVector<int64_t, 8> expected2 = {-1, 2, 0};
  expectWrapping(expected2, bs2, valid2, invalid2);
}

void expectCombineConstraint(SmallVector<int64_t, 8> expected,
                             SmallVector<int64_t, 8> c1,
                             SmallVector<int64_t, 8> c2, Fraction ratio) {
  SmallVector<int64_t, 8> result = combineConstraint(c1, c2, ratio);
  EXPECT_TRUE(sameConstraint(expected, result));
}

TEST(CoalesceTest, combineConstraint) {
  SmallVector<int64_t, 8> c1 = {0, 1, 1};
  SmallVector<int64_t, 8> c2 = {3, 1, 5};
  SmallVector<int64_t, 8> expected = {6, -1, 7};
  Fraction ratio1(0, 1);
  Fraction ratio2(3, 2);
  expectCombineConstraint(c2, c1, c2, ratio1);
  expectCombineConstraint(expected, c1, c2, ratio2);
}

void expectCoalesce(size_t expectedNumBasicSets, PresburgerSet set) {
  PresburgerSet newSet = coalesce(set);
  set.dump();
  newSet.dump();
  EXPECT_TRUE(PresburgerSet::equal(set, newSet));
  EXPECT_TRUE(expectedNumBasicSets == newSet.getFlatAffineConstraints().size());
}

/*TEST(CoalesceTest, failing) {
  PresburgerSet curr =
      setFromString("(d0, d1)[] : (-d0 + -2147483650 >= 0  or d0 + -2499 >= 0  "
                    "or d0  >= 0 and -d0 + 2498 >= 0 and d1  >= 0 )");
  PresburgerSet newSet = coalesce(curr);
  EXPECT_TRUE(PresburgerSet::equal(newSet, curr));
}*/

TEST(CoalesceTest, performance) {
  std::ifstream newfile("new_tests.txt");
  std::string curr;
  int i = 0;
  while (std::getline(newfile, curr)) {
    i++;
    std::cout << i << std::endl;
    PresburgerSet currentSet = setFromString(curr);
    PresburgerSet newSet = coalesce(currentSet);
    if (!PresburgerSet::equal(newSet, currentSet)) {
      newSet.dump();
      currentSet.dump();
      EXPECT_TRUE(false);
    }
  }
  newfile.close();
}

TEST(CoalesceTest, contained) {
  PresburgerSet contained =
      setFromString("(x0) : (x0 >= 0 and x0 <= 4 or x0 >= 1 and x0 <= 3)");
  expectCoalesce(1, contained);
}

TEST(CoalesceTest, cut) {
  PresburgerSet cut =
      setFromString("(x0) : (x0 >= 0 and x0 <= 3 or x0 >= 2 and x0 <= 4)");
  expectCoalesce(1, cut);
}

TEST(CoalesceTest, adjIneq) {
  PresburgerSet adjIneq =
      setFromString("(x0) : (x0 >= 0 and x0 <= 1 or x0 >= 2 and x0 <= 3)");
  expectCoalesce(1, adjIneq);
}

TEST(CoalesceTest, separate) {
  PresburgerSet separate =
      setFromString("(x0) : (x0 >= 0 and x0 <= 1 or x0 >= 3 and x0 <= 4)");
  expectCoalesce(2, separate);
}

TEST(CoalesceTest, adjEq) {
  PresburgerSet adjEq = setFromString("(x0) : (x0 = 1 or x0 >= 2 and x0 <= 3)");
  expectCoalesce(1, adjEq);
}

TEST(CoalesceTest, adjEqs) {
  PresburgerSet adjEqs = setFromString("(x0) : (x0 = 1 or x0 = 2)");
  expectCoalesce(1, adjEqs);
}

TEST(CoalesceTest, eqSeparate) {
  PresburgerSet eqSeparate =
      setFromString(" (x0) : (x0 = 0 or x0 >= 2 and x0 <= 3)");
  expectCoalesce(2, eqSeparate);
}

TEST(CoalesceTest, eqAsIneqContained) {
  PresburgerSet eqAsIneqContained =
      setFromString(" (x0) : (x0 <= 3 and x0 >= 3 or x0 >=2 and x0 <= 3)");
  expectCoalesce(1, eqAsIneqContained);
}

TEST(CoalesceTest, eqContained) {
  PresburgerSet eqContained =
      setFromString(" (x0) : (x0 = 3 or x0 >= 2 and x0 <= 3)");
  expectCoalesce(1, eqContained);
}

TEST(CoalesceTest, multiDimContained) {
  PresburgerSet multiDimContained =
      setFromString(" (x0, x1) : (x0 >= 0 and x0 <= 4 and x1 >= 0 and x1 <= 4 "
                    "or x0 >= 2 and x0 <= 3 and x1 >= 2 and x1 <= 3)");
  expectCoalesce(1, multiDimContained);
}

TEST(CoalesceTest, multiDimAdjIneq) {
  PresburgerSet multiDimAdjIneq = setFromString(
      " (x0, x1) : (x0 >= 0 and x0 <= 3 and x1 >= 0 and x1 <= 1 or x0 >= 0 and "
      "x0 <= 3 and x1 <= 2 and x1 <= 3)");
  expectCoalesce(1, multiDimAdjIneq);
}

TEST(CoalesceTest, multiDimSeparate) {
  PresburgerSet multiDimSeparate = setFromString(
      "(x0, x1) : (x0 >= 0 and x0 <= 1 and x1 >= 0 and x1 <= 1 or x0 >= 2 and "
      "x0 <= 3 and x1 >= 2 and x1 <= 3)");
  expectCoalesce(2, multiDimSeparate);
}

TEST(CoalesceTest, multiDimCut) {
  PresburgerSet multiDimCut = setFromString(
      "(x,y) : (4x -5y <= 0 and y <= 4 and 3x + 4y >= 0 and x - y + 7 >= 0 or "
      "y >= 0 and x <= 0 and y <= 4 and x - y + 9 >= 0 and 2x + 5y + 4 >= 0)");
  expectCoalesce(1, multiDimCut);
}

TEST(CoalesceTest, multiDimNonCut) {
  PresburgerSet multiDimNonCut =
      setFromString("(x,y) :  (y >= 0 and x <= 0 and y <= 4 and x - y + 9 "
                    ">= 0 and 2x + 5y + 4 >= 0 or 4x -5y <= 0 and 3x + 4y "
                    ">= 0 and x - y + 7 >= 0 and x + y - 10 <= 0)");
  expectCoalesce(2, multiDimNonCut);
}

TEST(CoalesceTest, multiDimAdjEqs) {
  PresburgerSet multiDimAdjEqs =
      setFromString("(x,y) :  (y = x and y >= 0 and y <= 4 or x - 1 = y "
                    "and y >= 0 and y <= 3)");
  expectCoalesce(1, multiDimAdjEqs);
}

TEST(CoalesceTest, multiDimAdjEqs2) {
  PresburgerSet multiDimAdjEqs2 =
      setFromString("(x,y) :  (y = x and y >= 0 and y <= 4 or x - 1 = y "
                    "and y >= 1 and y <= 3)");
  expectCoalesce(1, multiDimAdjEqs2);
}

TEST(CoalesceTest, multiDimAdjEqs3) {
  PresburgerSet multiDimAdjEqs3 =
      setFromString("(x,y) : ( 2x = 3y and y >= 0 and x <= 6 or 2x = 3y+1 "
                    "and y >= 2 and y <= 5)");
  expectCoalesce(1, multiDimAdjEqs3);
}

TEST(CoalesceTest, multiDimAdjEqToPoly) {
  PresburgerSet multiDimAdjEqToPoly = setFromString(
      "(x,y) :  (x <= 0 and y >= 0 and y <= 4 and x >= -8 and x <= -y + 3 or x "
      "= 1 and y >= 0 and y <= 2)");
  expectCoalesce(1, multiDimAdjEqToPoly);
}

TEST(CoalesceTest, multiDimAdjEqToPolyComplex) {
  PresburgerSet multiDimAdjEqToPolyComplex = setFromString(
      "(x,y) :  (x <= 0 and y >= 0 and y <= 4 and x >= -8 and x <= -y + 3 or x "
      "= 1 and y >= 1 and y <= 2)");
  expectCoalesce(1, multiDimAdjEqToPolyComplex);
}

TEST(CoalesceTest, multiSetsAllContained) {
  PresburgerSet multiSetsAllContained =
      setFromString("(x0) : (x0 >= 0 and x0 <= 5 or x0 >= 2 and x0 "
                    "<= 3 or x0 >= 4 and x0 <= 5)");
  expectCoalesce(1, multiSetsAllContained);
}

TEST(CoalesceTest, multiSetsOneContained) {
  PresburgerSet multiSetsOneContained =
      setFromString("(x0) : (x0 >= 0 and x0 <= 3 or x0 >= 2 and x0 "
                    "<= 3 or x0 >= 2 and x0 <= 5)");
  expectCoalesce(1, multiSetsOneContained);
}

TEST(CoalesceTest, multiSetsNotContained) {
  PresburgerSet multiSetsNotContained =
      setFromString("(x0) : (x0 >= 0 and x0 <= 1 or x0 >= 2 and x0 "
                    "<= 3 or x0 >= 4 and x0 <= 5)");
  expectCoalesce(1, multiSetsNotContained);
}

TEST(CoalesceTest, protrusion) {
  PresburgerSet protrusion = setFromString(
      "(x,y) : (x >= 0 and y >= 0 and y <= 3 and x <= 9 or y >= 1 "
      "and y <= 4 and y <= x - 1 and x + y - 11 <= 0)");
  expectCoalesce(1, protrusion);
}

TEST(CoalesceTest, nearlyProtrusion) {
  PresburgerSet nearlyProtrusion = setFromString(
      "(x,y) : (x >= 0 and y >= 0 and y <= 3 and x <= 9 or y >= 1 "
      "and y <= 5 and y <= x - 1 and x + y - 11 <= 0)");
  expectCoalesce(2, nearlyProtrusion);
}

/*TEST(CoalesceTest, twoAdj) {
  PresburgerSet twoAdj = setFromString(
      "(x,y) : (x = 1 and y >= 0 and y <= 2 or x = 2 and y >= 3 and y <= 5)");
  // The result should be something like: "(x,y) : ( x >= 1 and x <= 2 and 3x -y
  // -3 <= 0 and 3x -y-1 >= 0)");
  expectCoalesce(1, twoAdj);
}*/
