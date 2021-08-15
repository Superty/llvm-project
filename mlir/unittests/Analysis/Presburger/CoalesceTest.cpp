#include "mlir/Analysis/Presburger/Presburger-impl.h"
#include <fstream>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iostream>
#include <string>

using Int = int64_t;

using namespace mlir;
using namespace mlir::presburger;

static PresburgerSet<Int> setFromString(StringRef string) {
  ErrorCallback callback = [](SMLoc loc, const Twine &message) {
    // This is a hack to make the Parser compile
    llvm_unreachable("Parser Callback");
    MLIRContext context;
    return mlir::emitError(UnknownLoc::get(&context), message);
  };
  Parser<Int> parser(string, callback);
  PresburgerParser<Int> setParser(parser);
  PresburgerSet<Int> res;
  setParser.parsePresburgerSet(res);
  return res;
}

void expectContainedFacet(bool expected, SmallVector<Int, 8> &ineq,
                          PresburgerBasicSet<Int> &bs,
                          SmallVector<ArrayRef<Int>, 8> &cut) {
  EXPECT_TRUE(expected == containedFacet<Int>(ineq, bs, cut));
}

TEST(CoalesceTest, containedFacet1) {
  PresburgerSet<Int> set =
      setFromString("(x, y) : (x >= 0 and x <= 3 and y >= 0 and y <= 3)");
  PresburgerBasicSet<Int> bs = set.getBasicSets()[0];
  SmallVector<Int, 8> ineq = {1, 0, 0};
  SmallVector<Int, 8> cutConstraint1 = {1, 0, 2};
  SmallVector<ArrayRef<Int>, 8> cutting = {cutConstraint1};
  expectContainedFacet(true, ineq, bs, cutting);
}

TEST(CoalesceTest, containedFacet2) {
  PresburgerSet<Int> set =
      setFromString("(x, y) : (x >= 0 and x <= 3 and y >= 0 and y <= 3)");
  PresburgerBasicSet<Int> bs = set.getBasicSets()[0];
  SmallVector<Int, 8> ineq = {1, 0, 0};
  SmallVector<Int, 8> cutConstraint1 = {1, 0, 2};
  SmallVector<Int, 8> cutConstraint2 = {1, 0, 1};
  SmallVector<ArrayRef<Int>, 8> cutting = {cutConstraint1,
                                                   cutConstraint2};
  expectContainedFacet(true, ineq, bs, cutting);
}

TEST(CoalesceTest, containedFacet3) {
  PresburgerSet<Int> set =
      setFromString("(x, y) : (x >= 0 and x <= 3 and y >= 0 and y <= 3)");
  PresburgerBasicSet<Int> bs = set.getBasicSets()[0];
  SmallVector<Int, 8> ineq = {1, 0, 0};
  SmallVector<Int, 8> cutConstraint1 = {1, 0, -5};
  SmallVector<ArrayRef<Int>, 8> cutting = {cutConstraint1};
  expectContainedFacet(false, ineq, bs, cutting);
}

TEST(CoalesceTest, containedFacet4) {
  PresburgerSet<Int> set =
      setFromString("(x, y) : (x >= 0 and x <= 3 and y >= 0 and y <= 3)");
  PresburgerBasicSet<Int> bs = set.getBasicSets()[0];
  SmallVector<Int, 8> ineq = {1, 0, 0};
  SmallVector<Int, 8> cutConstraint1 = {0, 1, 2};
  SmallVector<Int, 8> cutConstraint2 = {1, 0, -5};
  SmallVector<ArrayRef<Int>, 8> cutting = {cutConstraint1,
                                                   cutConstraint2};
  expectContainedFacet(false, ineq, bs, cutting);
}

void expectWrapping(Optional<SmallVector<Int, 8>> expected,
                    PresburgerBasicSet<Int> bs, SmallVector<Int, 8> valid,
                    SmallVector<Int, 8> invalid) {
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
  PresburgerSet<Int> set1 = setFromString("(x,y) : (x = y and y <= 4 and y >= 0)");
  PresburgerBasicSet<Int> bs1 = set1.getBasicSets()[0];
  SmallVector<Int, 8> valid1 = {-1, 1, 1};
  SmallVector<Int, 8> invalid1 = {0, -1, 3};
  SmallVector<Int, 8> expected1 = {-1, 0, 4};
  expectWrapping(expected1, bs1, valid1, invalid1);
  PresburgerSet<Int> set2 = setFromString("(x,y) : (2x = 3y and x >= 0 and x <= 6)");
  PresburgerBasicSet<Int> bs2 = set2.getBasicSets()[0];
  SmallVector<Int, 8> valid2 = {-2, 3, 1};
  SmallVector<Int, 8> invalid2 = {0, 1, -1};
  SmallVector<Int, 8> expected2 = {-1, 2, 0};
  expectWrapping(expected2, bs2, valid2, invalid2);
}

void expectCombineConstraint(SmallVector<Int, 8> expected,
                             SmallVector<Int, 8> c1,
                             SmallVector<Int, 8> c2, Fraction<Int> ratio) {
  SmallVector<Int, 8> result = combineConstraint<Int>(c1, c2, ratio);
  EXPECT_TRUE(sameConstraint(expected, result));
}

TEST(CoalesceTest, combineConstraint) {
  SmallVector<Int, 8> c1 = {0, 1, 1};
  SmallVector<Int, 8> c2 = {3, 1, 5};
  SmallVector<Int, 8> expected = {6, -1, 7};
  Fraction<Int> ratio1(0, 1);
  Fraction<Int> ratio2(3, 2);
  expectCombineConstraint(c2, c1, c2, ratio1);
  expectCombineConstraint(expected, c1, c2, ratio2);
}

void expectCoalesce(size_t expectedNumBasicSets, PresburgerSet<Int> set) {
  PresburgerSet<Int> newSet = coalesce(set);
  EXPECT_TRUE(PresburgerSet<Int>::equal(set, newSet));
  EXPECT_TRUE(expectedNumBasicSets == newSet.getBasicSets().size());
}

// TEST(CoalesceTest, failing) {
//   PresburgerSet<Int> curr = setFromString(
//       "(d0, d1, d2, d3, d4, d5, d6)[] : (d5  + -3 = 0 and d4  = 0 and d2  +
//       -2 "
//       "= 0 and d0  + -1 = 0 and -d1  + 7999 >= 0 and d3  >= 0 and d1 + -d3  +
//       "
//       "-1 >= 0 and -d1 + 8d6  + 7 >= 0 and d1 + -8d6  >= 0 ) or (d5  + -2 = 0
//       " "and d4  = 0 and d2  + -2 = 0 and d0  + -1 = 0 and -d1  + 7999 >= 0
//       and " "d3  >= 0 and d1 + -d3  + -1 >= 0 and -d3 + 8d6  + 7 >= 0 and d3
//       + -8d6  "
//       ">= 0 ) or (d5  = 0 and d4  = 0 and d2  + -2 = 0 and d0  + -1 = 0 and "
//       "-d1  "
//       "+ 7999 >= 0 and d3  >= 0 and d1 + -d3  + -1 >= 0 and -d1 + 8d6  + 7 >=
//       " "0 and d1 + -8d6  >= 0 ) or (d5  + -2 = 0 and d4  = 0 and d3  = 0 and
//       d2 " " "
//       "+ -3 = 0 and d0  + -1 = 0 and d1  >= 0 and -d1  + 7999 >= 0 and -d1 +
//       " "8d6  + 7 >= 0 and d1 + -8d6  >= 0 ) or (d5  + -1 = 0 and d4  = 0 and
//       d3 " " "
//       "= 0 and d2  = 0 and d0  + -1 = 0 and d1  >= 0 and -d1  + 7999 >= 0 and
//       "
//       "-d1 + 8d6  + 7 >= 0 and d1 + -8d6  >= 0 ) or (d5  = 0 and d4  = 0 and
//       " "d3  "
//       "= 0 and d2  + -3 = 0 and d0  + -1 = 0 and d1  >= 0 and -d1  + 7999 >=
//       0 " "and -d1 + 8d6  + 7 >= 0 and d1 + -8d6  >= 0 )");
//   PresburgerSet<Int> newSet = coalesce(curr);
//   EXPECT_TRUE(PresburgerSet<Int>::equal(newSet, curr));
// }

/*TEST(CoalesceTest, performance) {
  std::ifstream newfile("new_tests.txt");
  std::string curr;
  std::ofstream f("hallo.txt");
  f << "hallo";
  f.close();
  int i = 0;
  EXPECT_TRUE(newfile.good());
  while (std::getline(newfile, curr)) {
    i++;
    PresburgerSet<Int> currentSet = setFromString(curr);
    PresburgerSet<Int> newSet = coalesce(currentSet);
    EXPECT_TRUE(i < 67);
    EXPECT_TRUE(i < 68);
    if (!PresburgerSet<Int>::equal(newSet, currentSet)) {
      newSet.dump();
      currentSet.dump();
      EXPECT_TRUE(false);
    }
  }
  newfile.close();
}*/

TEST(CoalesceTest, contained) {
  PresburgerSet<Int> contained =
      setFromString("(x0) : (x0 >= 0 and x0 <= 4) or (x0 >= 1 and x0 <= 3)");
  expectCoalesce(1, contained);
}

TEST(CoalesceTest, cut) {
  PresburgerSet<Int> cut =
      setFromString("(x0) : (x0 >= 0 and x0 <= 3) or (x0 >= 2 and x0 <= 4)");
  expectCoalesce(1, cut);
}

TEST(CoalesceTest, adjIneq) {
  PresburgerSet<Int> adjIneq =
      setFromString("(x0) : (x0 >= 0 and x0 <= 1) or (x0 >= 2 and x0 <= 3)");
  expectCoalesce(1, adjIneq);
}

TEST(CoalesceTest, separate) {
  PresburgerSet<Int> separate =
      setFromString("(x0) : (x0 >= 0 and x0 <= 1) or (x0 >= 3 and x0 <= 4)");
  expectCoalesce(2, separate);
}

TEST(CoalesceTest, adjEq) {
  PresburgerSet<Int> adjEq =
      setFromString("(x0) : (x0 = 1) or (x0 >= 2 and x0 <= 3)");
  expectCoalesce(1, adjEq);
}

TEST(CoalesceTest, adjEqs) {
  PresburgerSet<Int> adjEqs = setFromString("(x0) : (x0 = 1) or (x0 = 2)");
  expectCoalesce(1, adjEqs);
}

TEST(CoalesceTest, eqSeparate) {
  PresburgerSet<Int> eqSeparate =
      setFromString(" (x0) : (x0 = 0) or (x0 >= 2 and x0 <= 3)");
  expectCoalesce(2, eqSeparate);
}

TEST(CoalesceTest, eqAsIneqContained) {
  PresburgerSet<Int> eqAsIneqContained =
      setFromString(" (x0) : (x0 <= 3 and x0 >= 3) or (x0 >=2 and x0 <= 3)");
  expectCoalesce(1, eqAsIneqContained);
}

TEST(CoalesceTest, eqContained) {
  PresburgerSet<Int> eqContained =
      setFromString(" (x0) : (x0 = 3) or (x0 >= 2 and x0 <= 3)");
  expectCoalesce(1, eqContained);
}

TEST(CoalesceTest, multiDimContained) {
  PresburgerSet<Int> multiDimContained =
      setFromString(" (x0, x1) : (x0 >= 0 and x0 <= 4 and x1 >= 0 and x1 <= 4 )"
                    "or (x0 >= 2 and x0 <= 3 and x1 >= 2 and x1 <= 3)");
  expectCoalesce(1, multiDimContained);
}

TEST(CoalesceTest, multiDimAdjIneq) {
  PresburgerSet<Int> multiDimAdjIneq =
      setFromString(" (x0, x1) : (x0 >= 0 and x0 <= 3 and x1 >= 0 and x1 <= 1) "
                    "or (x0 >= 0 and "
                    "x0 <= 3 and x1 <= 2 and x1 <= 3)");
  expectCoalesce(1, multiDimAdjIneq);
}

TEST(CoalesceTest, multiDimSeparate) {
  PresburgerSet<Int> multiDimSeparate =
      setFromString("(x0, x1) : (x0 >= 0 and x0 <= 1 and x1 >= 0 and x1 <= 1) "
                    "or (x0 >= 2 and "
                    "x0 <= 3 and x1 >= 2 and x1 <= 3)");
  expectCoalesce(2, multiDimSeparate);
}

TEST(CoalesceTest, multiDimCut) {
  PresburgerSet<Int> multiDimCut = setFromString(
      "(x,y) : (4x -5y <= 0 and y <= 4 and 3x + 4y >= 0 and x - y + 7 >= 0) or "
      "("
      "y >= 0 and x <= 0 and y <= 4 and x - y + 9 >= 0 and 2x + 5y + 4 >= 0)");
  expectCoalesce(1, multiDimCut);
}

TEST(CoalesceTest, multiDimNonCut) {
  PresburgerSet<Int> multiDimNonCut =
      setFromString("(x,y) :  (y >= 0 and x <= 0 and y <= 4 and x - y + 9 "
                    ">= 0 and 2x + 5y + 4 >= 0) or (4x -5y <= 0 and 3x + 4y "
                    ">= 0 and x - y + 7 >= 0 and x + y - 10 <= 0)");
  expectCoalesce(2, multiDimNonCut);
}

TEST(CoalesceTest, multiDimAdjEqs) {
  PresburgerSet<Int> multiDimAdjEqs =
      setFromString("(x,y) :  (y = x and y >= 0 and y <= 4) or (x - 1 = y "
                    "and y >= 0 and y <= 3)");
  expectCoalesce(1, multiDimAdjEqs);
}

TEST(CoalesceTest, multiDimAdjEqs2) {
  PresburgerSet<Int> multiDimAdjEqs2 =
      setFromString("(x,y) :  (y = x and y >= 0 and y <= 4) or (x - 1 = y "
                    "and y >= 1 and y <= 3)");
  expectCoalesce(1, multiDimAdjEqs2);
}

TEST(CoalesceTest, multiDimAdjEqs3) {
  PresburgerSet<Int> multiDimAdjEqs3 =
      setFromString("(x,y) : ( 2x = 3y and y >= 0 and x <= 6) or (2x = 3y+1 "
                    "and y >= 2 and y <= 5)");
  expectCoalesce(1, multiDimAdjEqs3);
}

TEST(CoalesceTest, multiDimAdjEqToPoly) {
  PresburgerSet<Int> multiDimAdjEqToPoly =
      setFromString("(x,y) :  (x <= 0 and y >= 0 and y <= 4 and x >= -8 and x "
                    "<= -y + 3) or (x "
                    "= 1 and y >= 0 and y <= 2)");
  expectCoalesce(1, multiDimAdjEqToPoly);
}

TEST(CoalesceTest, multiDimAdjEqToPolyComplex) {
  PresburgerSet<Int> multiDimAdjEqToPolyComplex =
      setFromString("(x,y) :  (x <= 0 and y >= 0 and y <= 4 and x >= -8 and x "
                    "<= -y + 3) or (x "
                    "= 1 and y >= 1 and y <= 2)");
  expectCoalesce(1, multiDimAdjEqToPolyComplex);
}

TEST(CoalesceTest, multiSetsAllContained) {
  PresburgerSet<Int> multiSetsAllContained =
      setFromString("(x0) : (x0 >= 0 and x0 <= 5) or (x0 >= 2 and x0 "
                    "<= 3) or (x0 >= 4 and x0 <= 5)");
  expectCoalesce(1, multiSetsAllContained);
}

TEST(CoalesceTest, multiSetsOneContained) {
  PresburgerSet<Int> multiSetsOneContained =
      setFromString("(x0) : (x0 >= 0 and x0 <= 3) or (x0 >= 2 and x0 "
                    "<= 3) or (x0 >= 2 and x0 <= 5)");
  expectCoalesce(1, multiSetsOneContained);
}

TEST(CoalesceTest, multiSetsNotContained) {
  PresburgerSet<Int> multiSetsNotContained =
      setFromString("(x0) : (x0 >= 0 and x0 <= 1) or (x0 >= 2 and x0 "
                    "<= 3) or (x0 >= 4 and x0 <= 5)");
  expectCoalesce(1, multiSetsNotContained);
}

TEST(CoalesceTest, protrusion) {
  PresburgerSet<Int> protrusion = setFromString(
      "(x,y) : (x >= 0 and y >= 0 and y <= 3 and x <= 9) or (y >= 1 "
      "and y <= 4 and y <= x - 1 and x + y - 11 <= 0)");
  expectCoalesce(1, protrusion);
}

TEST(CoalesceTest, nearlyProtrusion) {
  PresburgerSet<Int> nearlyProtrusion = setFromString(
      "(x,y) : (x >= 0 and y >= 0 and y <= 3 and x <= 9) or (y >= 1 "
      "and y <= 5 and y <= x - 1 and x + y - 11 <= 0)");
  expectCoalesce(2, nearlyProtrusion);
}

TEST(CoalesceTest, params) {
  PresburgerSet<Int> params = setFromString("(x)[n] : (x + n = 3) or (x + n = 2)");
  expectCoalesce(1, params);
}

TEST(CoalesceTest, existentials) {
  PresburgerSet<Int> existentials = setFromString(
      "(x)[y] : (exists e0, e1, e2 : e0 + e1 + 2 <= 7) or (4x <= 6)");
  expectCoalesce(2, existentials);
}

TEST(CoalesceTest, existentials2) {
  PresburgerSet<Int> existentials2 = setFromString(
      "(x) : ( exists j : x = 4j and 0 <= x and x <= 100) or (exists j : 4j + "
      "1 <= x and x <= 4j + 2 and 0 <= x and x <= 100)");
  expectCoalesce(1, existentials2);
}

TEST(CoalesceTest, existentials3) {
  PresburgerSet<Int> existentials3 =
      setFromString("(x) : ( exists j : x = 4j and 0 <= x and x <= 100) or (x "
                    "= 2) or (x = 3)");
  expectCoalesce(2, existentials3);
}

/*TEST(CoalesceTest, twoAdj) {
  PresburgerSet<Int> twoAdj = setFromString(
      "(x,y) : (x = 1 and y >= 0 and y <= 2) or (x = 2 and y >= 3 and y <=
5)");
  // The result should be something like: "(x,y) : ( x >= 1 and x <= 2 and 3x
-y
  // -3 <= 0 and 3x -y-1 >= 0)");
  expectCoalesce(1, twoAdj);
}*/
