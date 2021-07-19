//===- SimplexTest.cpp - Tests for ParamLexSimplex ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Set.h"
#include "mlir/Dialect/Presburger/Parser.h"

#include <ostream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::presburger;

static PresburgerSet setFromString(StringRef string) {
  ErrorCallback callback = [](SMLoc loc, const Twine &message) {
    // This is a hack to make the Parser compile
    llvm::errs() << "Parsing error " << message << " at " << loc.getPointer() << '\n';
    llvm_unreachable("PARSING ERROR!!");
    MLIRContext context;
    return mlir::emitError(UnknownLoc::get(&context), message);
  };
  Parser parser(string, callback);
  PresburgerParser setParser(parser);
  PresburgerSet res;
  setParser.parsePresburgerSet(res);
  return res;
}

void expectEqual(StringRef sDesc, StringRef tDesc) {
  auto s = setFromString(sDesc);
  auto t = setFromString(tDesc);
  EXPECT_TRUE(PresburgerSet::equal(s, t));
}

TEST(PresburgerSetTest, Equality) {
  expectEqual("(x) : (exists y, z : x = y + 3z and x >= y and z >= 0 and y >= 0)",
              "(x) : (x >= 0)");
  expectEqual("(x) : (exists y, z : x = y + 3z and x >= y and z >= 0 and y >= 0)",
              "(x) : (exists y, z : x = y + 3z and x >= y and z >= 0 and y >= 0)");
  expectEqual("(x) : (exists q = [(x)/2] : x = 2q)",
              "(x) : (exists q = [(x)/2] : x = 2q)");
  expectEqual("(x) : (exists q = [(x)/2] : x = 2q) or (exists q = [(x)/3] : x = 3q)",
              "(x) : (exists q = [(x)/2] : x = 2q) or (exists q = [(x)/3] : x = 3q)");
  expectEqual("(x) : (exists q : x = q and q <= -1)",
              "(x) : (x <= -1)");
}

void expectEqualAfterNormalization(PresburgerSet &set) {
  auto newSet = set;
  newSet.simplify();

  EXPECT_TRUE(PresburgerSet::equal(set, newSet));

  // Normalization check
  for (const PresburgerBasicSet &pbs : newSet.getBasicSets()) {
    for (const DivisionConstraint &div : pbs.getDivisions()) {
      int64_t denom = div.getDenominator();
      for (const int64_t &coeff : div.getCoeffs())
        EXPECT_TRUE((std::abs(2 * coeff) < denom) || (2 * coeff == denom));
    }
  }
}

TEST(PresburgerSetTest, simplify1) {
  PresburgerSet simplify1 = setFromString(
      "(x) : (exists q = [(5x - 9) / 10], p = [(x - 5)/2] : x - 1 <= 3q "
      "and 3q <= x and p >= x) or (exists p = [(4x - 9)/2], q = "
      "[(4x - 3)/3] : x - 2 = 3q and 4p >= x)");
  expectEqualAfterNormalization(simplify1);
}

TEST(PresburgerSetTest, simplify2) {
  PresburgerSet set = setFromString(
      "(x) : (exists q = [(5p - 15) / 10], p = [(x - 5)/2] : x - 1 <= 3q "
      "and 3q <= x and p >= x) or (exists p = [(4x - 9)/2], q = "
      "[(4x - 3)/3] : x - 2 = 3q and 4p >= x)");
  expectEqualAfterNormalization(set);
}

TEST(PresburgerSetTest, simplify3) {
  PresburgerSet set = setFromString(
      "(d0, d1, d2, d3, d4, d5)[s0, s1] : (exists e0, e1, e2, e3, e4, e5 : d0 "
      "- d3 + 1 = 0 and d1 - d4 = 0 and d2 - d5 = 0 and d5 - e2 = 0 and s0 - "
      "e3 = 0 and s1 - e4 = 0 and e0 - e1 = 0 and e1 - e3 = 0 and d3 - e4 = 0 "
      "and d4 - s0 + e1 = 0 and d4 - 1 >= 0 and d3 - d4 + 1 >= 0 and -d3 + s0 "
      ">= 0 and d5 - 1 >= 0 and -d5 + s1 >= 0 and e3 >= 0 and e2 - e3 >= 0 and "
      "-e2 + e5 >= 0 and e4 >= 0 and -e4 + 1 >= 0 and e3 >= 0 and e2 - e3 >= 0 "
      "and -e2 + e5 >= 0 and e4 >= 0 and -e4 + 1 >= 0 and d3 >= 0)");
  expectEqualAfterNormalization(set);
}

TEST(PresburgerSetTest, simplify4) {
  PresburgerSet set = setFromString(
      "(d0, d1, d2, d3, d4, d5)[s0, s1] : (exists e0, e1, e2, e3, e4, e5 : d0 "
      "- d1 = 0 and d1 - d4 = 0 and d2 - d5 = 0 and d5 - e2 = 0 and s0 - e3 = "
      "0 and s1 - e4 = 0 and e0 - e1 = 0 and e1 - e3 = 0 and d3 - e4 = 0 and "
      "d4 - s0 + e1 = 0 and d4 - 1 >= 0 and d3 - d4 + 1 >= 0 and -d3 + s0 >= 0 "
      "and d5 - 1 >= 0 and -d5 + s1 >= 0 and e3 >= 0 and e2 - e3 >= 0 and -e2 "
      "+ e5 >= 0 and e4 >= 0 and -e4 + 1 >= 0 and e3 >= 0 and e2 - e3 >= 0 and "
      "-e2 + e5 >= 0 and e4 >= 0 and -e4 + 1 >= 0 and -d3 >= 0)");
  expectEqualAfterNormalization(set);
}

TEST(PresburgerSetTest, existentialTest) {
  PresburgerSet set = setFromString("(x) : (exists a, b = [(x + 1) / 2], c : a + x >= 5)");
  set.simplify();

  auto bs = set.getBasicSets()[0];
  EXPECT_TRUE(bs.getNumExists() == 1);
  EXPECT_TRUE(bs.getNumDivs() == 0);
}

TEST(PresburgerSetTest, existentialTest2) {
  PresburgerSet divisionOrder1 = setFromString(
      "(d0) : (exists q0 = [(d0 + 1)/3] : -d0 + 3q0 + 1 >= 0 and d0 - 3q0 >= "
      "0) or (exists q0 = [(d0 - 2)/3] : d0 - 3q0 - 2 = 0)");
  expectEqualAfterNormalization(divisionOrder1);
}

TEST(PresburgerSetTest, recoverDivisionTest1) {
  PresburgerSet set = setFromString(
      "(x) : (exists a : x - 1 <= 3a and 3a <= x) or (exists a : x - 2 = 3a)");
  expectEqualAfterNormalization(set);
}

TEST(PresburgerSetTest, recoverDivisionTest2) {
  PresburgerSet set =
      setFromString("(x) : (exists a, b, c : x - 1 <= 3a and 3a <= x and 3b = "
                    "4x and 5c >= 4x and 5c <= 4x)");
  expectEqualAfterNormalization(set);
}

TEST(PresburgerSet, falseEqualityTest) {
  PresburgerSet set =
      setFromString("(x)[] : (-1 >= 0)");
  PresburgerBasicSet bs = set.getBasicSets()[0];
  EXPECT_TRUE(bs.isIntegerEmpty());
}
