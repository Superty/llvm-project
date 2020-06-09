//===- AffineStructuresTest.cpp - Tests for AffineStructures ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/AffineStructures.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {

TEST(FlatAffineConstraintsTest, RemoveRedundantConstraints1D) {
  FlatAffineConstraints fac(2, 0, 2, 1);
  fac.addInequality({1, 0});  // x >= 0
  fac.addInequality({1, -2}); // x >= 2
  EXPECT_EQ(fac.getNumInequalities(), 2u);
  fac.removeRedundantConstraints();
  EXPECT_EQ(fac.getNumInequalities(), 1u);
  EXPECT_EQ(fac.getConstantLowerBound(0).getValueOr(-100), 2);
}

TEST(FlatAffineConstraintsTest, RemoveRedundantConstraints3D) {
  FlatAffineConstraints fac(4, 0, 4, 3);
  fac.addInequality({1, 1, 0, 2});   // x + y >= 2
  fac.addInequality({-1, -1, 0, 2}); // x + y <= 2
  fac.addInequality({1, -1, 0, 2});  // x - y >= 2
  fac.addInequality({-1, 1, 0, 2});  // x - y <= 2
  fac.addInequality({1, 0, 0, 1});   // x >= -1
  fac.addInequality({-1, 0, 0, 1});  // x <= 1
  fac.addInequality({0, 1, 0, 1});   // y >= 1
  fac.addInequality({0, -1, 0, 1});  // y <= 1
  fac.addEquality({0, 0, 1, -1});    // z = 1
  EXPECT_EQ(fac.getNumInequalities(), 8u);
  EXPECT_EQ(fac.getNumEqualities(), 1u);
  fac.removeRedundantConstraints();
  EXPECT_EQ(fac.getNumInequalities(), 4u);
  EXPECT_EQ(fac.getNumEqualities(), 1u);
  EXPECT_EQ(fac.getConstantLowerBound(0).getValueOr(-100), -1);
  EXPECT_EQ(fac.getConstantUpperBound(0).getValueOr(-100), 1);
  EXPECT_EQ(fac.getConstantLowerBound(1).getValueOr(-100), -1);
  EXPECT_EQ(fac.getConstantUpperBound(1).getValueOr(-100), 1);
  EXPECT_EQ(fac.getConstantLowerBound(2).getValueOr(-100), 1);
  EXPECT_EQ(fac.getConstantUpperBound(2).getValueOr(-100), 1);
}

TEST(FlatAffineConstraintsTest, RemoveRedundantConstraintsEqualities) {
  FlatAffineConstraints fac(4, 0, 4, 3);
  fac.addEquality({1, 1, 1, 0}); // x + y + z == 0
  fac.addEquality({1, 0, 0, 0}); // x == 0
  fac.addEquality({0, 1, 0, 0}); // y == 0
  fac.addEquality({0, 0, 1, 0}); // z == 0
  EXPECT_EQ(fac.getNumEqualities(), 4u);
  fac.removeRedundantConstraints();
  EXPECT_EQ(fac.getNumEqualities(), 3u);
  EXPECT_EQ(fac.getConstantLowerBound(0).getValueOr(-100), 0);
  EXPECT_EQ(fac.getConstantUpperBound(0).getValueOr(-100), 0);
  EXPECT_EQ(fac.getConstantLowerBound(1).getValueOr(-100), 0);
  EXPECT_EQ(fac.getConstantUpperBound(1).getValueOr(-100), 0);
  EXPECT_EQ(fac.getConstantLowerBound(2).getValueOr(-100), 0);
  EXPECT_EQ(fac.getConstantUpperBound(2).getValueOr(-100), 0);
}

int64_t valueAt(ArrayRef<int64_t> expr, const std::vector<int64_t> &point) {
  int64_t value = expr.back();
  assert(expr.size() == 1 + point.size());
  for (unsigned i = 0; i < point.size(); ++i)
    value += expr[i] * point[i];
  return value;
}

void checkSample(bool hasValue, const FlatAffineConstraints &fac) {
  auto maybeSample = fac.findIntegerSample();
  fac.dump();
  if (!hasValue)
    EXPECT_FALSE(maybeSample.hasValue());
  else {
    ASSERT_TRUE(maybeSample.hasValue());
    for (unsigned i = 0; i < fac.getNumEqualities(); ++i)
      EXPECT_EQ(valueAt(fac.getEquality(i), *maybeSample), 0);
    for (unsigned i = 0; i < fac.getNumInequalities(); ++i)
      EXPECT_GE(valueAt(fac.getInequality(i), *maybeSample), 0);
  }
}

FlatAffineConstraints
makeFACFromConstraints(unsigned dims, std::vector<std::vector<int64_t>> ineqs,
                       std::vector<std::vector<int64_t>> eqs) {
  FlatAffineConstraints fac(ineqs.size(), eqs.size(), dims + 1, dims);
  for (const auto &eq : eqs)
    fac.addEquality(eq);
  for (const auto &ineq : ineqs)
    fac.addInequality(ineq);
  return fac;
}

TEST(FlatAffineConstraintsTest, FindSampleTest) {
  // bounded sets, only inequalities

  // 1 <= 5x and 5x <= 4 (no solution)
  checkSample(false, makeFACFromConstraints(1, {{5, -1}, {-5, 4}}, {}));

  // 1 <= 5x and 5x <= 9 (solution: x = 1)
  checkSample(true, makeFACFromConstraints(1, {{5, -1}, {-5, 9}}, {}));

  // bounded sets with equalities

  // x >= 8 and 40 >= y and x = y
  checkSample(
      true, makeFACFromConstraints(2, {{1, 0, -8}, {0, -1, 40}}, {{1, -1, 0}}));

  // x <= 10 and y <= 10 and 10 <= z and x + 2y = 3z
  // solution: x = y = z = 10.
  checkSample(true, makeFACFromConstraints(
                        3, {{-1, 0, 0, 10}, {0, -1, 0, 10}, {0, 0, 1, -10}},
                        {{1, 2, -3, 0}}));

  // x <= 10 and y <= 10 and 11 <= z and x + 2y = 3z
  // This implies x + 2y >= 33 and x + 2y <= 30, which has no solution.
  checkSample(false, makeFACFromConstraints(
                         3, {{-1, 0, 0, 10}, {0, -1, 0, 10}, {0, 0, 1, -11}},
                         {{1, 2, -3, 0}}));

  // 0 <= r and r <= 3 and 4q + r = 7
  // Solution: q = 1, r = 3
  checkSample(true,
              makeFACFromConstraints(2, {{0, 1, 0}, {0, -1, 3}}, {{4, 1, -7}}));

  // 4q + r = 7 and r = 0
  // Solution: q = 1, r = 3
  checkSample(false, makeFACFromConstraints(2, {}, {{4, 1, -7}, {0, 1, 0}}));

  // q + r = 0 and r >= 0
  // Solution: q = r = 0
  checkSample(true, makeFACFromConstraints(2, {{0, 1, 0}}, {{1, 1, 0}}));

  // unbounded sets

  // 4q + r = 7 and r = 0
  // Solution: q = 1, r = 3
  checkSample(true, makeFACFromConstraints(2, {}, {{4, 1, -7}}));

  // q >= 7
  checkSample(true, makeFACFromConstraints(1, {{1, -7}}, {}));

  // 4x + 2y + z - 10 >= 0 and z = 0
  checkSample(true,
              makeFACFromConstraints(3, {{4, 2, 1, -10}}, {{0, 0, 1, 0}}));

  // 4x + 2y + 1 == 0
  // As x and y can only be integral, this implies that the above constraint
  // always has an odd value -> it cannot be zero
  checkSample(false, makeFACFromConstraints(2, {}, {{4, 2, 1}}));

  // x + y + z - 1 >= 0 and x + y + z <= 0
  checkSample(false,
              makeFACFromConstraints(3, {{1, 1, 1, -1}, {-1, -1, -1, 0}}, {}));
}

TEST(FlatAffineConstraintsTest, IsIntegerEmptyTest) {
  // 1 <= 5x and 5x <= 4 (no solution)
  EXPECT_TRUE(
      makeFACFromConstraints(1, {{5, -1}, {-5, 4}}, {}).isIntegerEmpty());
  // 1 <= 5x and 5x <= 9 (solution: x = 1)
  EXPECT_FALSE(
      makeFACFromConstraints(1, {{5, -1}, {-5, 9}}, {}).isIntegerEmpty());
}

} // namespace mlir
