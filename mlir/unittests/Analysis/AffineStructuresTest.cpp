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

int64_t valueAt(ArrayRef<int64_t> expr, const std::vector<int64_t> &point) {
  int64_t value = expr.back();
  assert(expr.size() == 1 + point.size());
  for (unsigned i = 0; i < point.size(); ++i)
    value += expr[i] * point[i];
  return value;
}

void checkSample(bool hasValue, const FlatAffineConstraints &fac) {
  auto maybeSample = fac.findSample();
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
}

} // namespace mlir
