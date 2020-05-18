//===- SimplexTest.cpp - Tests for Simplex ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Simplex.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {

TEST(SimplexTest, emptyRollback) {
  Simplex tab(2);
  // (u - v) >= 0
  tab.addInequality({1, -1, 0});
  EXPECT_FALSE(tab.isEmpty());

  auto snap = tab.getSnapshot();
  // (u - v) <= -1
  tab.addInequality({-1, 1, -1});
  EXPECT_TRUE(tab.isEmpty());
  tab.rollback(snap);
  EXPECT_FALSE(tab.isEmpty());
}

TEST(SimplexTest, addEquality_separate) {
  Simplex tab(1);
  tab.addInequality({1, -1}); // x >= 1
  ASSERT_FALSE(tab.isEmpty());
  tab.addEquality({1, 0}); // x == 0
  EXPECT_TRUE(tab.isEmpty());
}

void expectInequalityMakesTabEmpty(Simplex &tab, ArrayRef<int64_t> coeffs,
                                   bool expect) {
  ASSERT_FALSE(tab.isEmpty());
  auto snapshot = tab.getSnapshot();
  tab.addInequality(coeffs);
  EXPECT_EQ(tab.isEmpty(), expect);
  tab.rollback(snapshot);
}

TEST(SimplexTest, addInequality_rollback) {
  Simplex tab(3);

  std::vector<int64_t> coeffs[]{{1, 0, 0, 0},   // u >= 0
                                {-1, 0, 0, 0},  // u <= 0
                                {1, -1, 1, 0},  // u - v + w >= 0
                                {1, 1, -1, 0}}; // u + v - w >= 0
  // The above constraints force u = 0 and v = w.
  // The constraints below violate v = w.
  std::vector<int64_t> checkCoeffs[]{{0, 1, -1, -1},  // v - w >= 1
                                     {0, -1, 1, -1}}; // v - w <= -1
  for (int run = 0; run < 4; run++) {
    auto snapshot = tab.getSnapshot();

    expectInequalityMakesTabEmpty(tab, checkCoeffs[0], false);
    expectInequalityMakesTabEmpty(tab, checkCoeffs[1], false);

    for (int i = 0; i < 4; i++)
      tab.addInequality(coeffs[(run + i) % 4]);

    expectInequalityMakesTabEmpty(tab, checkCoeffs[0], true);
    expectInequalityMakesTabEmpty(tab, checkCoeffs[1], true);

    tab.rollback(snapshot);
    EXPECT_EQ(tab.numberConstraints(), 0u);

    expectInequalityMakesTabEmpty(tab, checkCoeffs[0], false);
    expectInequalityMakesTabEmpty(tab, checkCoeffs[1], false);
  }
}
} // namespace mlir
