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

TEST(SimplexTest, isMarkedRedundant_no_var_ge_zero) {
  Simplex tab(0);
  tab.addInequality({0}); // 0 >= 0

  tab.detectRedundant();
  ASSERT_FALSE(tab.isEmpty());
  EXPECT_TRUE(tab.isMarkedRedundant(0));
}

TEST(SimplexTest, isMarkedRedundant_no_var_eq) {
  Simplex tab(0);
  tab.addEquality({0}); // 0 == 0
  tab.detectRedundant();
  ASSERT_FALSE(tab.isEmpty());
  EXPECT_TRUE(tab.isMarkedRedundant(0));
}

TEST(SimplexTest, isMarkedRedundant_pos_var_eq) {
  Simplex tab(1);
  tab.addEquality({1, 0}); // x == 0

  tab.detectRedundant();
  ASSERT_FALSE(tab.isEmpty());
  EXPECT_FALSE(tab.isMarkedRedundant(0));
}

TEST(SimplexTest, isMarkedRedundant_zero_var_eq) {
  Simplex tab(1);
  tab.addEquality({0, 0}); // 0x == 0
  tab.detectRedundant();
  ASSERT_FALSE(tab.isEmpty());
  EXPECT_TRUE(tab.isMarkedRedundant(0));
}

TEST(SimplexTest, isMarkedRedundant_neg_var_eq) {
  Simplex tab(1);
  tab.addEquality({-1, 0}); // -x == 0
  tab.detectRedundant();
  ASSERT_FALSE(tab.isEmpty());
  EXPECT_FALSE(tab.isMarkedRedundant(0));
}

TEST(SimplexTest, isMarkedRedundant_pos_var_ge) {
  Simplex tab(1);
  tab.addInequality({1, 0}); // x >= 0
  tab.detectRedundant();
  ASSERT_FALSE(tab.isEmpty());
  EXPECT_FALSE(tab.isMarkedRedundant(0));
}

TEST(SimplexTest, isMarkedRedundant_zero_var_ge) {
  Simplex tab(1);
  tab.addInequality({0, 0}); // 0x >= 0
  tab.detectRedundant();
  ASSERT_FALSE(tab.isEmpty());
  EXPECT_TRUE(tab.isMarkedRedundant(0));
}

TEST(SimplexTest, isMarkedRedundant_neg_var_ge) {
  Simplex tab(1);
  tab.addInequality({-1, 0}); // x <= 0
  tab.detectRedundant();
  ASSERT_FALSE(tab.isEmpty());
  EXPECT_FALSE(tab.isMarkedRedundant(0));
}

TEST(SimplexTest, isMarkedRedundant_no_redundant) {
  Simplex tab(3);

  tab.addEquality({-1, 0, 1, 0});     // u = w
  tab.addInequality({-1, 16, 0, 15}); // 15 - (u - 16v) >= 0
  tab.addInequality({1, -16, 0, 0});  //      (u - 16v) >= 0

  tab.detectRedundant();
  ASSERT_FALSE(tab.isEmpty());

  for (unsigned i = 0; i < tab.numberConstraints(); ++i)
    EXPECT_FALSE(tab.isMarkedRedundant(i)) << "i = " << i << "\n";
}

TEST(SimplexTest, isMarkedRedundant_regression_test) {
  Simplex tab(17);

  tab.addEquality({0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0});
  tab.addEquality({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -10});
  tab.addEquality({0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -13});
  tab.addEquality({0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -10});
  tab.addEquality({1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -13});
  tab.addInequality({0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1});
  tab.addInequality({0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 500});
  tab.addInequality({0, 0, 0, -16, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  tab.addInequality({0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1});
  tab.addInequality({0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 998});
  tab.addInequality({0, 0, 0, 16, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15});
  tab.addInequality({0, 0, 0, 0, -16, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  tab.addInequality({0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1});
  tab.addInequality({0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 998});
  tab.addInequality({0, 0, 0, 0, 16, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15});
  tab.addInequality({0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  tab.addInequality({0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1});
  tab.addInequality({0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, -1});
  tab.addInequality({0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 500});
  tab.addInequality({0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 15});
  tab.addInequality({0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -16, 0, 0, 0, 0, 0, 0});
  tab.addInequality({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -16, 0, 1, 0, 0, 0});
  tab.addInequality({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1});
  tab.addInequality({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 998});
  tab.addInequality({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, -1, 0, 0, 15});
  tab.addInequality({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0});
  tab.addInequality({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1});
  tab.addInequality({0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8});
  tab.addInequality({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 8, 8});
  tab.addInequality({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -8, -1});
  tab.addInequality({0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -8, -1});

  tab.detectRedundant();
  ASSERT_FALSE(tab.isEmpty());
  for (unsigned i = 0; i < tab.numberConstraints(); ++i)
    EXPECT_FALSE(tab.isMarkedRedundant(i)) << "i = " << i << '\n';
}

TEST(SimplexTest, isMarkedRedundant_repeated_constraints) {
  Simplex tab(3);

  // [4] to [7] are repeats of [0] to [3].
  tab.addInequality({0, -1, 0, 1}); // [0]: y <= 1
  tab.addInequality({-1, 0, 8, 7}); // [1]: 8z >= x - 7
  tab.addInequality({1, 0, -8, 0}); // [2]: 8z <= x
  tab.addInequality({0, 1, 0, 0});  // [3]: y >= 0
  tab.addInequality({-1, 0, 8, 7}); // [4]: 8z >= 7 - x
  tab.addInequality({1, 0, -8, 0}); // [5]: 8z <= x
  tab.addInequality({0, 1, 0, 0});  // [6]: y >= 0
  tab.addInequality({0, -1, 0, 1}); // [7]: y <= 1

  tab.detectRedundant();
  ASSERT_FALSE(tab.isEmpty());

  EXPECT_EQ(tab.isMarkedRedundant(0), false);
  EXPECT_EQ(tab.isMarkedRedundant(1), false);
  EXPECT_EQ(tab.isMarkedRedundant(2), false);
  EXPECT_EQ(tab.isMarkedRedundant(3), false);
  EXPECT_EQ(tab.isMarkedRedundant(4), true);
  EXPECT_EQ(tab.isMarkedRedundant(5), true);
  EXPECT_EQ(tab.isMarkedRedundant(6), true);
  EXPECT_EQ(tab.isMarkedRedundant(7), true);
}

TEST(SimplexTest, isMarkedRedundant) {
  Simplex tab(3);
  tab.addInequality({0, -1, 0, 1}); // [0]: y <= 1
  tab.addInequality({1, 0, 0, -1}); // [1]: x >= 1
  tab.addInequality({-1, 0, 0, 2}); // [2]: x <= 2
  tab.addInequality({-1, 0, 2, 7}); // [3]: 2z >= x - 7
  tab.addInequality({1, 0, -2, 0}); // [4]: 2z <= x
  tab.addInequality({0, 1, 0, 0});  // [5]: y >= 0
  tab.addInequality({0, 1, -2, 1}); // [6]: y >= 2z - 1
  tab.addInequality({-1, 1, 0, 1}); // [7]: y >= x - 1

  tab.detectRedundant();
  ASSERT_FALSE(tab.isEmpty());

  // [0], [1], [3], [4], [7] together imply [2], [5], [6] must hold.
  //
  // From [7], [0]: x <= y + 1 <= 2, so we have [2].
  // From [7], [1]: y >= x - 1 >= 0, so we have [5].
  // From [4], [7]: 2z - 1 <= x - 1 <= y, so we have [6].
  EXPECT_FALSE(tab.isMarkedRedundant(0));
  EXPECT_FALSE(tab.isMarkedRedundant(1));
  EXPECT_TRUE(tab.isMarkedRedundant(2));
  EXPECT_FALSE(tab.isMarkedRedundant(3));
  EXPECT_FALSE(tab.isMarkedRedundant(4));
  EXPECT_TRUE(tab.isMarkedRedundant(5));
  EXPECT_TRUE(tab.isMarkedRedundant(6));
  EXPECT_FALSE(tab.isMarkedRedundant(7));
}

TEST(SimplexTest, addInequality_already_redundant) {
  Simplex tab(1);
  tab.addInequality({1, -1}); // x >= 1
  tab.addInequality({1, 0});  // x >= 0
  tab.detectRedundant();
  ASSERT_FALSE(tab.isEmpty());
  EXPECT_FALSE(tab.isMarkedRedundant(0));
  EXPECT_TRUE(tab.isMarkedRedundant(1));
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
