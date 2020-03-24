#include "mlir/Analysis/Simplex.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {

TEST(SimplexTest, Empty) {
  Simplex s(2);
  // (u - v) >= 0
  s.addIneq(0, {1, -1});
  // (u - v) <= -1
  EXPECT_FALSE(s.isEmpty());
  s.addIneq(-1, {-1, 1});
  EXPECT_TRUE(s.isEmpty());
}

TEST(SimplexTest, isMarkedRedundant_no_var_ge_zero) {
  Simplex tab(0);
  tab.addIneq(0, {});

  tab.detectRedundant();
  EXPECT_TRUE(tab.isMarkedRedundant(0));
}

TEST(SimplexTest, isMarkedRedundant_no_var_eq) {
  Simplex tab(0);
  tab.addEq(0, {});
  tab.detectRedundant();
  EXPECT_TRUE(tab.isMarkedRedundant(0));
}

TEST(SimplexTest, isMarkedRedundant_pos_var_eq) {
  Simplex tab(1);
  tab.addEq(0, {1});

  tab.detectRedundant();
  EXPECT_FALSE(tab.isMarkedRedundant(0));
}

TEST(SimplexTest, isMarkedRedundant_zero_var_eq) {
  Simplex tab(1);
  tab.addEq(0, {0});
  tab.detectRedundant();
  EXPECT_TRUE(tab.isMarkedRedundant(0));
}

TEST(SimplexTest, isMarkedRedundant_neg_var_eq) {
  Simplex tab(1);
  tab.addEq(0, {-1});
  tab.detectRedundant();
  EXPECT_FALSE(tab.isMarkedRedundant(0));
}

TEST(SimplexTest, isMarkedRedundant_pos_var_ge) {
  Simplex tab(1);
  tab.addIneq(0, {1});
  tab.detectRedundant();
  EXPECT_FALSE(tab.isMarkedRedundant(0));
}

TEST(SimplexTest, isMarkedRedundant_zero_var_ge) {
  Simplex tab(1);
  tab.addIneq(0, {0});
  tab.detectRedundant();
  EXPECT_TRUE(tab.isMarkedRedundant(0));
}

TEST(SimplexTest, isMarkedRedundant_neg_var_ge) {
  Simplex tab(1);
  tab.addIneq(0, {-1});
  tab.detectRedundant();
  EXPECT_FALSE(tab.isMarkedRedundant(0));
}

TEST(SimplexTest, isMarkedRedundant) {
  Simplex tab(3);

  tab.addEq(0, {-1, 0, 1});     // u = w
  tab.addIneq(15, {-1, 16, 0}); // 15 - (u - 16v) >= 0
  tab.addIneq(0, {1, -16, 0});  //      (u - 16v) >= 0

  tab.detectRedundant();

  for (size_t i = 0; i < tab.numberConstraints(); ++i)
    EXPECT_FALSE(tab.isMarkedRedundant(i)) << "i = " << i << "\n";
}

TEST(SimplexTest, isMarkedRedundant_a) {
  Simplex tab(17);

  tab.addEq(0, {0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0});
  tab.addEq(-10, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0});
  tab.addEq(-13, {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0});
  tab.addEq(-10, {0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  tab.addEq(-13, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  tab.addIneq(-1, {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  tab.addIneq(500, {0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  tab.addIneq(0, {0, 0, 0, -16, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  tab.addIneq(-1, {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  tab.addIneq(998, {0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  tab.addIneq(15, {0, 0, 0, 16, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  tab.addIneq(0, {0, 0, 0, 0, -16, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  tab.addIneq(-1, {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  tab.addIneq(998, {0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  tab.addIneq(15, {0, 0, 0, 0, 16, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  tab.addIneq(0, {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  tab.addIneq(1, {0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  tab.addIneq(-1, {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0});
  tab.addIneq(500, {0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0});
  tab.addIneq(15, {0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0});
  tab.addIneq(0, {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -16, 0, 0, 0, 0, 0});
  tab.addIneq(0, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -16, 0, 1, 0, 0});
  tab.addIneq(-1, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0});
  tab.addIneq(998, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0});
  tab.addIneq(15, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, -1, 0, 0});
  tab.addIneq(0, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0});
  tab.addIneq(1, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0});
  tab.addIneq(8, {0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 8});
  tab.addIneq(8, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 8});
  tab.addIneq(-1, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -8});
  tab.addIneq(-1, {0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -8});

  tab.detectRedundant();
  for (size_t i = 0; i < tab.numberConstraints(); ++i)
    EXPECT_FALSE(tab.isMarkedRedundant(i)) << "i = " << i << '\n';
}

TEST(SimplexTest, isMarkedRedundant1) {
  Simplex s(3);
  s.addIneq(1, {0, -1, 0});
  s.addIneq(7, {-1, 0, 8});
  s.addIneq(0, {1, 0, -8});
  s.addIneq(0, {0, 1, 0});
  s.addIneq(7, {-1, 0, 8});
  s.addIneq(0, {1, 0, -8});
  s.addIneq(0, {0, 1, 0});
  s.addIneq(1, {0, -1, 0});
  s.detectRedundant();
  EXPECT_EQ(s.isMarkedRedundant(1), false);
  EXPECT_EQ(s.isMarkedRedundant(2), false);
  EXPECT_EQ(s.isMarkedRedundant(3), false);
  EXPECT_EQ(s.isMarkedRedundant(4), true);
  EXPECT_EQ(s.isMarkedRedundant(5), true);
  EXPECT_EQ(s.isMarkedRedundant(6), true);
  EXPECT_EQ(s.isMarkedRedundant(7), true);
}

TEST(SimplexTest, isMarkedRedundant2) {
  Simplex tab(3);
  tab.addIneq(1, {0, -1, 0});
  tab.addIneq(-1, {1, 0, 0});
  tab.addIneq(2, {-1, 0, 0});
  tab.addIneq(7, {-1, 0, 2});
  tab.addIneq(0, {1, 0, -2});
  tab.addIneq(0, {0, 1, 0});
  tab.addIneq(1, {0, 1, -2});
  tab.addIneq(1, {-1, 1, 0});

  tab.detectRedundant();

  EXPECT_EQ(tab.isMarkedRedundant(0), false);
  EXPECT_EQ(tab.isMarkedRedundant(1), false);
  EXPECT_EQ(tab.isMarkedRedundant(2), true);
  EXPECT_EQ(tab.isMarkedRedundant(3), false);
  EXPECT_EQ(tab.isMarkedRedundant(4), false);
  EXPECT_EQ(tab.isMarkedRedundant(5), true);
  EXPECT_EQ(tab.isMarkedRedundant(6), true);
  EXPECT_EQ(tab.isMarkedRedundant(7), false);
}

TEST(SimplexTest, addIneqAlreadyRedundant) {
  Simplex tab(1);
  tab.addIneq(-1, {1}); // x >= 1
  tab.addIneq(0, {1});  // x >= 0
  tab.detectRedundant();
  EXPECT_TRUE(tab.isMarkedRedundant(1));
}

TEST(SimplexTest, addEqSeparate) {
  Simplex tab(1);
  tab.addIneq(-1, {1});
  ASSERT_FALSE(tab.isEmpty());
  tab.addEq(0, {1});
  EXPECT_TRUE(tab.isEmpty());
}
} // namespace mlir
