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
  fac.addInequality({1, 0}); // x >= 0
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
  fac.addInequality({1, 0, 0, 1});  // x >= -1
  fac.addInequality({-1, 0, 0, 1});  // x <= 1
  fac.addInequality({0, 1, 0, 1});  // y >= 1
  fac.addInequality({0, -1, 0, 1});  // y <= 1
  fac.addEquality({0, 0, 1, -1}); // z = 1
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
  fac.addEquality({1, 1, 1, 0});   // x + y + z == 0
  fac.addEquality({1, 0, 0, 0});   // x == 0
  fac.addEquality({0, 1, 0, 0});   // y == 0
  fac.addEquality({0, 0, 1, 0});   // z == 0
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

} // namespace mlir
