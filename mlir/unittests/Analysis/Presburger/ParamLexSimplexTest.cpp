//===- SimplexTest.cpp - Tests for ParamLexSimplex ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/ParamLexSimplex.h"

#include <ostream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {
void PrintTo(const Fraction& f, std::ostream* os) {
  *os << f.num << '/' << f.den;
}

void expectSample(ParamLexSimplex &simplex, std::initializer_list<Fraction> expected) {
  EXPECT_THAT(simplex.getSamplePoint(), testing::ElementsAreArray(expected));
}

TEST(ParamLexSimplexTest, NoParamsTest) {
  ParamLexSimplex simplex(3, 0);
  simplex.addInequality({1, 0, 0, 0});
  simplex.addInequality({0, 1, 0, 0});
  simplex.addInequality({0, 0, 1, 0});

  expectSample(simplex, {{0, 1}, {0, 1}, {0, 1}});
  simplex.addInequality({1, 1, 1, -1}); // x + y + z >= 1.
  expectSample(simplex, {{0, 1}, {0, 1}, {1, 1}});
  simplex.addEquality({1, 0, -1, 0}); // x == z.
  expectSample(simplex, {{0, 1}, {1, 1}, {0, 1}});
  simplex.addInequality({1, -2, 0, 0}); // x >= 2y.
  expectSample(simplex, {{2, 5}, {1, 5}, {2, 5}});
  simplex.addEquality({0, 1, -1, 0}); // y == z.
  EXPECT_TRUE(simplex.isEmpty());
}

} // namespace mlir
