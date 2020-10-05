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

// TEST(ParamLexSimplexTest, NoParamsTest) {
//   ParamLexSimplex simplex(3, 0);
//   simplex.addInequality({1, 0, 0, 0});
//   simplex.addInequality({0, 1, 0, 0});
//   simplex.addInequality({0, 0, 1, 0});

//   expectSample(simplex, {{0, 1}, {0, 1}, {0, 1}});
//   simplex.addInequality({1, 1, 1, -1}); // x + y + z >= 1.
//   expectSample(simplex, {{0, 1}, {0, 1}, {1, 1}});
//   simplex.addEquality({1, 0, -1, 0}); // x == z.
//   expectSample(simplex, {{0, 1}, {1, 1}, {0, 1}});
//   simplex.addInequality({1, -2, 0, 0}); // x >= 2y.
//   expectSample(simplex, {{2, 5}, {1, 5}, {2, 5}});
//   simplex.addEquality({0, 1, -1, 0}); // y == z.
//   EXPECT_TRUE(simplex.isEmpty());
// }

TEST(ParamLexSimplexTest, ParamLexMinTest) {
  // { // a] -> {[x] : x >= a}
  //   ParamLexSimplex simplex(2, 1);
  //   simplex.addInequality({1, 0, 0});
  //   simplex.addInequality({0, 1, 0});
  //   simplex.addInequality({-1, 1, 0});
  //   simplex.findParamLexmin().dump();
  // }

  // { // [a, b] -> {[x] : x >= a and x >= b}
  //   ParamLexSimplex simplex(3, 2);
  //   simplex.addInequality({1, 0, 0, 0});
  //   simplex.addInequality({0, 1, 0, 0});
  //   simplex.addInequality({0, 0, 1, 0});
  //   simplex.addInequality({-1, 0, 1, 0});
  //   simplex.addInequality({0, -1, 1, 0});
  //   simplex.findParamLexmin().dump();
  // }

  // { // [a, b, c] -> {[x] : x >= a and x >= b and x >= c}
  //   ParamLexSimplex simplex(4, 3);
  //   simplex.addInequality({1, 0, 0, 0, 0});
  //   simplex.addInequality({0, 1, 0, 0, 0});
  //   simplex.addInequality({0, 0, 1, 0, 0});
  //   simplex.addInequality({0, 0, 0, 1, 0});
  //   simplex.addInequality({-1, 0, 0, 1, 0});
  //   simplex.addInequality({0, -1, 0, 1, 0});
  //   simplex.addInequality({0, 0, -1, 1, 0});
  //   simplex.findParamLexmin().dump();
  // }

  // { // [a] -> {[x, y] : x >= a and x + y >= 0}
  //   ParamLexSimplex simplex(3, 1);
  //   simplex.addInequality({1, 0, 0, 0});
  //   simplex.addInequality({0, 1, 0, 0});
  //   simplex.addInequality({0, 0, 1, 0});
  //   simplex.addInequality({-1, 0, 1, 0});
  //   simplex.addInequality({0, 1, 1, 0});
  //   simplex.findParamLexmin().dump();
  // }

  // { // [a, b, c] -> {[x, y] : x >= a and y >= b and x + y <= c}
  //   ParamLexSimplex simplex(5, 3);
  //   simplex.addInequality({1, 0, 0, 0, 0, 0});
  //   simplex.addInequality({0, 1, 0, 0, 0, 0});
  //   simplex.addInequality({0, 0, 1, 0, 0, 0});
  //   simplex.addInequality({0, 0, 0, 1, 0, 0});
  //   simplex.addInequality({0, 0, 0, 0, 1, 0});
  //   simplex.addInequality({-1, 0, 0, 1, 0, 0});
  //   simplex.addInequality({0, -1, 0, 0, 1, 0});
  //   simplex.addInequality({0, 0, 1, -1, -1, 0});
  //   simplex.findParamLexmin().dump();
  // }

  // { // [a, b, c] -> {[x, y, z] : z <= c and y <= b and x + y + z = a}
  //   ParamLexSimplex simplex(6, 3);
  //   simplex.addInequality({1, 0, 0, 0, 0, 0, 0});
  //   simplex.addInequality({0, 1, 0, 0, 0, 0, 0});
  //   simplex.addInequality({0, 0, 1, 0, 0, 0, 0});
  //   simplex.addInequality({0, 0, 0, 1, 0, 0, 0});
  //   simplex.addInequality({0, 0, 0, 0, 1, 0, 0});
  //   simplex.addInequality({0, 0, 0, 0, 0, 1, 0});
  //   simplex.addInequality({0, 0, 1, 0, 0, -1, 0});
  //   simplex.addInequality({0, 1, 0, 0, -1, 0, 0});
  //   simplex.addInequality({1, 0, 0, -1, -1, -1, 0});
  //   simplex.addInequality({-1, 0, 0, 1, 1, 1, 0});
  //   simplex.findParamLexmin().dump();
  // }

  // { // [x, y] -> {[z] : x >= 0 and y >= 0 and z >= 0 and x + y + z >= 1}
  //   ParamLexSimplex simplex(3, 2);
  //   simplex.addInequality({1, 0, 0, 0});
  //   simplex.addInequality({0, 1, 0, 0});
  //   simplex.addInequality({0, 0, 1, 0});
  //   simplex.addInequality({1, 1, 1, -1});
  //   simplex.findParamLexmin().dump();
  // }

  // { // [a, b] -> {[x, y, z] : x = a and y = b and x >= 0 and y >= 0 and z >= 0 and x + y + z >= 1}
  //   ParamLexSimplex simplex(5, 2);
  //   simplex.addInequality({1, 0, 0, 0, 0, 0});
  //   simplex.addInequality({0, 1, 0, 0, 0, 0});
  //   simplex.addInequality({0, 0, 1, 0, 0, 0});
  //   simplex.addInequality({0, 0, 0, 1, 0, 0});
  //   simplex.addInequality({0, 0, 0, 0, 1, 0});
  //   simplex.addEquality({1, 0, 0, -1, 0, 0});
  //   simplex.addEquality({0, 1, 0, 0, -1, 0});
  //   simplex.addInequality({0, 0, 1, 1, 1, -1});
  //   simplex.findParamLexmin().dump();
  // }

  // { // [a] -> {[x, y] : x = a and x >= 0 and y >= 0 and x + y >= 1}
  //   ParamLexSimplex simplex(3, 1);
  //   simplex.addInequality({1, 0, 0, 0});
  //   simplex.addInequality({0, 1, 0, 0});
  //   simplex.addInequality({0, 0, 1, 0});
  //   simplex.addEquality({-1, 1, 0, 0});
  //   simplex.addInequality({0, 1, 1, -1});
  //   simplex.findParamLexmin().dump();
  // }

  // {
  //   /*
  //   [x, y] -> {[z] : 0 <= x and x <= 1 and
  //                    0 <= y and y <= 1 and
  //                    0 <= z and z <= 1 and
  //                    x + y + z >= 1}
  //   */
  //   ParamLexSimplex simplex(3, 2);
  //   simplex.addInequality({1, 0, 0, 0});
  //   simplex.addInequality({0, 1, 0, 0});
  //   simplex.addInequality({0, 0, 1, 0});
  //   simplex.addInequality({-1, 0, 0, 1});
  //   simplex.addInequality({0, -1, 0, 1});
  //   simplex.addInequality({0, 0, -1, 1});
  //   simplex.addInequality({1, 1, 1, -1});
  //   simplex.findParamLexmin().dump();
  // }

  // {
    
  //   [x, y, z, w] -> {[z] : 0 <= x and x <= 1 and
  //                    0 <= y and y <= 1 and
  //                    0 <= z and z <= 1 and
  //                    0 <= w and w <= 1 and
  //                    x + y + z >= 1 and
  //                    3 - x - y - w >= 1 and
  //                    w + x + 1 - y >= 1}
    
  //   ParamLexSimplex simplex(4, 4);
  //   simplex.addInequality({1, 0, 0, 0, 0});
  //   simplex.addInequality({0, 1, 0, 0, 0});
  //   simplex.addInequality({0, 0, 1, 0, 0});
  //   simplex.addInequality({0, 0, 0, 1, 0});

  //   simplex.addInequality({-1, 0, 0, 0, 1});
  //   simplex.addInequality({0, -1, 0, 0, 1});
  //   simplex.addInequality({0, 0, -1, 0, 1});
  //   simplex.addInequality({0, 0, 0, -1, 1});

  //   simplex.addInequality({1, 1, 1, 0, -1});
  //   simplex.addInequality({-1, -1, 0, -1, 2});
  //   simplex.addInequality({1, -1, 0, 1, 0});
  //   simplex.findParamLexmin().dump();
  // }

  // { // [a] -> {[x] : x = a}
  //   ParamLexSimplex simplex(2, 1);
  //   simplex.addInequality({1, 0, 0});
  //   simplex.addInequality({0, 1, 0});
  //   simplex.addEquality({1, -1, 0});
  //   simplex.findParamLexmin().dump();
  // }

  // { // [a, b] -> {[x] : x = a and x >= b}
  //   ParamLexSimplex simplex(3, 2);
  //   simplex.addInequality({1, 0, 0, 0});
  //   simplex.addInequality({0, 1, 0, 0});
  //   simplex.addInequality({0, 0, 1, 0});
  //   simplex.addEquality({-1, 0, 1, 0});
  //   simplex.addInequality({0, -1, 1, 0});
  //   simplex.findParamLexmin().dump();
  // }  

  // { // [x] -> {[y] : x = 1 + 3y and y >= 0}
  //   ParamLexSimplex simplex(2, 1);
  //   simplex.addInequality({1, 0, 0}); // x >= 0
  //   simplex.addInequality({0, 1, 0}); // y >= 0
  //   simplex.addEquality({1, -3, -1}); // x == 3y + 1
  //   simplex.findParamLexmin().dump();
  // }

  // { // [x] -> {[y, z] : x = y + 3z and z >= 0 and y = 1}
  //   ParamLexSimplex simplex(3, 1);
  //   simplex.addInequality({1, 0, 0, 0}); // x >= 0
  //   simplex.addInequality({0, 1, 0, 0}); // y >= 0
  //   simplex.addInequality({0, 0, 1, 0}); // z >= 0
  //   simplex.addEquality({1, -1, -3, 0}); // x == y + 3z
  //   simplex.addEquality({0, 1, 0, -1});  // y = 1
  //   simplex.findParamLexmin().dump();
  // }

  // { // [x] -> {[y, z] : x = y + 3z and z >= 0 and y = 0}
  //   ParamLexSimplex simplex(3, 1);
  //   simplex.addInequality({1, 0, 0, 0}); // x >= 0
  //   simplex.addInequality({0, 1, 0, 0}); // y >= 0
  //   simplex.addInequality({0, 0, 1, 0}); // z >= 0
  //   simplex.addEquality({1, -1, -3, 0}); // x == y + 3z
  //   simplex.addEquality({0, 1, 0, 0});   // y == 0
  //   simplex.findParamLexmin().dump();
  // }

  // { // [x] -> {[y, z] : x = y + 3z and z >= 0 and y = 2}
  //   ParamLexSimplex simplex(3, 1);
  //   simplex.addInequality({1, 0, 0, 0}); // x >= 0
  //   simplex.addInequality({0, 1, 0, 0}); // y >= 0
  //   simplex.addInequality({0, 0, 1, 0}); // z >= 0
  //   simplex.addEquality({1, -1, -3, 0}); // x == y + 3z
  //   simplex.addEquality({0, 1, 0, -2});   // y == 2
  //   simplex.findParamLexmin().dump();
  // }


  // // NOT CHECKED ANSWER; does not crash.
  // { // [x] -> {[y, z] : x = y + 3z and z >= 0 and 0 <= y and y <= 1}
  //   ParamLexSimplex simplex(3, 1);
  //   simplex.addInequality({1, 0, 0, 0});  // x >= 0
  //   simplex.addInequality({0, 1, 0, 0});  // y >= 0
  //   simplex.addInequality({0, 0, 1, 0});  // z >= 0
  //   simplex.addEquality({1, -1, -3, 0});  // x == y + 3z
  //   simplex.addInequality({0, -1, 0, 1}); // 1 >= y
  //   simplex.findParamLexmin().dump();
  // }

  // { // [x] -> {[y, z] : x = y + 2z and z >= 0 and 1 <= y and y <= 2}
  //   ParamLexSimplex simplex(3, 1);
  //   simplex.addInequality({1, 0, 0, 0});  // x >= 0
  //   simplex.addInequality({0, 1, 0, 0});  // y >= 0
  //   simplex.addInequality({0, 0, 1, 0});  // z >= 0
  //   simplex.addEquality({1, -1, -2, 0});  // x == y + 2z
  //   simplex.addInequality({0, 1, 0, -1}); // y >= 1
  //   simplex.addInequality({0, -1, 0, 2}); // y <= 2
  //   simplex.findParamLexmin().dump();
  // } 

  // // Seems correct
  // { // [x] -> {[y, z] : x = y + 3z and z >= 0 and 2 <= y and y <= 2}
  //   ParamLexSimplex simplex(3, 1);
  //   simplex.addInequality({1, 0, 0, 0});  // x >= 0
  //   simplex.addInequality({0, 1, 0, 0});  // y >= 0
  //   simplex.addInequality({0, 0, 1, 0});  // z >= 0
  //   simplex.addEquality({1, -1, -3, 0});  // x == y + 3z
  //   simplex.addInequality({0, 1, 0, -2}); // y >= 2
  //   simplex.addInequality({0, -1, 0, 2}); // y <= 2
  //   simplex.findParamLexmin().dump();
  // }

  // // NOT CHECKED ANSWER; does not crash.
  // { // [x] -> {[y, z] : x = y + 3z and z >= 0 and 0 <= y and y <= 2}
  //   ParamLexSimplex simplex(3, 1);
  //   simplex.addInequality({1, -1, 0, 0});  // x >= 0
  //   simplex.addInequality({0, 0, 1, 0});  // z >= 0
  //   simplex.addInequality({0, 1, 0, 0});  // y >= 0
  //   simplex.addInequality({0, -1, 0, 1}); // y <= 1
  //   simplex.addEquality({1, -1, -3, 0});  // x == y + 3z
  //   simplex.findParamLexmin().dump();
  // }

  // // CRASHES.
  // { // [x] -> {[y, z] : x = y + 3z and x >= y and z >= 0 and y >= 0}
  //   ParamLexSimplex simplex(3, 1);
  //   simplex.addInequality({1, -1, 0, 0});  // x >= y
  //   simplex.addInequality({0, 0, 1, 0});  // z >= 0
  //   simplex.addInequality({0, 1, 0, -0});  // y >= 0
  //   simplex.addEquality({1, -1, -3, 0});  // x == y + 3z
  //   simplex.findParamLexmin().dump();
  // }

  { // [x] -> {[y, z] : x = y + 3z and x >= y and z >= 0 and y >= 0}
    ParamLexSimplex simplex(3, 1);
    simplex.addInequality({1, -1, 0, 0});  // x >= y
    simplex.addInequality({0, 1, 0, -0});  // y >= 0
    simplex.addEquality({1, -1, -3, 0});  // x == y + 3z
    simplex.findParamLexmin().dump();
  }
}

} // namespace mlir
