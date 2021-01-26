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
    llvm::errs() << "Parsing error " << message << " at " << loc.getPointer()
                 << '\n';
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
  // expectEqual(
  //     "(x) : (exists y, z : x = y + 3z and x >= y and z >= 0 and y >= 0)",
  //     "(x) : (x >= 0)");
  // expectEqual(
  //     "(x) : (exists y, z : x = y + 3z and x >= y and z >= 0 and y >= 0)",
  //     "(x) : (exists y, z : x = y + 3z and x >= y and z >= 0 and y >= 0)");
  // expectEqual("(x) : (exists q = [(x)/2] : x = 2q)",
  //             "(x) : (exists q = [(x)/2] : x = 2q)");
  // expectEqual(
  //     "(x) : (exists q = [(x)/2] : x = 2q) or (exists q = [(x)/3] : x = 3q)",
  //     "(x) : (exists q = [(x)/2] : x = 2q) or (exists q = [(x)/3] : x =
  //     3q)");
  // expectEqual("(x) : (exists q : x = q and q <= -1)", "(x) : (x <= -1)");

  auto set = setFromString("(x0, x2, x3, x4, x5, y0, y2, y3, y4, y5) : ("
                           "-x0 + x2 - 4x3 + 32x4 - 32x5 - 1 >= 0 and "
                           "-x0 + x2 - 4x3 + 32x4 - 32x5 - 1 <= 0 and "
                           "x0 + 4x3 - 32x4 + 32x5 >= 0 and "
                           "x0 - 32x4 + 32x5 + 31 >= 0 and "
                           "-x0 + 32x4 - 32x5 >= 0 and "
                           "-x0 - 4x3 + 32x4 - 32x5 + 31 >= 0 and "
                           "x0 + 32x5 + 31 >= 0 and "
                           "-x0 - 32x5 >= 0 and "
                           "-x0 - 4x3 + 32x4 + 29 >= 0 and "
                           "x0 + 4x3 - 32x4 + 2 >= 0 and"
                           "-y0 + y2 - 4y3 + 32y4 - 32y5 - 1 >= 0 and "
                           "-y0 + y2 - 4y3 + 32y4 - 32y5 - 1 <= 0 and "
                           "y0 + 4y3 - 32y4 + 32y5 >= 0 and "
                           "y0 - 32y4 + 32y5 + 31 >= 0 and "
                           "-y0 + 32y4 - 32y5 >= 0 and "
                           "-y0 - 4y3 + 32y4 - 32y5 + 31 >= 0 and "
                           "y0 + 32y5 + 31 >= 0 and "
                           "-y0 - 32y5 >= 0 and "
                           "-y0 - 4y3 + 32y4 + 29 >= 0 and "
                           "y0 + 4y3 - 32y4 + 2 >= 0 and "
                           "x3 - y3 >= 0 and "
                           "x3 - y3 <= 0 and "
                           "x2 - y2 >= 0 and "
                           "x2 - y2 <= 0)"
                           /*stay here*/);
  SmallVector<int64_t, 8> duals = {
      0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 1, 4, 0, 0, 1,
  };

  SmallVector<int64_t, 8> dualsCorrect = {
      0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
  };
  EXPECT_TRUE(set.getBasicSets()[0].satisfiesDual(duals));
  EXPECT_TRUE(set.getBasicSets()[0].satisfiesDual(dualsCorrect));
}
