//===- SimplexTest.cpp - Tests for ParamLexSimplex ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Presburger-impl.h"
#include <ostream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using Int = int64_t;

using namespace mlir;
using namespace mlir::presburger;

static PresburgerSet<Int> setFromString(StringRef string) {
  ErrorCallback callback = [](SMLoc loc, const Twine &message) {
    // This is a hack to make the Parser compile
    llvm::errs() << "Parsing error " << message << " at " << loc.getPointer()
                 << '\n';
    llvm_unreachable("PARSING ERROR!!");
    MLIRContext context;
    return mlir::emitError(UnknownLoc::get(&context), message);
  };
  Parser<Int> parser(string, callback);
  PresburgerParser<Int> setParser(parser);
  PresburgerSet<Int> res;
  setParser.parsePresburgerSet(res);
  return res;
}

void expectEqual(StringRef sDesc, StringRef tDesc) {
  auto s = setFromString(sDesc);
  auto t = setFromString(tDesc);
  EXPECT_TRUE(PresburgerSet<Int>::equal(s, t));
}

TEST(PresburgerSetTest, Equality) {
  expectEqual(
      "(x) : (exists y, z : x = y + 3z and x >= y and z >= 0 and y >= 0)",
      "(x) : (x >= 0)");
  expectEqual(
      "(x) : (exists y, z : x = y + 3z and x >= y and z >= 0 and y >= 0)",
      "(x) : (exists y, z : x = y + 3z and x >= y and z >= 0 and y >= 0)");
  expectEqual("(x) : (exists q = [(x)/2] : x = 2q)",
              "(x) : (exists q = [(x)/2] : x = 2q)");
  expectEqual(
      "(x) : (exists q = [(x)/2] : x = 2q) or (exists q = [(x)/3] : x = 3q)",
      "(x) : (exists q = [(x)/2] : x = 2q) or (exists q = [(x)/3] : x = 3q)");
  expectEqual("(x) : (exists q : x = q and q <= -1)", "(x) : (x <= -1)");
}

TEST(PresburgerSetTest, Empty) {
  EXPECT_TRUE(
      setFromString(
          "(d0) : (exists q0 = [(2d0)/3], q1 = [(2d0)/3], q2 = [(q1)/2], q3 = "
          "[(2d0)/3], q4 = [(q3)/2] : d0 >= 0 and d0 - 1 >= 0 and 2d0 - 3q0 - "
          "1 >= 0 and -2d0 + 3q1 >= 0 and -2d0 + 3q3 >= 0)")
          .isIntegerEmpty());
}
