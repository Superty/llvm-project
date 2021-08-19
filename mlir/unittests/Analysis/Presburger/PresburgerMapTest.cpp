//===- PresburgerMapTest.cpp - Tests for ParamLexSimplex ------------------------===//
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

inline static PresburgerSet<Int> setFromString(StringRef string) {
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

inline void expectEqual(StringRef sDesc, StringRef tDesc) {
  auto s = setFromString(sDesc);
  auto t = setFromString(tDesc);
  EXPECT_TRUE(PresburgerSet<Int>::equal(s, t));
}

TEST(PresburgerSet, codeLexMaxTest) {
  PresburgerSet set = setFromString("(k, i, j)[n, m] : (0 <= i and i <= m and "
                                    "0 <= j and j <= n and 2i + j = k)");
  PresburgerMap<Int> map(1, 2, 2);

  for (const auto &bs: set.getBasicSets())
    map.addBasicSet(bs);

  map.dump();

  map.lexMaxRange();

  map.dump();

  EXPECT_TRUE(true);
}
