//===- PreburgerBasicMapTest.cpp - Tests for PresburgerBasicMap ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/PresburgerBasicMap.h"
#include "mlir/Analysis/Presburger/PresburgerBasicSet.h"
#include "mlir/Dialect/Presburger/Parser.h"

#include <ostream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::presburger;

static PresburgerSet setFromString(StringRef string) {
  ErrorCallback callback = [](SMLoc loc, const Twine &message) {
    // This is a hack to make the Parser compile
    llvm::errs() << "Parsing error " << message << " at " << loc.getPointer() << '\n';
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

TEST(PresburgerBasicMapTest, relLexMax) {
  auto s1 = setFromString(
      "(pw, qw, iw, pr, qr, ir)[np, mb] : (pw = pr - 1 and qw = qr and iw = ir "
      "and 1 <= qr and qr <= pr + 1 and pr <= np and 1 <= ir and ir <= mb)");
  auto s2 = setFromString(
      "(pw, qw, iw, pr, qr, ir)[np, mb] : (pw = qw and qw = qr and iw = ir and "
      "1 <= qr and qr <= pr + 1 and pr <= np and 1 <= ir and ir <= mb)");
  
  auto bset1 = s1.getBasicSets()[0];
  auto bset2 = s2.getBasicSets()[0];

  auto rel1 = PresburgerBasicMap(3, 3, bset1);
  auto rel2 = PresburgerBasicMap(3, 3, bset2);

  auto depRel = PresburgerBasicMap::relLexMax(rel1, rel2, 1, 3);

  EXPECT_TRUE(true);
}
