//===- ParserTest.cpp - Tests for Parser ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Parser.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {

TEST(ParserTest, emptySet) {
  MLIRContext ctx;
  auto str = "(i)[] : ()";

  FailureOr<PresburgerSet> set = parsePresburgerSet(str, &ctx);

  EXPECT_TRUE(succeeded(set));
}

TEST(ParserTest, invalid) {
  MLIRContext ctx;
  auto str = "(i)[] : (i = )";

  FailureOr<PresburgerSet> set = parsePresburgerSet(str, &ctx);

  EXPECT_TRUE(failed(set));
}

TEST(ParserTest, simpleEq) {
  MLIRContext ctx;
  auto str = "(i)[] : (i = 0)";

  FailureOr<PresburgerSet> set = parsePresburgerSet(str, &ctx);

  EXPECT_TRUE(succeeded(set));
}

TEST(ParserTest, simpleIneq) {
  MLIRContext ctx;
  auto str = "(i)[] : (i >= 0)";

  FailureOr<PresburgerSet> set = parsePresburgerSet(str, &ctx);

  EXPECT_TRUE(succeeded(set));
}

TEST(ParserTest, simpleAnd) {
  MLIRContext ctx;
  auto str = "(i)[] : (i >= 0 and i <= 3)";

  FailureOr<PresburgerSet> set = parsePresburgerSet(str, &ctx);

  EXPECT_TRUE(succeeded(set));
}

TEST(ParserTest, simpleOr) {
  MLIRContext ctx;
  auto str = "(i)[] : (i >= 0 or i <= 3)";

  FailureOr<PresburgerSet> set = parsePresburgerSet(str, &ctx);

  EXPECT_TRUE(succeeded(set));
}
} // namespace mlir
