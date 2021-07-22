//===- ParserTest.cpp - Tests for Parser ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/PresburgerParser.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {

/// Construct a FlatAffineConstraints from a set of inequality and
/// equality constraints.
static FlatAffineConstraints makeFACFromConstraints(
    unsigned dims, unsigned syms, ArrayRef<SmallVector<int64_t, 4>> ineqs,
    ArrayRef<SmallVector<int64_t, 4>> eqs,
    ArrayRef<std::pair<SmallVector<int64_t, 4>, int64_t>> divs) {
  FlatAffineConstraints fac(ineqs.size(), eqs.size(), dims + syms + 1, dims,
                            syms);
  for (const auto &div : divs) {
    fac.addLocalFloorDiv(div.first, div.second);
  }
  for (const SmallVector<int64_t, 4> &eq : eqs)
    fac.addEquality(eq);
  for (const SmallVector<int64_t, 4> &ineq : ineqs)
    fac.addInequality(ineq);
  return fac;
}

static FlatAffineConstraints
makeFACFromIneqs(unsigned dims, unsigned syms,
                 ArrayRef<SmallVector<int64_t, 4>> ineqs) {
  return makeFACFromConstraints(dims, syms, ineqs, {}, {});
}

static PresburgerSet makeSetFromFACs(unsigned dims, unsigned syms,
                                     ArrayRef<FlatAffineConstraints> facs) {
  PresburgerSet set = PresburgerSet::getEmptySet(dims, syms);
  for (const FlatAffineConstraints &fac : facs)
    set.unionFACInPlace(fac);
  return set;
}

TEST(ParserTest, universeSet) {
  MLIRContext ctx;
  auto str = "(i)[] : ()";

  FailureOr<PresburgerSet> set = parsePresburgerSet(str, &ctx);

  EXPECT_TRUE(succeeded(set));

  auto u = PresburgerSet::getUniverse(1, 0);
  EXPECT_TRUE(set->isEqual(u));
}

TEST(ParserTest, invalid) {
  MLIRContext ctx;
  auto str = "(i)[] : (i = )";

  FailureOr<PresburgerSet> set = parsePresburgerSet(str, &ctx);
  EXPECT_TRUE(failed(set));

  // The complete StringRef should be checked.
  str = "(i)[] : () )";

  set = parsePresburgerSet(str, &ctx);
  EXPECT_TRUE(failed(set));

  // `and` is only allowed inside a convex set.
  str = "(i)[] : (i <= 2) and (i >= 3)";

  set = parsePresburgerSet(str, &ctx);
  EXPECT_TRUE(failed(set));

  // Reused variable names are not supported
  str = "(i,i)[] : (i <= 2) or (i >= 3)";

  set = parsePresburgerSet(str, &ctx);
  EXPECT_TRUE(failed(set));

  str = "(i)[i] : (i <= 2) or (i >= 3)";

  set = parsePresburgerSet(str, &ctx);
  EXPECT_TRUE(failed(set));

  str = "(i) : ";

  set = parsePresburgerSet(str, &ctx);
  EXPECT_TRUE(failed(set));

  str = "(i) : (i - >= 2) ";

  set = parsePresburgerSet(str, &ctx);
  EXPECT_TRUE(failed(set));

  str = "(i) : (1i >= -) ";

  set = parsePresburgerSet(str, &ctx);
  EXPECT_TRUE(failed(set));

  str = "(i) : (-i - -2 >= - -2) ";

  set = parsePresburgerSet(str, &ctx);
  EXPECT_TRUE(failed(set));
}

TEST(ParserTest, simpleEq) {
  MLIRContext ctx;
  auto str = "(i) : (i = 0)";

  FailureOr<PresburgerSet> set = parsePresburgerSet(str, &ctx);
  EXPECT_TRUE(succeeded(set));

  PresburgerSet ex =
      makeSetFromFACs(1, 0, {makeFACFromConstraints(1, 0, {}, {{1, 0}}, {})});
  EXPECT_TRUE(set->isEqual(ex));
}

TEST(ParserTest, simpleIneq) {
  MLIRContext ctx;
  auto str = "(i)[] : (i >= 0)";

  FailureOr<PresburgerSet> set = parsePresburgerSet(str, &ctx);
  EXPECT_TRUE(succeeded(set));

  PresburgerSet ex = makeSetFromFACs(1, 0, {makeFACFromIneqs(1, 0, {{1, 0}})});
  EXPECT_TRUE(set->isEqual(ex));
}

TEST(ParserTest, simpleAnd) {
  MLIRContext ctx;
  auto str = "(i)[] : (i >= 0 and i <= 3)";

  FailureOr<PresburgerSet> set = parsePresburgerSet(str, &ctx);
  EXPECT_TRUE(succeeded(set));

  PresburgerSet ex =
      makeSetFromFACs(1, 0, {makeFACFromIneqs(1, 0, {{1, 0}, {-1, 3}})});
  EXPECT_TRUE(set->isEqual(ex));
}

TEST(ParserTest, simpleOr) {
  MLIRContext ctx;
  auto str = "(i)[] : (i >= 0) or (i <= 3)";

  FailureOr<PresburgerSet> set = parsePresburgerSet(str, &ctx);
  EXPECT_TRUE(succeeded(set));

  PresburgerSet ex = makeSetFromFACs(
      1, 0,
      {makeFACFromIneqs(1, 0, {{1, 0}}), makeFACFromIneqs(1, 0, {{-1, 3}})});
  EXPECT_TRUE(set->isEqual(ex));
}

TEST(ParserTest, andOr) {
  MLIRContext ctx;
  auto str = "(x) : (x >= 0 and x <= 5) or (x >= -2 and x <= -5)";

  FailureOr<PresburgerSet> set = parsePresburgerSet(str, &ctx);
  EXPECT_TRUE(succeeded(set));

  PresburgerSet ex =
      makeSetFromFACs(1, 0,
                      {makeFACFromIneqs(1, 0, {{1, 0}, {-1, 5}}),
                       makeFACFromIneqs(1, 0, {{1, -2}, {-1, -5}})});
  EXPECT_TRUE(set->isEqual(ex));
}

TEST(ParserTest, higherDim) {
  MLIRContext ctx;
  auto str = "(x,y)[] : (x >= 2 and y >= 2 and x <= 10 and y <= 10 and x + y "
             ">= 2 and x + y <= 30 and x - y >= 0 and x -y <= 10)";

  FailureOr<PresburgerSet> set = parsePresburgerSet(str, &ctx);
  EXPECT_TRUE(succeeded(set));

  PresburgerSet ex =
      makeSetFromFACs(2, 0,
                      {makeFACFromIneqs(2, 0,
                                        {
                                            {1, 0, -2},   // x >= 2.
                                            {0, 1, -2},   // y >= 2.
                                            {-1, 0, 10},  // x <= 10.
                                            {0, -1, 10},  // y <= 10.
                                            {1, 1, -2},   // x + y >= 2.
                                            {-1, -1, 30}, // x + y <= 30.
                                            {1, -1, 0},   // x - y >= 0.
                                            {-1, 1, 10},  // x - y <= 10.
                                        })});
  EXPECT_TRUE(set->isEqual(ex));
}

TEST(ParserTest, coeff1) {
  MLIRContext ctx;
  auto str = "(x) : (1x -2 >= -2)";

  FailureOr<PresburgerSet> set = parsePresburgerSet(str, &ctx);
  EXPECT_TRUE(succeeded(set));

  PresburgerSet ex = makeSetFromFACs(1, 0, {makeFACFromIneqs(1, 0, {{1, 0}})});
  EXPECT_TRUE(set->isEqual(ex));
}

TEST(ParserTest, plusMinus) {
  MLIRContext ctx;
  auto str = "(x) : (-1x + -2 >= 2)";

  FailureOr<PresburgerSet> set = parsePresburgerSet(str, &ctx);
  EXPECT_TRUE(succeeded(set));

  PresburgerSet ex =
      makeSetFromFACs(1, 0, {makeFACFromIneqs(1, 0, {{-1, -4}})});
  EXPECT_TRUE(set->isEqual(ex));
}

TEST(ParserTest, floorDivSimple) {
  MLIRContext ctx;
  auto str = "(x, y) : (y = floor(x / 3))";

  FailureOr<PresburgerSet> set = parsePresburgerSet(str, &ctx);
  EXPECT_TRUE(succeeded(set));

  // TODO why only 2 for the division argument?
  PresburgerSet ex = makeSetFromFACs(
      2, 0,
      {makeFACFromConstraints(2, 0, {}, {{0, 1, -1, 0}}, {{{1, 0, 0}, 3}})});
  set->print(llvm::errs());
  ex.print(llvm::errs());
  EXPECT_TRUE(false) << "No way to compare as of yet!";
  // EXPECT_TRUE(set->isEqual(ex));
}

TEST(ParserTest, floorDivWithCoeffs) {
  MLIRContext ctx;
  auto str = "(x, y) : (y = 3floor(x + y - 13 / 3) + 42)";

  FailureOr<PresburgerSet> set = parsePresburgerSet(str, &ctx);
  EXPECT_TRUE(succeeded(set));

  // TODO why only 2 for the division argument?
  PresburgerSet ex =
      makeSetFromFACs(2, 0,
                      {makeFACFromConstraints(2, 0, {}, {{0, 1, -3, -42}},
                                              {{{1, 1, -13}, 3}})});
  set->print(llvm::errs());
  ex.print(llvm::errs());
  EXPECT_TRUE(false) << "No way to compare as of yet!";
  // EXPECT_TRUE(set->isEqual(ex));
}

TEST(ParserTest, floorDivMultiple) {
  MLIRContext ctx;
  auto str = "(x, y) : (y = floor(x / 3) + floor(y / 2))";

  FailureOr<PresburgerSet> set = parsePresburgerSet(str, &ctx);
  EXPECT_TRUE(succeeded(set));

  // TODO why only 2 for the division argument?
  PresburgerSet ex = makeSetFromFACs(
      2, 0,
      {makeFACFromConstraints(2, 0, {}, {{0, 1, -1, -1, 0}},
                              {{{1, 0, 0}, 3}, {{0, 1, 0, 0}, 2}})});
  set->print(llvm::errs());
  ex.print(llvm::errs());
  EXPECT_TRUE(false) << "No way to compare as of yet!";
  // EXPECT_TRUE(set->isEqual(ex));
}

TEST(ParserTest, floorDivNested) {
  MLIRContext ctx;
  auto str = "(x, y) : (y = floor(x + floor(y / 2) / 3))";

  FailureOr<PresburgerSet> set = parsePresburgerSet(str, &ctx);
  EXPECT_TRUE(succeeded(set));

  // TODO why only 2 for the division argument?
  PresburgerSet ex = makeSetFromFACs(
      2, 0,
      {makeFACFromConstraints(2, 0, {}, {{0, 1, 0, -1, 0}},
                              {{{0, 1, 0}, 2}, {{1, 0, 1, 0}, 3}})});
  set->print(llvm::errs());
  ex.print(llvm::errs());
  EXPECT_TRUE(false) << "No way to compare as of yet!";
  // EXPECT_TRUE(set->isEqual(ex));
}

} // namespace mlir
