#include "mlir/Analysis/Presburger/Coalesce.h"
#include "mlir/Dialect/Presburger/Parser.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::presburger;

PresburgerSet setFromString(StringRef string) {
  ErrorCallback callback = [](SMLoc loc, const Twine &message) {
   llvm_unreachable("Parser Callback");
   MLIRContext context;
   return mlir::emitError(UnknownLoc::get(&context), message); 
  };
  PresburgerSetParser parser(string, callback);
  PresburgerSet res;
  parser.parsePresburgerSet(res);
  return res;
}


void expectContainedFacet(bool expected, SmallVector<int64_t, 8> &ineq, FlatAffineConstraints &bs,
                          SmallVector<SmallVector<int64_t, 8>, 8> &cut) {
  EXPECT_TRUE(expected == containedFacet(ineq, bs, cut));
}

TEST(CoalesceTest, containedFacet1) {
  PresburgerSet set = setFromString("(x, y) : (x >= 0 and x <= 3 and y >= 0 and y <= 3)");
  FlatAffineConstraints bs = set.getFlatAffineConstraints()[0];
  SmallVector<int64_t, 8> ineq;
  ineq.push_back(1);
  ineq.push_back(0);
  ineq.push_back(0);
  SmallVector<int64_t, 8> cutConstraint1;
  cutConstraint1.push_back(1);
  cutConstraint1.push_back(0);
  cutConstraint1.push_back(2);
  SmallVector<SmallVector<int64_t, 8>, 8> cutting = {cutConstraint1};
  expectContainedFacet(true, ineq, bs, cutting);
}

TEST(CoalesceTest, containedFacet2) {
  PresburgerSet set = setFromString("(x, y) : (x >= 0 and x <= 3 and y >= 0 and y <= 3)");
  FlatAffineConstraints bs = set.getFlatAffineConstraints()[0];
  SmallVector<int64_t, 8> ineq;
  ineq.push_back(1);
  ineq.push_back(0);
  ineq.push_back(0);
  SmallVector<int64_t, 8> cutConstraint1;
  cutConstraint1.push_back(1);
  cutConstraint1.push_back(0);
  cutConstraint1.push_back(2);
  SmallVector<int64_t, 8> cutConstraint2;
  cutConstraint2.push_back(1);
  cutConstraint2.push_back(0);
  cutConstraint2.push_back(1);
  SmallVector<SmallVector<int64_t, 8>, 8> cutting = {cutConstraint1, cutConstraint2};
  expectContainedFacet(true, ineq, bs, cutting);
}

TEST(CoalesceTest, containedFacet3) {
  PresburgerSet set = setFromString("(x, y) : (x >= 0 and x <= 3 and y >= 0 and y <= 3)");
  FlatAffineConstraints bs = set.getFlatAffineConstraints()[0];
  SmallVector<int64_t, 8> ineq;
  ineq.push_back(1);
  ineq.push_back(0);
  ineq.push_back(0);
  SmallVector<int64_t, 8> cutConstraint1;
  cutConstraint1.push_back(1);
  cutConstraint1.push_back(0);
  cutConstraint1.push_back(-5);
  SmallVector<SmallVector<int64_t, 8>, 8> cutting = {cutConstraint1};
  expectContainedFacet(false, ineq, bs, cutting);
}

TEST(CoalesceTest, containedFacet4) {
  PresburgerSet set = setFromString("(x, y) : (x >= 0 and x <= 3 and y >= 0 and y <= 3)");
  FlatAffineConstraints bs = set.getFlatAffineConstraints()[0];
  SmallVector<int64_t, 8> ineq;
  ineq.push_back(1);
  ineq.push_back(0);
  ineq.push_back(0);
  SmallVector<int64_t, 8> cutConstraint1;
  cutConstraint1.push_back(0);
  cutConstraint1.push_back(1);
  cutConstraint1.push_back(2);
  SmallVector<int64_t, 8> cutConstraint2;
  cutConstraint2.push_back(1);
  cutConstraint2.push_back(0);
  cutConstraint2.push_back(-5);
  SmallVector<SmallVector<int64_t, 8>, 8> cutting = {cutConstraint1, cutConstraint2};
  expectContainedFacet(false, ineq, bs, cutting);
}

void expectWrapping(Optional<SmallVector<int64_t, 8>> expected, FlatAffineConstraints bs,
                    SmallVector<int64_t, 8> valid, SmallVector<int64_t, 8> invalid) {
  if (!expected) {
    EXPECT_FALSE(wrapping(bs, valid, invalid).hasValue());
  } else {
    auto result = wrapping(bs, valid, invalid);
    EXPECT_TRUE(result.hasValue());
    if (!result) {
      EXPECT_TRUE(false);
    } else {
      EXPECT_TRUE(sameConstraint(result.getValue(), expected.getValue()));
    }
  }
}

TEST(CoalesceTest, wrapping) {
  PresburgerSet set1 = setFromString("(x,y) : (x = y and y <= 4 and y >= 0)");
  FlatAffineConstraints bs1 = set1.getFlatAffineConstraints()[0];
  SmallVector<int64_t, 8> valid1, invalid1, expected1;
  valid1.push_back(-1);
  valid1.push_back(1);
  valid1.push_back(1);
  invalid1.push_back(0);
  invalid1.push_back(-1);
  invalid1.push_back(3);
  expected1.push_back(-1);
  expected1.push_back(0);
  expected1.push_back(4);
  expectWrapping(expected1, bs1, valid1, invalid1);
  PresburgerSet set2 = setFromString("(x,y) : (2x = 3y and x >= 0 and x <= 6)");
  FlatAffineConstraints bs2 = set2.getFlatAffineConstraints()[0];
  SmallVector<int64_t, 8> valid2, invalid2, expected2;
  valid2.push_back(-2);
  valid2.push_back(3);
  valid2.push_back(1);
  invalid2.push_back(0);
  invalid2.push_back(1);
  invalid2.push_back(-1);
  expected2.push_back(-1);
  expected2.push_back(2);
  expected2.push_back(0);
  expectWrapping(expected2, bs2, valid2, invalid2);
}


void expectCombineConstraint(SmallVector<int64_t, 8> expected, SmallVector<int64_t, 8> c1, SmallVector<int64_t, 8> c2, Fraction<int64_t> ratio) {
  SmallVector<int64_t, 8> result = combineConstraint(c1, c2, ratio);
  EXPECT_TRUE(sameConstraint(expected, result));
}

TEST(CoalesceTest, combineConstraint) {
  SmallVector<int64_t, 8> c1, c2, expected;
  c1.push_back(0);
  c1.push_back(1);
  c1.push_back(1);
  c2.push_back(3);
  c2.push_back(1);
  c2.push_back(5);
  Fraction<int64_t> ratio1(0, 1);
  Fraction<int64_t> ratio2(3, 2);
  expectCombineConstraint(c2, c1, c2, ratio1);
  expected.push_back(6);
  expected.push_back(-1);
  expected.push_back(7); 
  expectCombineConstraint(expected, c1, c2, ratio2);
}


TEST(CoalesceTest, addAsIneq) {
  SmallVector<SmallVector<int64_t, 8>, 8> vec_1;
  SmallVector<SmallVector<int64_t, 8>, 8> vec_2;
  SmallVector<int64_t, 8> const_1;
  const_1.push_back(1);
  const_1.push_back(-3);  
  vec_1.push_back(const_1);
  addAsIneq(vec_1, vec_2);
  EXPECT_TRUE(vec_1.size() == 1);
  EXPECT_TRUE(vec_2.size() == 2);
  auto ineq_1 = vec_2[0];
  auto ineq_2 = vec_2[1];
  EXPECT_TRUE(ineq_1[0] == 1);
  EXPECT_TRUE(ineq_1[1] == -3);
  EXPECT_TRUE(ineq_2[0] == -1);
  EXPECT_TRUE(ineq_2[1] == 3);
}

void expectCoalesce(size_t expectedNumBasicSets, PresburgerSet set) {
  PresburgerSet new_set = coalesce(set);
  EXPECT_TRUE(PresburgerSet::equal(set, new_set));
  EXPECT_TRUE(expectedNumBasicSets == new_set.getFlatAffineConstraints().size());
}

TEST(CoalesceTest, contained) {
  PresburgerSet contained = setFromString("(x0) : (x0 >= 0 and x0 <= 4 or x0 >= 1 and x0 <= 3)");
  expectCoalesce(1, contained);
}

TEST(CoalesceTest, cut) {
  PresburgerSet cut = setFromString("(x0) : (x0 >= 0 and x0 <= 3 or x0 >= 2 and x0 <= 4)");
  expectCoalesce(1, cut);
}

TEST(CoalesceTest, adj_ineq) {
  PresburgerSet adj_ineq = setFromString("(x0) : (x0 >= 0 and x0 <= 1 or x0 >= 2 and x0 <= 3)");
  expectCoalesce(1, adj_ineq);
}

TEST(CoalesceTest, separate) {
  PresburgerSet separate = setFromString("(x0) : (x0 >= 0 and x0 <= 1 or x0 >= 3 and x0 <= 4)");
  expectCoalesce(2, separate);
}

TEST(CoalesceTest, adj_eq) {
  PresburgerSet adj_eq = setFromString("(x0) : (x0 = 1 or x0 >= 2 and x0 <= 3)");
  expectCoalesce(1, adj_eq);
}

TEST(CoalesceTest, adj_eqs) {
  PresburgerSet adj_eq = setFromString("(x0) : (x0 = 1 or x0 = 2)");
  expectCoalesce(1, adj_eq);
}

TEST(CoalesceTest, eq_separate) {
  PresburgerSet eq_separate = setFromString(" (x0) : (x0 = 0 or x0 >= 2 and x0 <= 3)");
  expectCoalesce(2, eq_separate);
}

TEST(CoalesceTest, eq_as_ineq_contained) {
  PresburgerSet eq_as_ineq = setFromString(" (x0) : (x0 <= 3 and x0 >= 3 or x0 >=2 and x0 <= 3)");
  expectCoalesce(1, eq_as_ineq);
}

TEST(CoalesceTest, eq_contained) {
  PresburgerSet eq_contained = setFromString(" (x0) : (x0 = 3 or x0 >= 2 and x0 <= 3)");
  expectCoalesce(1, eq_contained);
}

TEST(CoalesceTest, multi_dim_contained) {
  PresburgerSet multi_dim_contained = setFromString (
      " (x0, x1) : (x0 >= 0 and x0 <= 4 and x1 >= 0 and x1 <= 4 or x0 >= 2 and x0 <= 3 and x1 >= 2 and x1 <= 3)");
  expectCoalesce(1, multi_dim_contained);
}

TEST(CoalesceTest, multi_dim_adj_ineq) {
  PresburgerSet multi_dim_adj_ineq = setFromString(
      " (x0, x1) : (x0 >= 0 and x0 <= 3 and x1 >= 0 and x1 <= 1 or x0 >= 0 and "
      "x0 <= 3 and x1 <= 2 and x1 <= 3)");
  expectCoalesce(1, multi_dim_adj_ineq);
}

TEST(CoalesceTest, multi_dim_separate) {
  PresburgerSet multi_dim_separate = setFromString(
      "(x0, x1) : (x0 >= 0 and x0 <= 1 and x1 >= 0 and x1 <= 1 or x0 >= 2 and "
      "x0 <= 3 and x1 >= 2 and x1 <= 3)");
  expectCoalesce(2, multi_dim_separate);
}

TEST(CoalesceTest, multi_dim_cut) {
  PresburgerSet multi_dim_cut = setFromString(
      "(x,y) : (4x -5y <= 0 and y <= 4 and 3x + 4y >= 0 and x - y + 7 >= 0 or "
      "y >= 0 and x <= 0 and y <= 4 and x - y + 9 >= 0 and 2x + 5y + 4 >= 0)");
  expectCoalesce(1, multi_dim_cut);
}

TEST(CoalesceTest, multi_dim_non_cut) {
  PresburgerSet multi_dim_non_cut = setFromString("(x,y) :  (y >= 0 and x <= 0 and y <= 4 and x - y + 9 "
                        ">= 0 and 2x + 5y + 4 >= 0 or 4x -5y <= 0 and 3x + 4y "
                        ">= 0 and x - y + 7 >= 0 and x + y - 10 <= 0)");
  expectCoalesce(2, multi_dim_non_cut);
}

TEST(CoalesceTest, multi_dim_adj_eqs) {
  PresburgerSet multi_dim_adj_eqs = setFromString("(x,y) :  (y = x and y >= 0 and y <= 4 or x - 1 = y "
                        "and y >= 0 and y <= 3)");
  expectCoalesce(1, multi_dim_adj_eqs);
}

TEST(CoalesceTest, multi_dim_adj_eqs2) {
  PresburgerSet multi_dim_adj_eqs2 = setFromString("(x,y) :  (y = x and y >= 0 and y <= 4 or x - 1 = y "
                         "and y >= 1 and y <= 3)");
  expectCoalesce(1, multi_dim_adj_eqs2);
}

TEST(CoalesceTest, multi_dim_adj_eqs3) {
  PresburgerSet multi_dim_adj_eqs2 = setFromString("(x,y) : ( 2x = 3y and y >= 0 and x <= 6 or 2x = 3y+1 "
                         "and y >= 2 and y <= 5)");
  expectCoalesce(1, multi_dim_adj_eqs2);
}

TEST(CoalesceTest, multi_dim_adj_eq_to_poly) {
  PresburgerSet multi_dim_adj_eq_to_poly = setFromString(
      "(x,y) :  (x <= 0 and y >= 0 and y <= 4 and x >= -8 and x <= -y + 3 or x "
      "= 1 and y >= 0 and y <= 2)");
  expectCoalesce(1, multi_dim_adj_eq_to_poly);
}

TEST(CoalesceTest, multi_dim_adj_eq_to_poly_complex) {
  PresburgerSet multi_dim_adj_eq_to_poly = setFromString(
      "(x,y) :  (x <= 0 and y >= 0 and y <= 4 and x >= -8 and x <= -y + 3 or x "
      "= 1 and y >= 1 and y <= 2)");
  expectCoalesce(1, multi_dim_adj_eq_to_poly);
}

TEST(CoalesceTest, multi_sets_all_contained) {
  PresburgerSet multi_sets_all_contained = setFromString("(x0) : (x0 >= 0 and x0 <= 5 or x0 >= 2 and x0 "
                               "<= 3 or x0 >= 4 and x0 <= 5)");
  expectCoalesce(1, multi_sets_all_contained);
}

TEST(CoalesceTest, multi_sets_one_contained) {
  PresburgerSet multi_sets_one_contained = setFromString("(x0) : (x0 >= 0 and x0 <= 3 or x0 >= 2 and x0 "
                               "<= 3 or x0 >= 2 and x0 <= 5)");
  expectCoalesce(1, multi_sets_one_contained);
}

TEST(CoalesceTest, multi_set_not_contained) {
  PresburgerSet multi_sets_not_contained = setFromString("(x0) : (x0 >= 0 and x0 <= 1 or x0 >= 2 and x0 "
                               "<= 3 or x0 >= 4 and x0 <= 5)");
  expectCoalesce(1, multi_sets_not_contained);
}

TEST(CoalesceTest, protrusion) {
  PresburgerSet protrusion = setFromString("(x,y) : (x >= 0 and y >= 0 and y <= 3 and x <= 9 or y >= 1 "
                 "and y <= 4 and y <= x - 1 and x + y - 11 <= 0)");
  expectCoalesce(1, protrusion);
}

TEST(CoalesceTest, nearly_protrusion) {
  PresburgerSet protrusion = setFromString("(x,y) : (x >= 0 and y >= 0 and y <= 3 and x <= 9 or y >= 1 "
                 "and y <= 5 and y <= x - 1 and x + y - 11 <= 0)");
  expectCoalesce(2, protrusion);
}


