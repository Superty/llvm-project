#include "mlir/Analysis/Presburger/Coalesce.h"
#include "mlir/Dialect/Presburger/Parser.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::presburger;

void expectCoalesce(size_t expectedNumBasicSets, PresburgerSet set) {
  PresburgerSet new_set = coalesce(set);
  EXPECT_TRUE(PresburgerSet::equal(set, new_set));
  EXPECT_TRUE(expectedNumBasicSets == new_set.getFlatAffineConstraints().size());
}

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

TEST(CoalesceTest, contained) {
  PresburgerSet contained = setFromString("(x0) : (x0 >= 0 and x0 <= 4 or x0 >= 1 and x0 <= 3)");
  expectCoalesce(1, contained);
}


