#include "./ParserUtil.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Parser.h"

using namespace mlir;

FailureOr<FlatAffineConstraints> mlir::parseFAC(StringRef str) {
  MLIRContext context;
  IntegerSet set = parseIntegerSet(str, &context);
  if (!set)
    return failure();

  return FlatAffineConstraints(set);
}

