#ifndef PRESBURGER_PARSER_H
#define PRESBURGER_PARSER_H

#include "mlir/Analysis/PresburgerSet.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// PresburgerParser
//===----------------------------------------------------------------------===//

FailureOr<PresburgerSet> parsePresburgerSet(StringRef str, MLIRContext *ctx);

}; // namespace mlir

#endif // PRESBURGER_PARSER_H
