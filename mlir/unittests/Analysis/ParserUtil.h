#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
FailureOr<FlatAffineConstraints> parseFAC(llvm::StringRef);
} // namespace mlir
