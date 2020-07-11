#ifndef MLIR_CONVERSION_PRESBURGERTOSTANDARD_PRESBURGERTOSTANDARD_H
#define MLIR_CONVERSION_PRESBURGERTOSTANDARD_PRESBURGERTOSTANDARD_H

namespace mlir {
class MLIRContext;

// Owning list of rewriting patterns.
class OwningRewritePatternList;

/// Collect a set of patterns to convert from the Presburger dialect to the
/// Standard dialect
void populatePresburgerToStdConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx);
} // namespace mlir

#endif // MLIR_CONVERSION_PRESBURGERTOSTANDARD_PRESBURGERTOSTANDARD_H
