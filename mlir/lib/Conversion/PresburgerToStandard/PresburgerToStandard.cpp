#include "mlir/Conversion/PresburgerToStandard/PresburgerToStandard.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Presburger/PresburgerDialect.h"
#include "mlir/Dialect/Presburger/PresburgerOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::presburger;

namespace {

/// This class is a helper for lowering Presburger constructs to the standard
/// dialect.
struct PresburgerSetTransformer {

  PresburgerSetTransformer(OpBuilder &builder, ValueRange dimAndSymbolValues,
                           Location loc)
      : builder(builder), dimAndSymbolValues(dimAndSymbolValues), loc(loc) {}

  /// This creates a sequence of std operations that check if a provided point
  /// satisfies all the Presburger constraints. This is done in a straight
  /// forward fashion by just genreating code that computes the value of an
  /// expression and checks if it satisfies the given constraint. This is done
  /// for all constraints and the end result is true, if every check succeeded.
  ///
  /// Note that there are no short circuit evaluation or other simplification
  /// applied.
  Value lowerPresburgerSet(const PresburgerSet &set) {
    Value condition = builder.create<ConstantIntOp>(loc, 0, 1);
    for (const FlatAffineConstraints &basicSet :
         set.getFlatAffineConstraints()) {
      Value isInBasicSet = lowerFlatAffineConstraints(basicSet);

      condition = builder.create<OrOp>(loc, condition, isInBasicSet);
    }
    return condition;
  }

  Value lowerFlatAffineConstraints(const FlatAffineConstraints &basicSet) {
    if (basicSet.isEmpty()) {
      return builder.create<ConstantIntOp>(loc, 0, 1);
    }

    Value condition = builder.create<ConstantIntOp>(loc, 1, 1);
    for (unsigned i = 0, e = basicSet.getNumEqualities(); i < e; ++i) {
      ArrayRef<int64_t> eq = basicSet.getEquality(i);
      Value isConsSat = lowerPresburgerConstraint(eq.drop_back(), eq.back(),
                                                  CmpIPredicate::eq);
      condition = builder.create<AndOp>(loc, condition, isConsSat);
    }

    for (unsigned i = 0, e = basicSet.getNumInequalities(); i < e; ++i) {
      ArrayRef<int64_t> ineq = basicSet.getInequality(i);
      Value isConsSat = lowerPresburgerConstraint(ineq.drop_back(), ineq.back(),
                                                  CmpIPredicate::sge);
      condition = builder.create<AndOp>(loc, condition, isConsSat);
    }
    return condition;
  }

  Value lowerPresburgerConstraint(ArrayRef<int64_t> coeffs, int64_t c,
                                  CmpIPredicate pred) {
    assert(coeffs.size() == dimAndSymbolValues.size() &&
           "expect coefficients for every dim and symbol");

    Value sum = builder.create<ConstantIndexOp>(loc, c);

    // TODO can we do a kind of zip().fold() ?
    for (unsigned i = 0, e = coeffs.size(); i < e; ++i) {
      Value coeff = builder.create<ConstantIndexOp>(loc, coeffs[i]);
      Value val = dimAndSymbolValues[i];

      Value prod = builder.create<MulIOp>(loc, coeff, val);
      sum = builder.create<AddIOp>(loc, sum, prod);
    }

    Value zeroConstant = builder.create<ConstantIndexOp>(loc, 0);
    Value cmp = builder.create<CmpIOp>(loc, pred, sum, zeroConstant);
    return cmp;
  }

  OpBuilder &builder;
  ValueRange dimAndSymbolValues;
  Location loc;
};

} // namespace

/// Presburger contains are replaced by runtime checks
/// TODO this should perhaps be matched on a complete function, as it otherwise
/// might read outdated values, i.e. violating an invariant of the rewrite
/// framework
class PresburgerContainsLowering : public OpRewritePattern<ContainsOp> {
public:
  using OpRewritePattern<ContainsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ContainsOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // TODO find a nicer way to solve this problem
    // PresburgerSetType setType = op.set().getType().cast<PresburgerSetType>();
    SetOp setDef = op.set().getDefiningOp<SetOp>();
    if (!setDef)
      return failure();
    PresburgerSetAttr setAttr = setDef.getAttrOfType<PresburgerSetAttr>("set");
    PresburgerSet set(setAttr.getValue());

    PresburgerSetTransformer t(rewriter, op.dimAndSyms(), loc);

    Value reduced = t.lowerPresburgerSet(set);
    rewriter.replaceOp(op, reduced);
    return success();
  }
};

void mlir::populatePresburgerToStdConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  // clang-format off
  patterns.insert<PresburgerContainsLowering>(ctx);
  // clang-format on
}

namespace {
class ConvertPresburgerToStandardPass
    : public ConvertPresburgerToStandardBase<ConvertPresburgerToStandardPass> {
  void runOnOperation() override {
    OwningRewritePatternList patterns;
    populatePresburgerToStdConversionPatterns(patterns, &getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<StandardOpsDialect>();
    if (failed(applyPartialConversion(getOperation(), target, patterns)))
      signalPassFailure();
  }
};
} // namespace

/// Lowers Presburger operations to standard dialect
std::unique_ptr<Pass> mlir::createPresburgerToStandardPass() {
  return std::make_unique<ConvertPresburgerToStandardPass>();
}
