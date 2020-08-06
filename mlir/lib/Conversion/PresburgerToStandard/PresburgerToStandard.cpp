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
struct PresburgerTransformer {

  PresburgerTransformer(OpBuilder &builder, ValueRange dimAndSymbolValues,
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
    if (set.getNumBasicSets() == 0) {
      return builder.create<ConstantIntOp>(loc, 1, 1);
    }
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

  /// TODO add comment
  Value lowerExpr(ArrayRef<int64_t> coeffs, int64_t c) {
    Value sum = builder.create<ConstantIndexOp>(loc, c);

    // TODO can we do a kind of zip().fold() ?
    for (unsigned i = 0, e = coeffs.size(); i < e; ++i) {
      Value coeff = builder.create<ConstantIndexOp>(loc, coeffs[i]);
      Value val = dimAndSymbolValues[i];

      Value prod = builder.create<MulIOp>(loc, coeff, val);
      sum = builder.create<AddIOp>(loc, sum, prod);
    }
    return sum;
  }

  Value lowerPresburgerConstraint(ArrayRef<int64_t> coeffs, int64_t c,
                                  CmpIPredicate pred) {
    assert(coeffs.size() == dimAndSymbolValues.size() &&
           "expect coefficients for every dim and symbol");

    Value sum = lowerExpr(coeffs, c);

    Value zeroConstant = builder.create<ConstantIndexOp>(loc, 0);
    Value cmp = builder.create<CmpIOp>(loc, pred, sum, zeroConstant);
    return cmp;
  }

  OpBuilder &builder;
  ValueRange dimAndSymbolValues;
  Location loc;
};

} // namespace

/// Presburger contains is replaced by runtime checks
///
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

    const PresburgerSet &set = setAttr.getValue();

    PresburgerTransformer t(rewriter, op.dimAndSyms(), loc);

    Value reduced = t.lowerPresburgerSet(set);
    rewriter.replaceOp(op, reduced);
    return success();
  }
};

/// Presburger apply is replaced by runtime checks.
///
/// The lowered code goes over the pieces and checks if the point is part of a
/// certain piece. If this is the case it evaluates the expression of the
/// according piece and returns the result.
///
/// Every contains check and evaluation is in a separate block that are
/// connected by conditional branches.
///
class PresburgerApplyLowering : public OpRewritePattern<ApplyOp> {
public:
  using OpRewritePattern<ApplyOp>::OpRewritePattern;
  using ExprType = PresburgerExpr::ExprType;

  LogicalResult matchAndRewrite(ApplyOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ExprOp exprDef = op.expr().getDefiningOp<ExprOp>();
    if (!exprDef)
      return failure();

    PresburgerExprAttr exprAttr =
        exprDef.getAttrOfType<PresburgerExprAttr>("expr");
    const PresburgerExpr &expr = exprAttr.getValue();

    PresburgerTransformer t(rewriter, op.dimAndSyms(), loc);

    assert(expr.getDomains().size() > 0 &&
           "expect a Presburger expression to have atleast one domain");

    Block *before = rewriter.getInsertionBlock();
    Block *after = rewriter.splitBlock(before, rewriter.getInsertionPoint());

    Region *region = before->getParent();
    Block *containsBlock = new Block();
    region->push_back(containsBlock);

    rewriter.setInsertionPointToEnd(before);
    rewriter.create<BranchOp>(loc, containsBlock);
    rewriter.setInsertionPointToStart(containsBlock);

    for (unsigned i = 0, e = expr.getDomains().size(); i < e; ++i) {
      Value condition = t.lowerPresburgerSet(expr.getDomains()[i]);

      // A block that contains the code of the expression evaluation
      Block *evalBlock = new Block();
      region->push_back(evalBlock);

      rewriter.setInsertionPointToStart(evalBlock);

      ExprType pieceExpr = expr.getExprs()[i];
      Value res = t.lowerExpr(pieceExpr.second, pieceExpr.first);
      rewriter.create<BranchOp>(loc, after, res);

      rewriter.setInsertionPointToEnd(containsBlock);

      if (i + 1 < e) {
        // As long as there are reminaing pieces we either jump to the apply
        // block or we check the next piece, depending on the evaluated
        // condition.
        containsBlock = new Block();
        region->push_back(containsBlock);

        rewriter.create<CondBranchOp>(loc, condition, evalBlock, containsBlock);

        rewriter.setInsertionPointToStart(containsBlock);
      } else {
        // If we are at the end, we either succeeded the check or we have to
        // return a default value
        // TODO define default value
        Value destOp = rewriter.create<ConstantIndexOp>(loc, 0);
        rewriter.create<CondBranchOp>(loc, condition, evalBlock, after, destOp);
      }
    }

    BlockArgument blockArgument =
        after->addArgument(IndexType::get(rewriter.getContext()));
    rewriter.setInsertionPoint(op);

    rewriter.replaceOp(op.getOperation(), blockArgument);

    return success();
  }
};

void mlir::populatePresburgerToStdConversionPatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  // clang-format off
  patterns.insert<
    PresburgerContainsLowering,
    PresburgerApplyLowering
    >(ctx);
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
    // This is somehow needed as otherwise new blocks will not be
    // considered legal.
    target.addLegalOp<FuncOp>();

    if (failed(applyPartialConversion(getOperation(), target, patterns)))
      signalPassFailure();
  }
};
} // namespace

/// Lowers Presburger operations to standard dialect
std::unique_ptr<Pass> mlir::createPresburgerToStandardPass() {
  return std::make_unique<ConvertPresburgerToStandardPass>();
}
