#include "PassDetail.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Presburger/Coalesce.h"
#include "mlir/Analysis/Presburger/Set.h"
#include "mlir/Dialect/Presburger/Attributes.h"
#include "mlir/Dialect/Presburger/Passes.h"
#include "mlir/Dialect/Presburger/PresburgerOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::presburger;

static SetOp unionSets(PatternRewriter &rewriter, Operation *op,
                       PresburgerSetAttr attr1, PresburgerSetAttr attr2) {
  PresburgerSet ps(attr1.getValue());
  ps.unionSet(attr2.getValue());

  PresburgerSetType type = PresburgerSetType::get(
      rewriter.getContext(), ps.getNumDims(), ps.getNumSyms());

  PresburgerSetAttr newAttr = PresburgerSetAttr::get(type, ps);
  ps.dumpISL();
  return rewriter.create<SetOp>(op->getLoc(), type, newAttr);
}

static SetOp intersectSets(PatternRewriter &rewriter, Operation *op,
                           PresburgerSetAttr attr1, PresburgerSetAttr attr2) {
  PresburgerSet ps(attr1.getValue());
  ps.intersectSet(attr2.getValue());

  PresburgerSetType type = PresburgerSetType::get(
      rewriter.getContext(), ps.getNumDims(), ps.getNumSyms());

  PresburgerSetAttr newAttr = PresburgerSetAttr::get(type, ps);
  ps.dumpISL();
  return rewriter.create<SetOp>(op->getLoc(), type, newAttr);
}

static SetOp subtractSets(PatternRewriter &rewriter, Operation *op,
                          PresburgerSetAttr attr1, PresburgerSetAttr attr2) {
  PresburgerSet ps(attr1.getValue());
  ps.subtract(attr2.getValue());

  PresburgerSetType type = PresburgerSetType::get(
      rewriter.getContext(), ps.getNumDims(), ps.getNumSyms());

  PresburgerSetAttr newAttr = PresburgerSetAttr::get(type, ps);
  ps.dumpISL();
  return rewriter.create<SetOp>(op->getLoc(), type, newAttr);
}

static SetOp coalesceSet(PatternRewriter &rewriter, Operation *op,
                         PresburgerSetAttr attr) {
  // TODO: change Namespace of coalesce
  PresburgerSet in = attr.getValue();
  PresburgerSet ps = coalesce(in);

  PresburgerSetType type = PresburgerSetType::get(
      rewriter.getContext(), ps.getNumDims(), ps.getNumSyms());

  PresburgerSetAttr newAttr = PresburgerSetAttr::get(type, ps);
  ps.dumpISL();
  return rewriter.create<SetOp>(op->getLoc(), type, newAttr);
}

static SetOp eliminateExistentialsSet(PatternRewriter &rewriter, Operation *op,
                                      PresburgerSetAttr attr) {
  // TODO: change Namespace of coalesce
  PresburgerSet in = attr.getValue();
  PresburgerSet ps = PresburgerSet::eliminateExistentials(in);

  PresburgerSetType type = PresburgerSetType::get(
      rewriter.getContext(), ps.getNumDims(), ps.getNumSyms());

  PresburgerSetAttr newAttr = PresburgerSetAttr::get(type, ps);
  ps.dumpISL();
  return rewriter.create<SetOp>(op->getLoc(), type, newAttr);
}

static SetOp complementSet(PatternRewriter &rewriter, Operation *op,
                           PresburgerSetAttr attr) {
  PresburgerSet ps = PresburgerSet::complement(attr.getValue());

  PresburgerSetType type = PresburgerSetType::get(
      rewriter.getContext(), ps.getNumDims(), ps.getNumSyms());

  PresburgerSetAttr newAttr = PresburgerSetAttr::get(type, ps);
  ps.dumpISL();
  return rewriter.create<SetOp>(op->getLoc(), type, newAttr);
}

static ConstantOp areEqualSets(PatternRewriter &rewriter, Operation *op,
                               PresburgerSetAttr attr1,
                               PresburgerSetAttr attr2) {
  bool eq = PresburgerSet::equal(attr1.getValue(), attr2.getValue());

  IntegerType type = rewriter.getI1Type();
  IntegerAttr attr = IntegerAttr::get(type, eq);

  llvm::errs() << eq << '\n';
  return rewriter.create<ConstantOp>(op->getLoc(), type, attr);
}

static ConstantOp emptySet(PatternRewriter &rewriter, Operation *op,
                           PresburgerSetAttr attr) {
  PresburgerSet ps = attr.getValue();
  bool empty = ps.isIntegerEmpty();

  IntegerType type = rewriter.getI1Type();
  IntegerAttr iAttr = IntegerAttr::get(type, empty);

  return rewriter.create<ConstantOp>(op->getLoc(), type, iAttr);
}
namespace {

#include "mlir/Dialect/Presburger/Transforms/EvaluationPatterns.cpp.inc"

} // end anonymous namespace

void mlir::populatePresburgerEvaluatePatterns(
    OwningRewritePatternList &patterns, MLIRContext *ctx) {
  // clang-format off
  patterns.insert<
    FoldIntersectPattern,
    FoldUnionPattern,
    FoldSubtractPattern,
    FoldCoalescePattern,
    FoldEmptyPattern,
    FoldEliminateExPattern,
    FoldComplementPattern,
    FoldEqualPattern
    >(ctx);
  // clang-format on
}

struct PresburgerEvaluatePass
    : public PresburgerEvaluateBase<PresburgerEvaluatePass> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    populatePresburgerEvaluatePatterns(patterns, &getContext());
    applyPatternsAndFoldGreedily(getFunction(), patterns);
  }
};

std::unique_ptr<OperationPass<FuncOp>> mlir::createPresburgerEvaluatePass() {
  return std::make_unique<PresburgerEvaluatePass>();
}
