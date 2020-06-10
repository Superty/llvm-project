#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Dialect/Presburger/Attributes.h"
#include "mlir/Dialect/Presburger/PresburgerOps.h"
#include "mlir/Dialect/Presburger/Set.h"
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
  return rewriter.create<SetOp>(op->getLoc(), type, newAttr);
}

static SetOp intersectSets(PatternRewriter &rewriter, Operation *op,
                           PresburgerSetAttr attr1, PresburgerSetAttr attr2) {
  PresburgerSet ps(attr1.getValue());
  ps.intersectSet(attr2.getValue());

  PresburgerSetType type = PresburgerSetType::get(
      rewriter.getContext(), ps.getNumDims(), ps.getNumSyms());

  PresburgerSetAttr newAttr = PresburgerSetAttr::get(type, ps);
  return rewriter.create<SetOp>(op->getLoc(), type, newAttr);
}

static SetOp subtractSets(PatternRewriter &rewriter, Operation *op,
                          PresburgerSetAttr attr1, PresburgerSetAttr attr2) {
  PresburgerSet ps(attr1.getValue());
  ps.subtract(attr2.getValue());

  PresburgerSetType type = PresburgerSetType::get(
      rewriter.getContext(), ps.getNumDims(), ps.getNumSyms());

  PresburgerSetAttr newAttr = PresburgerSetAttr::get(type, ps);
  return rewriter.create<SetOp>(op->getLoc(), type, newAttr);
}

static SetOp complementSet(PatternRewriter &rewriter, Operation *op,
                           PresburgerSetAttr attr) {
  PresburgerSet ps = PresburgerSet::complement(attr.getValue());

  PresburgerSetType type = PresburgerSetType::get(
      rewriter.getContext(), ps.getNumDims(), ps.getNumSyms());

  PresburgerSetAttr newAttr = PresburgerSetAttr::get(type, ps);
  return rewriter.create<SetOp>(op->getLoc(), type, newAttr);
}

static ConstantOp areEqualSets(PatternRewriter &rewriter, Operation *op,
                               PresburgerSetAttr attr1,
                               PresburgerSetAttr attr2) {
  bool eq = PresburgerSet::equal(attr1.getValue(), attr2.getValue());

  IntegerType type = rewriter.getI1Type();
  IntegerAttr attr = IntegerAttr::get(type, eq);

  return rewriter.create<ConstantOp>(op->getLoc(), type, attr);
}

namespace {

#include "mlir/Dialect/Presburger/Transforms/Transforms.cpp.inc"

} // end anonymous namespace

// TODO should we really register the transforms as canonicalization passes?
void IntersectOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<FoldIntersectPattern>(context);
}

void UnionOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context) {
  results.insert<FoldUnionPattern>(context);
}

void SubtractOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                             MLIRContext *context) {
  results.insert<FoldSubtractPattern>(context);
}

void ComplementOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<FoldComplementPattern>(context);
}

void EqualOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context) {
  results.insert<FoldEqualPattern>(context);
}
