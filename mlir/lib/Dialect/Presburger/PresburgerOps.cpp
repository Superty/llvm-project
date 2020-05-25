#include "mlir/Dialect/Presburger/PresburgerOps.h"
#include "mlir/Dialect/Presburger/Attributes.h"
#include "mlir/Dialect/Presburger/PresburgerDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::presburger;

//===---------------------------------------------------------------------===//
// Presburger Operations
//===---------------------------------------------------------------------===//

// Presburger set

static void print(OpAsmPrinter &printer, SetOp op) {
  printer << "presburger.set " << op.setAttr();
}

static ParseResult parseSetOp(OpAsmParser &parser, OperationState &result) {
  PresburgerSetAttr set;

  if (parser.parseAttribute(set, "set", result.attributes))
    return failure();

  // TODO Currently we inherit the type from the PresburgerSetAttr, I'm
  // not sure if this is desirable.
  Type outType = set.getType();

  parser.addTypeToList(outType, result.types);
  return success();
}

static LogicalResult verify(SetOp op) {
  PresburgerSetType s = op.setAttr().getType().cast<PresburgerSetType>();
  PresburgerSetType res = op.res().getType().cast<PresburgerSetType>();

  if (s.getDimCount() != res.getDimCount() ||
      s.getSymbolCount() != res.getSymbolCount()) {
    op.emitError(
        "expects attribute and result to be of equal dim and symbol counts");
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Presburger ops
//===----------------------------------------------------------------------===//

static ParseResult parseBinSetOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 2> obsOperands;

  if (parser.parseOperandList(obsOperands, OpAsmParser::Delimiter::None))
    return failure();

  Type outType;
  if (parser.parseColon() || parser.parseType(outType))
    return failure();

  parser.addTypeToList(outType, result.types);

  if (parser.resolveOperands(obsOperands, outType, result.operands))
    return failure();

  return success();
}

// TODO  Discuss if we want the types or not.

static void print(OpAsmPrinter &printer, UnionOp op) {
  printer << "presburger.union ";
  printer.printOperand(op.set1());
  printer << ", ";
  printer.printOperand(op.set2());
  printer << " : ";
  printer.printType(op.getType());
}

// intersect

static void print(OpAsmPrinter &printer, IntersectOp op) {
  printer << "presburger.intersect ";
  printer.printOperand(op.set1());
  printer << ", ";
  printer.printOperand(op.set2());
  printer << " : ";
  printer.printType(op.getType());
}

namespace mlir {
namespace presburger {
#define GET_OP_CLASSES
#include "mlir/Dialect/Presburger/PresburgerOps.cpp.inc"
} // namespace presburger
} // namespace mlir
