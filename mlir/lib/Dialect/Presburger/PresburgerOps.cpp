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

// Presburger pw expr

static void print(OpAsmPrinter &printer, PwExprOp op) {
  printer << "presburger.pwExpr " << op.pwExprAttr();
}

static ParseResult parsePwExprOp(OpAsmParser &parser, OperationState &result) {
  PresburgerPwExprAttr pwExpr;

  if (parser.parseAttribute(pwExpr, "pwExpr", result.attributes))
    return failure();

  // TODO Currently we inherit the type from the PresburgerPwExprAttr, I'm
  // not sure if this is desirable.
  Type outType = pwExpr.getType();

  parser.addTypeToList(outType, result.types);
  return success();
}

static LogicalResult verify(PwExprOp op) {
  // TODO do we need something here?
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

/// verifies that the sets definition is actually reachable
template <typename OpTy>
static LogicalResult verifyLocality(Value set, OpTy op) {
  Operation *defOp = set.getDefiningOp();
  if (!defOp)
    return op.emitError("expect local set definitions");

  if (!defOp->hasTrait<OpTrait::ProducesPresburgerSet>())
    return op.emitError("expect operand to have trait ProducesPresburgerSet");

  return success();
}

/// Verifies that both set operands are defined locally
template <typename OpTy>
static LogicalResult verifyBinSetOp(OpTy op) {
  if (failed(verifyLocality(op.set1(), op)) ||
      failed(verifyLocality(op.set2(), op)))
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

// subtract

static void print(OpAsmPrinter &printer, SubtractOp op) {
  printer << "presburger.subtract ";
  printer.printOperand(op.set1());
  printer << ", ";
  printer.printOperand(op.set2());
  printer << " : ";
  printer.printType(op.getType());
}

// complement

static ParseResult parseComplementOp(OpAsmParser &parser,
                                     OperationState &result) {
  OpAsmParser::OperandType op;

  if (parser.parseOperand(op))
    return failure();

  Type outType;
  if (parser.parseColon() || parser.parseType(outType))
    return failure();

  parser.addTypeToList(outType, result.types);

  if (parser.resolveOperands(op, outType, result.operands))
    return failure();

  return success();
}

static void print(OpAsmPrinter &printer, ComplementOp op) {
  printer << "presburger.complement ";
  printer.printOperand(op.set());
  printer << " : ";
  printer.printType(op.getType());
}

static LogicalResult verify(ComplementOp op) {
  return verifyLocality(op.set(), op);
}

// equal

static ParseResult parseEqualOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 2> obsOperands;

  if (parser.parseOperandList(obsOperands, OpAsmParser::Delimiter::None))
    return failure();

  Type outType = parser.getBuilder().getI1Type();
  parser.addTypeToList(outType, result.types);

  SmallVector<Type, 2> types;

  if (parser.parseColonTypeList(types))
    return failure();

  if (parser.resolveOperands(obsOperands, types, parser.getCurrentLocation(),
                             result.operands))
    return failure();

  return success();
}

// Equal

// TODO discuss if we want to print this in that fashion. Especialy discuss the
// type stuff
static void print(OpAsmPrinter &printer, EqualOp op) {
  printer << "presburger.equal ";
  printer.printOperand(op.set1());
  printer << ", ";
  printer.printOperand(op.set2());
  printer << " : ";
  printer.printType(op.set1().getType());
  printer << ", ";
  printer.printType(op.set2().getType());
}

// Contains

static ParseResult parseContainsOp(OpAsmParser &parser,
                                   OperationState &result) {
  unsigned numDims;
  if (parseDimAndSymbolList(parser, result.operands, numDims) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  unsigned numSyms = result.operands.size() - numDims;

  OpAsmParser::OperandType setOp;
  if (parser.parseOperand(setOp))
    return failure();

  PresburgerSetType setType = PresburgerSetType::get(
      parser.getBuilder().getContext(), numDims, numSyms);

  if (parser.resolveOperand(setOp, setType, result.operands))
    return failure();

  Type outType = parser.getBuilder().getI1Type();
  parser.addTypeToList(outType, result.types);

  return success();
}

static void print(OpAsmPrinter &printer, ContainsOp op) {
  PresburgerSetType setType = op.set().getType().cast<PresburgerSetType>();
  printer << "presburger.contains ";
  printDimAndSymbolList(op.operand_begin(),
                        op.operand_begin() + setType.getDimCount() +
                            setType.getSymbolCount(),
                        setType.getDimCount(), printer);
  printer << " ";
  printer.printOperand(op.set());
}

static LogicalResult verify(ContainsOp op) {
  return verifyLocality(op.set(), op);
}

namespace mlir {
namespace presburger {
#define GET_OP_CLASSES
#include "mlir/Dialect/Presburger/PresburgerOps.cpp.inc"
} // namespace presburger
} // namespace mlir
