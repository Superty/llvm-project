#include "mlir/Dialect/Presburger/PresburgerDialect.h"
#include "mlir/Dialect/Presburger/Attributes.h"
#include "mlir/Dialect/Presburger/Parser.h"
#include "mlir/Dialect/Presburger/PresburgerOps.h"
#include "mlir/Dialect/Presburger/Types.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace mlir::presburger;

//===----------------------------------------------------------------------===//
// Presburger Dialect
//===----------------------------------------------------------------------===//

PresburgerDialect::PresburgerDialect(mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addAttributes<PresburgerSetAttr, PresburgerExprAttr>();
  addTypes<PresburgerSetType, PresburgerExprType>();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Presburger/PresburgerOps.cpp.inc"
      >();
}

Attribute PresburgerDialect::parseAttribute(DialectAsmParser &parser,
                                            Type attrT) const {

  auto callback = [&parser](SMLoc loc, const Twine &msg) {
    return parser.emitError(loc, msg);
  };

  Parser p(parser.getFullSymbolSpec(), callback);

  // Parse the kind keyword first.
  StringRef attrKind;
  if (failed(p.parseKeyword(attrKind)))
    return {};

  if (attrKind == PresburgerSetAttr::getKindName()) {
    PresburgerParser setParser(p);
    PresburgerSet set;

    if (failed(setParser.parsePresburgerSet(set))) {
      return Attribute();
    }

    PresburgerSetType type = PresburgerSetType::get(
        getContext(), set.getNumDims(), set.getNumSyms());

    return PresburgerSetAttr::get(type, set);
  } else if (attrKind == PresburgerExprAttr::getKindName()) {
    PresburgerParser pwParser(p);
    PresburgerExpr pwExpr;

    if ((failed(pwParser.parsePresburgerExpr(pwExpr))))
      return Attribute();

    PresburgerExprType type = PresburgerExprType::get(
        getContext(), pwExpr.getNumDims(), pwExpr.getNumSyms());

    return PresburgerExprAttr::get(type, pwExpr);
  }
  parser.emitError(parser.getCurrentLocation(),
                   "unknown Presburger attribute kind: " + attrKind);
  return {};
}

void PresburgerDialect::printAttribute(Attribute attr,
                                       DialectAsmPrinter &printer) const {
  switch (attr.getKind()) {
  case PresburgerAttributes::PresburgerSet:
    printer << PresburgerSetAttr::getKindName();
    attr.cast<PresburgerSetAttr>().getValue().print(printer.getStream());
    break;
  case PresburgerAttributes::PresburgerExpr:
    printer << PresburgerExprAttr::getKindName();
    attr.cast<PresburgerExprAttr>().getValue().print(printer.getStream());
    break;
  default:
    llvm_unreachable("unknown PresburgerAttr kind");
  }
}

template <class PresburgerType>
Type parsePresburgerType(DialectAsmParser &parser, MLIRContext *context) {
  unsigned dimCount, symbolCount;
  if (parser.parseLess() || parser.parseInteger<unsigned>(dimCount) ||
      parser.parseComma() || parser.parseInteger<unsigned>(symbolCount) ||
      parser.parseGreater()) {
    return Type();
  }
  return PresburgerType::get(context, dimCount, symbolCount);
}

Type PresburgerDialect::parseType(DialectAsmParser &parser) const {
  llvm::SMLoc loc = parser.getCurrentLocation();
  llvm::StringRef typeKeyword;
  if (parser.parseKeyword(&typeKeyword))
    return Type();

  if (typeKeyword == PresburgerSetType::getKeyword())
    return parsePresburgerType<PresburgerSetType>(parser, getContext());

  if (typeKeyword == PresburgerExprType::getKeyword())
    return parsePresburgerType<PresburgerExprType>(parser, getContext());

  parser.emitError(loc, "unknown Presburger type");
  return Type();
}

void printPresburgerSetType(PresburgerSetType set, DialectAsmPrinter &printer) {
  printer << PresburgerSetType::getKeyword();
  printer << "<" << set.getDimCount() << "," << set.getSymbolCount() << ">";
}

void printPresburgerExprType(PresburgerExprType set,
                             DialectAsmPrinter &printer) {
  printer << PresburgerExprType::getKeyword();
  printer << "<" << set.getDimCount() << "," << set.getSymbolCount() << ">";
}

void PresburgerDialect::printType(Type type, DialectAsmPrinter &printer) const {
  switch (type.getKind()) {
  case PresburgerTypes::Set: {
    printPresburgerSetType(type.cast<PresburgerSetType>(), printer);
    break;
  }
  case PresburgerTypes::Expr: {
    printPresburgerExprType(type.cast<PresburgerExprType>(), printer);
    break;
  }
  default:
    llvm_unreachable("unknown PresburgerType kind");
  }
}
