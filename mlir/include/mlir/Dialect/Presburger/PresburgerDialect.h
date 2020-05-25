#ifndef MLIR_DIALECT_Presburger_PresburgerDIALECT_H
#define MLIR_DIALECT_Presburger_PresburgerDIALECT_H

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace presburger {
namespace detail {} // namespace detail

class PresburgerDialect : public Dialect {
public:
  explicit PresburgerDialect(MLIRContext *context);

  /// Returns the prefix used in the textual IR to refer to Presburger
  /// operations
  static StringRef getDialectNamespace() { return "presburger"; }

  /// Parses a type registered to this dialect
  Attribute parseAttribute(DialectAsmParser &parser, Type attrT) const override;

  /// Print a type registered to this dialect
  void printAttribute(Attribute attr,
                      DialectAsmPrinter &printer) const override;

  Type parseType(DialectAsmParser &parser) const override;

  void printType(Type type, DialectAsmPrinter &printer) const override;
};

} // namespace presburger
} // namespace mlir

#endif // MLIR_DIALECT_Presburger_PresburgerDIALECT_H
