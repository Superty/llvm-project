#ifndef PRESBURGER_TYPES_H
#define PRESBURGER_TYPES_H

#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace presburger {
namespace detail {
struct PresburgerSetTypeStorage;
struct PresburgerExprTypeStorage;
} // namespace detail

namespace PresburgerTypes {
enum Kinds {
  Set = Type::Kind::FIRST_PRIVATE_EXPERIMENTAL_3_TYPE,
  Expr,
};
} // namespace PresburgerTypes

class PresburgerSetType
    : public Type::TypeBase<PresburgerSetType, Type,
                            detail::PresburgerSetTypeStorage> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == PresburgerTypes::Set; }

  static PresburgerSetType get(MLIRContext *context, unsigned dimCount,
                               unsigned symbolCount);

  static llvm::StringRef getKeyword() { return "set"; }

  unsigned getDimCount() const;
  unsigned getSymbolCount() const;
};

class PresburgerExprType
    : public Type::TypeBase<PresburgerExprType, Type,
                            detail::PresburgerExprTypeStorage> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == PresburgerTypes::Expr; }

  static PresburgerExprType get(MLIRContext *context, unsigned dimCount,
                                unsigned symbolCount);

  static llvm::StringRef getKeyword() { return "expr"; }

  unsigned getDimCount() const;
  unsigned getSymbolCount() const;
};

} // namespace presburger
} // namespace mlir

#endif // PRESBURGER_TYPES_H
