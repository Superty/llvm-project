#ifndef PRESBURGER_ATTRIBUTES_H
#define PRESBURGER_ATTRIBUTES_H

#include "mlir/Analysis/Presburger/Expr.h"
#include "mlir/Analysis/Presburger/Set.h"
#include "mlir/Dialect/Presburger/Types.h"
#include "mlir/IR/Attributes.h"

namespace mlir {

namespace presburger {

namespace detail {
struct PresburgerSetAttributeStorage;
struct PresburgerExprAttributeStorage;
} // namespace detail

namespace PresburgerAttributes {
enum Kind {
  PresburgerSet = Attribute::FIRST_PRIVATE_EXPERIMENTAL_3_ATTR,
  PresburgerExpr
};

} // namespace PresburgerAttributes

class PresburgerSetAttr
    : public Attribute::AttrBase<PresburgerSetAttr, Attribute,
                                 detail::PresburgerSetAttributeStorage> {
public:
  using Base::Base;
  using ValueType = PresburgerSet;

  static PresburgerSetAttr get(PresburgerSetType t, PresburgerSet value);

  PresburgerSet getValue() const;

  static StringRef getKindName();

  static bool kindof(unsigned kind) {
    return kind == PresburgerAttributes::PresburgerSet;
  }
};

class PresburgerExprAttr
    : public Attribute::AttrBase<PresburgerExprAttr, Attribute,
                                 detail::PresburgerExprAttributeStorage> {
public:
  using Base::Base;
  using ValueType = PresburgerExpr;

  static PresburgerExprAttr get(PresburgerExprType t, PresburgerExpr value);

  static StringRef getKindName();

  PresburgerExpr getValue() const;

  static bool kindof(unsigned kind) {
    return kind == PresburgerAttributes::PresburgerExpr;
  }
};

} // namespace presburger
} // namespace mlir

#endif // PRESBURGER_ATTRIBUTES_H
