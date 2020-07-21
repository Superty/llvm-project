#ifndef PRESBURGER_ATTRIBUTES_H
#define PRESBURGER_ATTRIBUTES_H

#include "mlir/Analysis/Presburger/Set.h"
#include "mlir/Dialect/Presburger/Types.h"
#include "mlir/IR/Attributes.h"

namespace mlir {

namespace presburger {

namespace detail {
struct PresburgerSetAttributeStorage;
}

namespace PresburgerAttributes {
enum Kind {
  PresburgerSet = Attribute::FIRST_PRIVATE_EXPERIMENTAL_3_ATTR,
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

} // namespace presburger
} // namespace mlir

#endif // PRESBURGER_ATTRIBUTES_H
