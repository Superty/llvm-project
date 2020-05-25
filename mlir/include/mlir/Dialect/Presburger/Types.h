#ifndef PRESBURGER_TYPES_H
#define PRESBURGER_TYPES_H

#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace presburger {
namespace detail {
struct PresburgerSetTypeStorage;
}

namespace PresburgerTypes {
enum Kinds {
  Set = Type::Kind::FIRST_PRIVATE_EXPERIMENTAL_3_TYPE,
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

} // namespace presburger
} // namespace mlir

#endif // PRESBURGER_TYPES_H
