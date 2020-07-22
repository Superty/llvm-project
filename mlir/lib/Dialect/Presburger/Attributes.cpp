#include "mlir/Dialect/Presburger/Attributes.h"
#include "mlir/Dialect/Presburger/Types.h"

namespace mlir {
namespace presburger {
namespace detail {

struct PresburgerSetAttributeStorage : public AttributeStorage {
  using KeyTy = std::pair<PresburgerSetType, PresburgerSet>;

  PresburgerSetAttributeStorage(Type t, PresburgerSet value)
      : AttributeStorage(t), value(value) {}

  bool operator==(const KeyTy &key) const {
    // TODO is this a good idea? it might be too expensive for the amount of
    // memory we can save with it
    return PresburgerSet::equal(key.second, value);
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return key.second.hash_value();
  }

  static PresburgerSetAttributeStorage *
  construct(AttributeStorageAllocator &allocator, KeyTy key) {
    return new (allocator.allocate<PresburgerSetAttributeStorage>())
        PresburgerSetAttributeStorage(std::get<0>(key), std::get<1>(key));
  }

  PresburgerSet value;
};

struct PresburgerPwExprAttributeStorage : public AttributeStorage {
  using KeyTy = std::pair<PresburgerPwExprType, PresburgerPwExpr>;

  PresburgerPwExprAttributeStorage(Type t, PresburgerPwExpr value)
      : AttributeStorage(t), value(value) {}

  bool operator==(const KeyTy &key) const {
    return false;
    // TODO
    // PresburgerSet::equal(key.second, value);
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return key.second.hash_value();
  }

  static PresburgerPwExprAttributeStorage *
  construct(AttributeStorageAllocator &allocator, KeyTy key) {
    return new (allocator.allocate<PresburgerPwExprAttributeStorage>())
        PresburgerPwExprAttributeStorage(std::get<0>(key), std::get<1>(key));
  }

  PresburgerPwExpr value;
};

} // namespace detail

//===----------------------------------------------------------------------===//
// PresburgerSetAttr
//===----------------------------------------------------------------------===//

PresburgerSetAttr PresburgerSetAttr::get(PresburgerSetType t,
                                         PresburgerSet value) {
  return Base::get(t.getContext(), PresburgerAttributes::PresburgerSet, t,
                   value);
}

PresburgerSet PresburgerSetAttr::getValue() const { return getImpl()->value; }

StringRef PresburgerSetAttr::getKindName() { return "set"; }

//===----------------------------------------------------------------------===//
// PresburgerPwExprAttr
//===----------------------------------------------------------------------===//

PresburgerPwExprAttr PresburgerPwExprAttr::get(PresburgerPwExprType t,
                                               PresburgerPwExpr value) {
  return Base::get(t.getContext(), PresburgerAttributes::PresburgerPwExpr, t,
                   value);
}

PresburgerPwExpr PresburgerPwExprAttr::getValue() const {
  return getImpl()->value;
}

StringRef PresburgerPwExprAttr::getKindName() { return "pwExpr"; }

} // namespace presburger
} // namespace mlir
