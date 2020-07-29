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

struct PresburgerExprAttributeStorage : public AttributeStorage {
  using KeyTy = std::pair<PresburgerExprType, PresburgerExpr>;

  PresburgerExprAttributeStorage(Type t, PresburgerExpr value)
      : AttributeStorage(t), value(value) {}

  bool operator==(const KeyTy &key) const {
    return false;
    // TODO
    // PresburgerSet::equal(key.second, value);
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return key.second.hash_value();
  }

  static PresburgerExprAttributeStorage *
  construct(AttributeStorageAllocator &allocator, KeyTy key) {
    return new (allocator.allocate<PresburgerExprAttributeStorage>())
        PresburgerExprAttributeStorage(std::get<0>(key), std::get<1>(key));
  }

  PresburgerExpr value;
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
// PresburgerExprAttr
//===----------------------------------------------------------------------===//

PresburgerExprAttr PresburgerExprAttr::get(PresburgerExprType t,
                                           PresburgerExpr value) {
  return Base::get(t.getContext(), PresburgerAttributes::PresburgerExpr, t,
                   value);
}

PresburgerExpr PresburgerExprAttr::getValue() const { return getImpl()->value; }

StringRef PresburgerExprAttr::getKindName() { return "expr"; }

} // namespace presburger
} // namespace mlir
