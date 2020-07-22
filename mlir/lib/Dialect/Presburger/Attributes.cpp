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
    return false; // PresburgerSet::equal(key.second, value);
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

} // namespace presburger
} // namespace mlir
