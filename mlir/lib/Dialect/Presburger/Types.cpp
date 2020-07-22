#include "mlir/Dialect/Presburger/Types.h"

namespace mlir {
namespace presburger {
namespace detail {

struct PresburgerSetTypeStorage : public TypeStorage {
  PresburgerSetTypeStorage(unsigned dimCount, unsigned symbolCount)
      : dimCount(dimCount), symbolCount(symbolCount) {}

  using KeyTy = std::pair<unsigned, unsigned>;

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(dimCount, symbolCount);
  }

  static KeyTy getKey(unsigned dimCount, unsigned symbolCount) {
    return KeyTy(dimCount, symbolCount);
  }

  static PresburgerSetTypeStorage *construct(TypeStorageAllocator &allocator,
                                             const KeyTy &key) {
    return new (allocator.allocate<PresburgerSetTypeStorage>())
        PresburgerSetTypeStorage(key.first, key.second);
  }

  unsigned dimCount, symbolCount;
};

struct PresburgerPwExprTypeStorage : public TypeStorage {
  PresburgerPwExprTypeStorage(unsigned dimCount, unsigned symbolCount)
      : dimCount(dimCount), symbolCount(symbolCount) {}

  using KeyTy = std::pair<unsigned, unsigned>;

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(dimCount, symbolCount);
  }

  static KeyTy getKey(unsigned dimCount, unsigned symbolCount) {
    return KeyTy(dimCount, symbolCount);
  }

  static PresburgerPwExprTypeStorage *construct(TypeStorageAllocator &allocator,
                                                const KeyTy &key) {
    return new (allocator.allocate<PresburgerPwExprTypeStorage>())
        PresburgerPwExprTypeStorage(key.first, key.second);
  }

  unsigned dimCount, symbolCount;
};
} // namespace detail
} // namespace presburger
} // namespace mlir

using namespace mlir;
using namespace mlir::presburger;

PresburgerSetType PresburgerSetType::get(MLIRContext *context,
                                         unsigned dimCount,
                                         unsigned symbolCount) {
  return Base::get(context, PresburgerTypes::Set, dimCount, symbolCount);
}

unsigned PresburgerSetType::getDimCount() const { return getImpl()->dimCount; }

unsigned PresburgerSetType::getSymbolCount() const {
  return getImpl()->symbolCount;
}

PresburgerPwExprType PresburgerPwExprType::get(MLIRContext *context,
                                               unsigned dimCount,
                                               unsigned symbolCount) {
  return Base::get(context, PresburgerTypes::PwExpr, dimCount, symbolCount);
}

unsigned PresburgerPwExprType::getDimCount() const {
  return getImpl()->dimCount;
}

unsigned PresburgerPwExprType::getSymbolCount() const {
  return getImpl()->symbolCount;
}
