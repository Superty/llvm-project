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

struct PresburgerExprTypeStorage : public TypeStorage {
  PresburgerExprTypeStorage(unsigned dimCount, unsigned symbolCount)
      : dimCount(dimCount), symbolCount(symbolCount) {}

  using KeyTy = std::pair<unsigned, unsigned>;

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(dimCount, symbolCount);
  }

  static KeyTy getKey(unsigned dimCount, unsigned symbolCount) {
    return KeyTy(dimCount, symbolCount);
  }

  static PresburgerExprTypeStorage *construct(TypeStorageAllocator &allocator,
                                              const KeyTy &key) {
    return new (allocator.allocate<PresburgerExprTypeStorage>())
        PresburgerExprTypeStorage(key.first, key.second);
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

PresburgerExprType PresburgerExprType::get(MLIRContext *context,
                                           unsigned dimCount,
                                           unsigned symbolCount) {
  return Base::get(context, PresburgerTypes::Expr, dimCount, symbolCount);
}

unsigned PresburgerExprType::getDimCount() const { return getImpl()->dimCount; }

unsigned PresburgerExprType::getSymbolCount() const {
  return getImpl()->symbolCount;
}
