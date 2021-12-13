#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"


namespace mlir {

struct PWAFunction {
  SmallVector<FlatAffineConstraints, 8> domain;
  SmallVector<SmallVector<SmallVector<int64_t, 8>, 8>, 8> value;

  void dump() const;
};

}
