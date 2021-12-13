#include "mlir/Analysis/PWAFunction.h"

using namespace mlir;

void PWAFunction::dump() const {
  for (unsigned i = 0; i < value.size(); ++i) {
    domain[i].dump();
    llvm::errs() << "\n";
    for (unsigned j = 0; j < value[i].size(); ++j) {
      llvm::errs() << "a" << j << " = ";
      for (unsigned k = 0; k < value[i][j].size() - 1; ++k) {
        if (value[i][j][k] == 0)
          continue;
        llvm::errs() << value[i][j][k] << "x" << k << " + ";
      }
      llvm::errs() << value[i][j].back() << '\n';
    }
    llvm::errs() << '\n';
  }
}

