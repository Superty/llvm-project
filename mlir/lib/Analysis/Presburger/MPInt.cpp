//===- MPInt.cpp - MLIR MPInt Class ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/MPInt.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace presburger;

llvm::hash_code mlir::presburger::hash_value(const MPInt &x) {
  return std::visit([](auto &&x) {
    return llvm::hash_value(x);
  }, x.val);
}

/// ---------------------------------------------------------------------------
/// Printing.
/// ---------------------------------------------------------------------------
llvm::raw_ostream &MPInt::print(llvm::raw_ostream &os) const {
  return std::visit([&](auto &&x) -> llvm::raw_ostream& {
    return os << x;
  }, val);
}

void MPInt::dump() const { print(llvm::errs()); }

llvm::raw_ostream &mlir::presburger::operator<<(llvm::raw_ostream &os,
                                                const MPInt &x) {
  x.print(os);
  return os;
}

