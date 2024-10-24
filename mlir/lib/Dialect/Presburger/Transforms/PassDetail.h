//===- PassDetail.h - Presburger Pass class details -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_PRESBURGER_TRANSFORMS_PASSDETAIL_H_
#define DIALECT_PRESBURGER_TRANSFORMS_PASSDETAIL_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_CLASSES
#include "mlir/Dialect/Presburger/Passes.h.inc"

} // end namespace mlir

#endif // DIALECT_PRESBURGER_TRANSFORMS_PASSDETAIL_H_
