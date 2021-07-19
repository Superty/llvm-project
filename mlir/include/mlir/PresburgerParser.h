//===- Parser.h - MLIR Presburger Parser ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the interface to the Presburger parser.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PRESBURGER_PARSER_H
#define MLIR_PRESBURGER_PARSER_H

#include "mlir/Analysis/PresburgerSet.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {

/// This parses a single PresburgerSet to an MLIR context if it was valid. If
/// not, an error message is emitted.
FailureOr<PresburgerSet> parsePresburgerSet(StringRef str, MLIRContext *ctx);

} // namespace mlir

#endif // MLIR_PRESBURGER_PARSER_H
