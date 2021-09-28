//===- Parser.cpp - MLIR Presburger Parser --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a parser for Presburger sets.
//
//===----------------------------------------------------------------------===//

#include "mlir/PresburgerParser.h"
#include "./Parser.h"
#include "mlir/Parser/AsmParserState.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace detail;

using llvm::MemoryBuffer;
using llvm::SMLoc;
using llvm::SourceMgr;
using llvm::StringMap;

/// Parse a PresburgerSet from a given StringRef
FailureOr<PresburgerSet> mlir::parsePresburgerSet(StringRef str,
                                                  MLIRContext *ctx) {
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(MemoryBuffer::getMemBuffer(str), SMLoc());

  SymbolState symbols;
  AsmParserState asmParserState;
  ParserState state(sourceMgr, ctx, symbols, &asmParserState);
  Parser parser(state);

  // This set will be overwritten
  PresburgerSet set = PresburgerSet::getUniverse(1, 1);
  if (parser.parsePresburgerSet(set))
    return failure();

  if (!parser.getToken().is(Token::eof))
    return failure();

  return set;
}
