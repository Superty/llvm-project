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
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace detail;

using llvm::MemoryBuffer;
using llvm::SMLoc;
using llvm::SourceMgr;
using llvm::StringMap;

namespace {

enum class ConstraintKind { LE, GE, EQ };

/// This is a specialized parser for Presburger structures (Presburger set).
class PresburgerParser : public Parser {
public:
  PresburgerParser(ParserState &state)
      : Parser(state), numDims(0), numSymbols(0){};

  /// Parse a Presburger set into set
  ParseResult parsePresburgerSet(PresburgerSet &set);

private:
  ParseResult parseFlatAffineConstraints(FlatAffineConstraints &fac);

  ParseResult parseIdList(unsigned &numVar, Token::Kind rightToken);
  ParseResult parseDimIdList();
  ParseResult parseSymbolIdList();
  ParseResult parseDimAndOptionalSymbolIdList();

  ParseResult parsePresburgerSetConstraints(PresburgerSet &set);

  ParseResult parseConstraint(FlatAffineConstraints &fac);
  ParseResult parseSum(SmallVector<int64_t, 8> &coefs);
  ParseResult parseTerm(SmallVector<int64_t, 8> &coefs,
                        bool is_negated = false);
  ParseResult parseVariable(StringRef &var);

  /// Mapping from names to ids
  unsigned numDims;
  unsigned numSymbols;
  StringMap<unsigned> dimAndSymNameToIndex;
};

//===----------------------------------------------------------------------===//
// PresburgerParser
//===----------------------------------------------------------------------===//

/// Parse a Presburger set.
///
///  pb-set        ::= dim-and-symbol-use-list `:` pb-or-expr
///  pb-or-expr    ::= pb-and-expr (`or` pb-and-expr)*
///  pb-and-expr   ::= `(` `)`
///                  | `(` pb-constraint (`and` pb-constraint)* `)`
///  pb-constraint ::= pb-sum (`>=` | `=` | `<=`) pb-sum
///  pb-sum        ::= pb-term ((`+` | `-`) pb-term)*
///  pb-term       ::= `-`? integer-literal? bare-id
///                  | `-`? integer-literal
///
ParseResult PresburgerParser::parsePresburgerSet(PresburgerSet &set) {
  if (parseDimAndOptionalSymbolIdList())
    return failure();

  if (parseToken(Token::colon, "expected ':'"))
    return failure();

  set = PresburgerSet::getEmptySet(numDims, numSymbols);

  if (parsePresburgerSetConstraints(set))
    return failure();

  // checks that we are at the end of the string
  if (!getToken().is(Token::eof))
    return emitError("expected to be at the end of the set");
  return success();
}

/// Helper to parse a list of identifiers
ParseResult PresburgerParser::parseIdList(unsigned &numVar,
                                          Token::Kind rightToken) {
  auto parseElt = [&]() -> ParseResult {
    StringRef name;
    if (parseVariable(name))
      return failure();

    auto it = dimAndSymNameToIndex.find(name);
    if (it != dimAndSymNameToIndex.end())
      return emitError("repeated variable names in the tuple are not allowed");

    dimAndSymNameToIndex.insert_or_assign(name, numVar++);
    return success();
  };
  return parseCommaSeparatedListUntil(rightToken, parseElt);
}

/// Parse the list of dimensional identifiers to a Presburger set
ParseResult PresburgerParser::parseDimIdList() {
  if (parseToken(Token::l_paren,
                 "expected '(' at start of dimensional identifiers list")) {
    return failure();
  }

  return parseIdList(numDims, Token::r_paren);
}

/// Parse the list of symbolic identifiers to an affine map.
ParseResult PresburgerParser::parseSymbolIdList() {
  consumeToken(Token::l_square);
  return parseIdList(numSymbols, Token::r_square);
}

/// Parse the list of symbolic identifiers
///
/// dim-and-symbol-use-list is defined in the AffineParser
///
ParseResult PresburgerParser::parseDimAndOptionalSymbolIdList() {
  if (parseDimIdList()) {
    return failure();
  }
  if (!getToken().is(Token::l_square)) {
    numSymbols = 0;
    return success();
  }
  return parseSymbolIdList();
}

/// Parse an or expression.
///
///  pb-or-expr ::= pb-and-expr (`or` pb-and-expr)*
///
ParseResult
PresburgerParser::parsePresburgerSetConstraints(PresburgerSet &set) {
  FlatAffineConstraints fac;
  if (parseFlatAffineConstraints(fac))
    return failure();

  set.unionFACInPlace(fac);

  while (consumeIf(Token::kw_or)) {
    if (parseFlatAffineConstraints(fac))
      return failure();
    set.unionFACInPlace(fac);
  }

  return success();
}

/// Parse an and expression.
///
///  pb-and-expr ::= `(` `)`
///                | `(` pb-constraint (`and` pb-constraint)* `)`
///
ParseResult
PresburgerParser::parseFlatAffineConstraints(FlatAffineConstraints &fac) {
  if (parseToken(Token::l_paren, "expected '('"))
    return failure();

  fac = FlatAffineConstraints(numDims, numSymbols);

  if (consumeIf(Token::r_paren)) {
    return success();
  }

  if (parseConstraint(fac))
    return failure();

  while (consumeIf(Token::kw_and)) {
    if (parseConstraint(fac))
      return failure();
  }

  if (parseToken(Token::r_paren, "expected ')'"))
    return failure();

  return success();
}

/// Parse a Presburger constraint
///
///  pb-constraint ::= pb-expr (`>=` | `=` | `<=`) pb-expr
///
ParseResult PresburgerParser::parseConstraint(FlatAffineConstraints &fac) {
  // space for constant
  unsigned size = dimAndSymNameToIndex.size() + 1;
  SmallVector<int64_t, 8> left(size, 0);
  SmallVector<int64_t, 8> right(size, 0);

  if (parseSum(left))
    return failure();

  ConstraintKind kind;
  switch (getToken().getKind()) {
  case Token::greater:
    consumeToken(Token::greater);
    if (consumeIf(Token::equal)) {
      kind = ConstraintKind::GE;
      break;
    }
    return emitError("strict inequalities are not supported");
  case Token::equal:
    consumeToken(Token::equal);
    kind = ConstraintKind::EQ;
    break;
  case Token::less:
    consumeToken(Token::less);
    if (consumeIf(Token::equal)) {
      kind = ConstraintKind::LE;
      break;
    }
    return emitError("strict inequalities are not supported");
  case Token::bang:
    consumeToken(Token::bang);
    if (consumeIf(Token::equal)) {
      return emitError("!= constraints are not supported");
    }
    [[clang::fallthrough]];
  default:
    return emitError("expected comparison operator");
  }

  if (parseSum(right))
    return failure();

  // TODO move to separate function, check if we can reuse one of the vectors.
  // TODO can we use std::move?
  SmallVector<int64_t, 8> coeffs;

  switch (kind) {
  case ConstraintKind::LE:
    for (unsigned i = 0; i < left.size(); i++)
      coeffs.push_back(right[i] - left[i]);
    break;

  case ConstraintKind::GE:
  case ConstraintKind::EQ:
    for (unsigned i = 0; i < left.size(); i++)
      coeffs.push_back(left[i] - right[i]);
    break;
  }

  if (kind == ConstraintKind::EQ)
    fac.addEquality(coeffs);
  else
    fac.addInequality(coeffs);

  return success();
}

/// Parse a Presburger sum.
///
///  pb-sum ::= pb-term ((`+` | `-`) pb-term)*
///
ParseResult PresburgerParser::parseSum(SmallVector<int64_t, 8> &coeffs) {
  if (parseTerm(coeffs))
    return failure();

  while (getToken().isAny(Token::plus, Token::minus)) {
    bool isMinus = getToken().is(Token::minus);
    consumeToken();

    if (parseTerm(coeffs, isMinus))
      return failure();
  }

  return success();
}

/// Parse a Presburger term.
///
///  pb-term ::= `-`? integer-literal? bare-id
///            | `-`? integer-literal
///
ParseResult PresburgerParser::parseTerm(SmallVector<int64_t, 8> &coeffs,
                                        bool isNegated) {
  if (consumeIf(Token::minus))
    isNegated = !isNegated;

  int64_t coeff = 1;
  bool intFound = false;

  if (parseOptionalInteger(coeff).hasValue()) {
    intFound = true;
  }
  if (isNegated) {
    coeff = -coeff;
  }

  if (intFound && getToken().isNot(Token::bare_identifier)) {
    if (!intFound)
      return emitError("expected non empty term");
    coeffs[coeffs.size() - 1] += coeff;
    return success();
  }

  StringRef identifier;
  if (parseVariable(identifier))
    return failure();

  // add term to coeffs
  auto it = dimAndSymNameToIndex.find(identifier);
  if (it != dimAndSymNameToIndex.end()) {
    coeffs[it->second] += coeff;
    return success();
  }

  return emitError("encountered unknown variable name: " + identifier);
}

/// Parse a bare-identifier
ParseResult PresburgerParser::parseVariable(StringRef &var) {
  if (getToken().isNot(Token::bare_identifier))
    return failure();
  var = getTokenSpelling();
  consumeToken();
  return success();
}
} // namespace

/// Parse a PresburgerSet from a given StringRef
FailureOr<PresburgerSet> mlir::parsePresburgerSet(StringRef str,
                                                  MLIRContext *ctx) {
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(MemoryBuffer::getMemBuffer(str), SMLoc());

  SymbolState symbols;
  ParserState state(sourceMgr, ctx, symbols);
  PresburgerParser parser(state);

  // This set will be overwritten
  PresburgerSet set = PresburgerSet::getUniverse(1, 1);
  if (parser.parsePresburgerSet(set))
    return failure();
  return set;
}
