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
  ParseResult parseSum(FlatAffineConstraints &fac,
                       SmallVector<int64_t, 8> &coefs);
  ParseResult parseTerm(FlatAffineConstraints &fac,
                        SmallVector<int64_t, 8> &coefs,
                        bool is_negated = false);
  ParseResult parseDivision(FlatAffineConstraints &fac, int64_t &localId);
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
///  pb-term       ::= integer-literal? bare-id
///                  | `-` bare-id
///                  | integer-literal? pb-floor-div
///                  | `-` pb-floor-div
///                  | integer-literal
///  pb-floor      ::= floor(pb-sum `/` integer-literal)
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

namespace {
/// Addapts the existing coefficient array for newly added local Ids
/// This is done by making space for them and moving the value of the constant
/// to its new position
void makeSpaceForDivIds(SmallVector<int64_t, 8> &coeffs, size_t newIds) {
  int64_t prevConst = coeffs[coeffs.size() - 1];
  coeffs[coeffs.size() - 1] = 0;

  coeffs.append(newIds, 0);
  // set the constant in the new possition
  coeffs[coeffs.size() - 1] = prevConst;
}
} // namespace

/// Parse a Presburger constraint
///
///  pb-constraint ::= pb-expr (`>=` | `=` | `<=`) pb-expr
///
ParseResult PresburgerParser::parseConstraint(FlatAffineConstraints &fac) {
  // space for constant
  SmallVector<int64_t, 8> left(fac.getNumCols(), 0);

  if (parseSum(fac, left))
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
  default:
    return emitError("expected comparison operator");
  }

  SmallVector<int64_t, 8> right(fac.getNumCols(), 0);
  if (parseSum(fac, right))
    return failure();

  // check if the right side introduced new local ids
  if (left.size() < right.size())
    makeSpaceForDivIds(left, right.size() - left.size());

  SmallVector<int64_t, 8> coeffs;
  coeffs.reserve(left.size());

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
ParseResult PresburgerParser::parseSum(FlatAffineConstraints &fac,
                                       SmallVector<int64_t, 8> &coeffs) {
  if (parseTerm(fac, coeffs))
    return failure();

  while (getToken().isAny(Token::plus, Token::minus)) {
    bool isMinus = getToken().is(Token::minus);
    consumeToken();

    if (parseTerm(fac, coeffs, isMinus))
      return failure();
  }

  return success();
}

/// Parse a Presburger term.
///
///  pb-term ::= integer-literal? bare-id
///            | `-` bare-id
///            | integer-literal? pb-floor-div
///            | `-` pb-floor-div
///            | integer-literal
///
ParseResult PresburgerParser::parseTerm(FlatAffineConstraints &fac,
                                        SmallVector<int64_t, 8> &coeffs,
                                        bool isNegated) {
  if (consumeIf(Token::minus))
    isNegated = !isNegated;

  // needed as parseOptionalInteger would consume this even on failure.
  // TODO check if this bug is still present
  // TODO fix this as soon as parseOptionalInteger works as expected
  if (getToken().is(Token::minus))
    return emitError("expected integer literal or identifier");

  int64_t coeff = 1;
  // TODO how to handle other precisions?
  APInt parsedInt;
  bool intFound = parseOptionalInteger(parsedInt).hasValue();

  // TODO check what happens if input has too high precision
  if (intFound) {
    coeff = parsedInt.getZExtValue();
  }
  if (isNegated) {
    coeff = -coeff;
  }

  if (getToken().isNot(Token::bare_identifier) &&
      getToken().isNot(Token::kw_floor)) {
    if (!intFound)
      return emitError("expected non empty term");
    coeffs[coeffs.size() - 1] += coeff;
    return success();
  }

  if (getToken().is(Token::kw_floor)) {
    // Problem: a new division changes the dimensionality
    int64_t divId;
    if (parseDivision(fac, divId))
      return failure();

    makeSpaceForDivIds(coeffs, fac.getNumCols() - coeffs.size());
    coeffs[divId] = coeff;
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

/// Parse a floor division. The division is added to the provided FAC and the
/// col id of the newly added local id is returned.
///
///  pb-floor ::= floor(pb-sum `/` integer-literal)
///
ParseResult PresburgerParser::parseDivision(FlatAffineConstraints &fac,
                                            int64_t &localId) {
  consumeToken(Token::kw_floor);
  if (parseToken(Token::l_paren, "expected `(`"))
    return failure();

  SmallVector<int64_t, 8> coeffs(fac.getNumIds() + 1, 0);
  if (parseSum(fac, coeffs))
    return failure();

  if (parseToken(Token::slash, "expected `/`"))
    return failure();

  APInt divisor;
  if (!parseOptionalInteger(divisor).hasValue())
    return failure();

  if (divisor.isNegative())
    return emitError("expected divisor to be non-negative");

  if (parseToken(Token::r_paren, "expected `)`"))
    return failure();

  fac.addLocalFloorDiv(coeffs, divisor.getZExtValue());
  localId = fac.getNumIds() - 1;

  return success();
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
  AsmParserState asmParserState;
  ParserState state(sourceMgr, ctx, symbols, &asmParserState);
  PresburgerParser parser(state);

  // This set will be overwritten
  PresburgerSet set = PresburgerSet::getUniverse(1, 1);
  if (parser.parsePresburgerSet(set))
    return failure();
  return set;
}
