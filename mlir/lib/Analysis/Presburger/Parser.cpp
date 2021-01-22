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

#include "mlir/Analysis/Presburger/Parser.h"
#include "../../Parser/Parser.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace detail;

using llvm::MemoryBuffer;
using llvm::SMLoc;
using llvm::SourceMgr;
using llvm::StringMap;

namespace {

// TODO: renaming and comments
struct Coefficients {
  int64_t constant;
  SmallVector<int64_t, 8> coeffs;
};

enum class ConstraintKind { LE, GE, EQ };

/// Uses the Lexer to transform a token stream into an AST representing
/// different Presburger constructs.
class PresburgerParser : public Parser {
public:
  PresburgerParser(ParserState &state) : Parser(state){};

  /// Parse a Presburger set into set
  ParseResult parsePresburgerSet(PresburgerSet &set);

private:
  ParseResult parseFlatAffineConstraints(FlatAffineConstraints &fac);
  ParseResult initVariables(const SmallVector<StringRef, 8> &vars,
                            StringMap<unsigned> &map);

  StringMap<unsigned> dimNameToIndex;
  StringMap<unsigned> symNameToIndex;

  ParseResult parseDimAndOptionalSymbolIdList(
      std::pair<SmallVector<StringRef, 8>, SmallVector<StringRef, 8>>
          &dimSymPair);

  ParseResult parsePresburgerSetConstraints(PresburgerSet &set);

  ParseResult parseConstraint(FlatAffineConstraints &fac);
  ParseResult parseSum(Coefficients &coefs);
  ParseResult parseTerm(Coefficients &coefs, bool is_negated = false);
  ParseResult parseVariable(StringRef &var);
};

//===----------------------------------------------------------------------===//
// PresburgerParser
//===----------------------------------------------------------------------===//

/// initializes a name to id mapping for variables
ParseResult
PresburgerParser::initVariables(const SmallVector<StringRef, 8> &vars,
                                StringMap<unsigned> &map) {
  map.clear();
  for (auto &name : vars) {
    auto it = map.find(name);
    if (it != map.end())
      return emitError(
          "repeated variable names in the tuple are not yet supported");

    map.insert_or_assign(name, map.size());
  }
  return success();
}

/// Parse a Presburger set.
///
///  pb-set        ::= dim-and-symbol-use-list `:` `(` pb-or-expr? `)`
///  pb-or-expr    ::= pb-and-expr (`or` pb-and-expr)*
///  pb-and-expr   ::= pb-constraint (`and` pb-constraint)*
///  pb-constraint ::= pb-sum (`>=` | `=` | `<=`) pb-sum
///  pb-sum        ::= pb-term (('+' | '-') pb-term)*
///  pb-term       ::= '-'? pb-int? pb-var
///                ::= '-'? pb-int
///  pb-var        ::= letter (digit | letter)*
///  pb-int        ::= digit+
///
/// TODO adapt grammar to future changes
ParseResult PresburgerParser::parsePresburgerSet(PresburgerSet &set) {
  std::pair<SmallVector<StringRef, 8>, SmallVector<StringRef, 8>> dimSymPair;

  // TODO merge dimSymPair with the id list
  if (parseDimAndOptionalSymbolIdList(dimSymPair))
    return failure();

  if (parseToken(Token::colon, "expected ':'"))
    return failure();

  // init the name to id bindings
  initVariables(dimSymPair.first, dimNameToIndex);
  initVariables(dimSymPair.second, symNameToIndex);

  set =
      PresburgerSet::getUniverse(dimNameToIndex.size(), symNameToIndex.size());

  if (parsePresburgerSetConstraints(set))
    return failure();

  // TODO
  // checks that we are at the end of the string
  // if (lexer.reachedEOF())
  //  return success();
  // return emitErrorForToken(lexer.peek(),
  //                         "expected to be at the end of the set");
  return success();
}

/// Parse the list of symbolic identifiers
///
/// dim-and-symbol-use-list is defined elsewhere
///
ParseResult PresburgerParser::parseDimAndOptionalSymbolIdList(
    std::pair<SmallVector<StringRef, 8>, SmallVector<StringRef, 8>>
        &dimSymPair) {
  // TODO refactor this method!
  if (parseToken(Token::l_paren,
                 "expected '(' at start of dimensional identifiers list")) {
    return failure();
  }

  auto parseElt = [&]() -> ParseResult {
    // TODO base this on AffineParser?

    if (getToken().isNot(Token::bare_identifier))
      return emitError("expected bare identifier");

    auto name = getTokenSpelling();
    dimSymPair.first.push_back(name);
    consumeToken(Token::bare_identifier);

    return success();
  };
  if (parseCommaSeparatedListUntil(Token::r_paren, parseElt, false))
    return failure();

  if (getToken().is(Token::l_square)) {
    consumeToken(Token::l_square);
    auto parseElt = [&]() -> ParseResult {
      if (getToken().isNot(Token::bare_identifier))
        return emitError("expected bare identifier");

      auto name = getTokenSpelling();
      dimSymPair.second.push_back(name);
      consumeToken(Token::bare_identifier);

      return success();
    };
    return parseCommaSeparatedListUntil(Token::r_square, parseElt);
  }

  return success();
}

/// Parse an or expression.
///
///  pb-or-expr ::= pb-and-expr (`or` pb-and-expr)*
///
ParseResult
PresburgerParser::parsePresburgerSetConstraints(PresburgerSet &set) {
  if (parseToken(Token::l_paren, "expected '('"))
    return failure();

  if (getToken().is(Token::r_paren))
    return success();

  FlatAffineConstraints fac;
  if (parseFlatAffineConstraints(fac))
    return failure();

  set.unionFACInPlace(fac);

  while (consumeIf(Token::kw_or)) {
    if (parseFlatAffineConstraints(fac))
      return failure();
    set.unionFACInPlace(fac);
  }

  if (parseToken(Token::r_paren, "expected ')'"))
    return failure();

  return success();
}

/// Parse an and expression.
///
///  pb-and-expr ::= pb-constraint (`and` pb-constraint)*
///
ParseResult
PresburgerParser::parseFlatAffineConstraints(FlatAffineConstraints &fac) {
  fac = FlatAffineConstraints(dimNameToIndex.size(), symNameToIndex.size());

  if (parseConstraint(fac))
    return failure();

  while (consumeIf(Token::kw_and)) {
    if (parseConstraint(fac))
      return failure();
  }

  return success();
}

/// Parse a Presburger constraint
///
///  pb-constraint ::= pb-expr (`>=` | `=` | `<=`) pb-expr
///
ParseResult PresburgerParser::parseConstraint(FlatAffineConstraints &fac) {
  Coefficients left;
  Coefficients right;
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

  int64_t constant;
  SmallVector<int64_t, 8> coeffs;

  switch (kind) {
  case ConstraintKind::LE:
    constant = right.constant - left.constant;
    for (unsigned i = 0; i < left.coeffs.size(); i++)
      coeffs.push_back(right.coeffs[i] - left.coeffs[i]);
    break;

  case ConstraintKind::GE:
  case ConstraintKind::EQ:
    constant = left.constant - right.constant;
    for (unsigned i = 0; i < left.coeffs.size(); i++)
      coeffs.push_back(left.coeffs[i] - right.coeffs[i]);
    break;
  }

  // FACs have the constant at the end of the coefficients
  coeffs.push_back(constant);

  if (kind == ConstraintKind::EQ)
    fac.addEquality(coeffs);
  else
    fac.addInequality(coeffs);

  return success();
}

/// Parse a Presburger sum.
///
///  pb-sum ::= pb-term (('+' | '-') pb-term)*
///
ParseResult PresburgerParser::parseSum(Coefficients &coeffs) {
  unsigned size = dimNameToIndex.size() + symNameToIndex.size();
  coeffs = {0, SmallVector<int64_t, 8>(size, 0)};

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
///  pb-term       ::= '-'? pb-int? pb-var
///                ::= '-'? pb-int
///
ParseResult PresburgerParser::parseTerm(Coefficients &coeffs, bool isNegated) {
  if (consumeIf(Token::minus))
    isNegated = !isNegated;

  int64_t coeff = 1;
  StringRef identifier;
  bool intFound = false;
  bool varFound = false;

  if (parseOptionalInteger(coeff).hasValue()) {
    intFound = true;
    if (isNegated)
      coeff = -coeff;
  } else if (isNegated) {
    coeff = -1;
    intFound = true;
  }

  if (getToken().is(Token::bare_identifier)) {
    if (parseVariable(identifier))
      return failure();
    varFound = true;
  }

  if (!intFound && !varFound)
    return emitError("expected non empty term");

  // add term to coeffs
  if (!varFound) {
    coeffs.constant += coeff;
    return success();
  }

  auto it = dimNameToIndex.find(identifier);
  if (it != dimNameToIndex.end()) {
    coeffs.coeffs[it->second] += coeff;
    return success();
  }

  it = symNameToIndex.find(identifier);
  if (it != symNameToIndex.end()) {
    coeffs.coeffs[dimNameToIndex.size() + it->second] += coeff;
    return success();
  }

  return emitError("encountered unknown variable name: " + identifier);
}

/// Parse a variable.
///
///  pb-var ::= letter (digit | letter)*
///
ParseResult PresburgerParser::parseVariable(StringRef &var) {
  if (getToken().isNot(Token::bare_identifier))
    return failure();
  var = getTokenSpelling();
  consumeToken();
  return success();
}
} // namespace

/// Parse an PresburgerSet from a given StringRef
FailureOr<PresburgerSet> mlir::parsePresburgerSet(StringRef str,
                                                  MLIRContext *ctx) {
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(MemoryBuffer::getMemBuffer(str), SMLoc());

  SymbolState symbols;
  ParserState state(sourceMgr, ctx, symbols);
  PresburgerParser parser(state);
  PresburgerSet set = PresburgerSet::getUniverse(1, 1);
  if (parser.parsePresburgerSet(set))
    return failure();
  return set;
}
