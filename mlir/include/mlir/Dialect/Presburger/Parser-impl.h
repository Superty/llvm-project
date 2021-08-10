/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 * SPDX-FileCopyrightText: 2019 Tobias Grosser <tobias@grosser.es>
 * SPDX-FileCopyrightText: 2019 Arjun Pitchanathan <arjunpitchanathan@gmail.com>
 */

#include "mlir/Dialect/Presburger/Parser.h"
#include "mlir/Analysis/Presburger/Constraint.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::analysis::presburger;
using namespace mlir::presburger;

bool Token::isa(Kind mayKind) { return kind == mayKind; }

StringRef Token::string() { return content; }

StringRef Token::name(Token::Kind kind) {
  switch (kind) {
  case Token::Kind::Integer:
    return "integer";
  case Token::Kind::Identifier:
    return "identifier";
  case Token::Kind::LeftParen:
    return "'('";
  case Token::Kind::RightParen:
    return "')'";
  case Token::Kind::LeftSquare:
    return "'['";
  case Token::Kind::RightSquare:
    return "']'";
  case Token::Kind::LeftCurly:
    return "'{'";
  case Token::Kind::RightCurly:
    return "'}'";
  case Token::Kind::LessThan:
    return "'<'";
  case Token::Kind::LessEqual:
    return "\"<=\"";
  case Token::Kind::GreaterThan:
    return "'>'";
  case Token::Kind::GreaterEqual:
    return "'>='";
  case Token::Kind::Equal:
    return "'='";
  case Token::Kind::NotEqual:
    return "\"!=\"";
  case Token::Kind::Plus:
    return "'+'";
  case Token::Kind::Minus:
    return "'-'";
  case Token::Kind::Times:
    return "'*'";
  case Token::Kind::Divide:
    return "'/'";
  case Token::Kind::Modulo:
    return "'%'";
  case Token::Kind::Colon:
    return "':'";
  case Token::Kind::Comma:
    return "','";
  case Token::Kind::And:
    return "\"and\"";
  case Token::Kind::Or:
    return "\"or\"";
  case Token::Kind::Empty:
    return "\"empty\"";
  case Token::Kind::Exists:
    return "\"exists\"";
  case Token::Kind::Arrow:
    return "\"->\"";
  case Token::Kind::Semicolon:
    return "';'";
  case Token::Kind::Eof:
    return "EOF";
  case Token::Kind::Unknown:
    return "unknown";
  }
  llvm_unreachable("Unexpected token");
}

//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//

Lexer::Lexer(StringRef buffer, ErrorCallback callback)
    : buffer(buffer), curPos(0), callback(callback) {
  current = nextToken();
}

bool Lexer::isSpace(char c) { return std::isspace(c); }

bool Lexer::isDigit(char c) { return std::isdigit(c); }

bool Lexer::isAlpha(char c) { return std::isalpha(c); }

Token Lexer::getAtom(Token::Kind kind, unsigned start) {
  curPos++;
  return Token(kind, StringRef(start + buffer.begin(), curPos - start));
}

Token Lexer::consumeInteger(unsigned start) {
  while (isDigit(buffer[curPos]))
    curPos++;

  return Token(Token::Kind::Integer,
               StringRef(start + buffer.begin(), curPos - start));
}

/// Create an identifier of keyword. An identifier has to start witha n
/// alphabetic char and after that contains a sequence of alphanumeric chars.
///
/// If the resulting string matches a keyword an according token is returned.
Token Lexer::consumeIdentifierOrKeyword(unsigned start) {
  char c = buffer[curPos];
  assert(isAlpha(c) && "identifier or keyword should begin with an alphabet");

  while (isDigit(c) || isAlpha(c)) {
    ++curPos;
    if (reachedEOF())
      break;
    c = buffer[curPos];
  }

  StringRef content(start + buffer.begin(), curPos - start);

  // if we need more keyword, do something similar to the mlir parser
  Token::Kind kind = llvm::StringSwitch<Token::Kind>(content)
                         .Case("and", Token::Kind::And)
                         .Case("or", Token::Kind::Or)
                         .Case("empty", Token::Kind::Empty)
                         .Case("exists", Token::Kind::Exists)
                         .Default(Token::Kind::Identifier);

  return Token(kind, content);
}

/// Determines the next token and consumes it.
/// As an invariant we have that curPos always points to the next unread char
///
/// If the end of the buffer is reached, this will return a Token with kind
/// `Eof`.
Token Lexer::nextToken() {
  if (reachedEOF()) {
    return Token(Token::Kind::Eof, StringRef(buffer.begin() + curPos, 0));
  }
  while (isSpace(buffer[curPos]))
    curPos++;

  unsigned tokStart = curPos;

  char c = buffer[curPos];

  if (isDigit(c))
    return consumeInteger(tokStart);

  if (isAlpha(c))
    return consumeIdentifierOrKeyword(tokStart);

  switch (c) {
  case '(':
    return getAtom(Token::Kind::LeftParen, tokStart);
  case ')':
    return getAtom(Token::Kind::RightParen, tokStart);
  case '[':
    return getAtom(Token::Kind::LeftSquare, tokStart);
  case ']':
    return getAtom(Token::Kind::RightSquare, tokStart);
  case '{':
    return getAtom(Token::Kind::LeftCurly, tokStart);
  case '}':
    return getAtom(Token::Kind::RightCurly, tokStart);
  case '+':
    return getAtom(Token::Kind::Plus, tokStart);
  case '-':
    if (buffer[curPos + 1] == '>') {
      curPos++;
      return getAtom(Token::Kind::Arrow, tokStart);
    }
    return getAtom(Token::Kind::Minus, tokStart);
  case '*':
    return getAtom(Token::Kind::Times, tokStart);
  case '/':
    return getAtom(Token::Kind::Divide, tokStart);
  case '%':
    return getAtom(Token::Kind::Modulo, tokStart);
  case '<':
    if (buffer[curPos + 1] == '=') {
      curPos++;
      return getAtom(Token::Kind::LessEqual, tokStart);
    }
    return getAtom(Token::Kind::LessThan, tokStart);
  case '>':
    if (buffer[curPos + 1] == '=') {
      curPos++;
      return getAtom(Token::Kind::GreaterEqual, tokStart);
    }
    return getAtom(Token::Kind::GreaterThan, tokStart);
  case '=':
    return getAtom(Token::Kind::Equal, tokStart);
  case '!':
    if (buffer[curPos + 1] == '=') {
      curPos++;
      return getAtom(Token::Kind::NotEqual, tokStart);
    }
    return getAtom(Token::Kind::Unknown, tokStart);
  case ':':
    return getAtom(Token::Kind::Colon, tokStart);
  case ',':
    return getAtom(Token::Kind::Comma, tokStart);
  case ';':
    return getAtom(Token::Kind::Semicolon, tokStart);
  default:
    return getAtom(Token::Kind::Unknown, tokStart);
  }
}

/// Returns the next token without consuming it.
Token Lexer::peek() { return current; }

/// Returns the next token and consumes it.
Token Lexer::next() {
  Token result = current;
  current = nextToken();
  return result;
}

/// Consumes a token with specified kind. If non is present it is a no-op.
void Lexer::consume(Token::Kind kind) {
  if (current.isa(kind))
    next();
}

/// Consumes the next token and emits an error if it doesn't match the
/// provided kind.
LogicalResult Lexer::consumeKindOrError(Token::Kind kind) {
  Token token = next();
  if (!token.isa(kind))
    return emitError("expected " + Token::name(kind));

  return success();
}

/// Consumes the next token and emits an error if it doesn't match the
/// provided kind.
LogicalResult Lexer::consumeKindOrError(Token::Kind kind, Token &token) {
  token = next();
  if (!token.isa(kind))
    return emitError("expected " + Token::name(kind));

  return success();
}

bool Lexer::reachedEOF() {
  assert(curPos <= buffer.size() && "read outside of the buffer");
  return curPos == buffer.size();
}

/// Emits the provided error message at the location provided.
InFlightDiagnostic Lexer::emitError(const char *loc, const Twine &message) {
  return callback(SMLoc::getFromPointer(loc), message);
}

/// Emits the provided error message at the location provided.
InFlightDiagnostic Lexer::emitError(SMLoc loc, const Twine &message) {
  return callback(loc, message);
}

/// Emits the provided error message at the start of the buffer.
InFlightDiagnostic Lexer::emitErrorAtStart(const Twine &message) {
  return emitError(SMLoc::getFromPointer(buffer.begin()), message);
}

/// Emits the provided error message at the current possition of the Lexer.
InFlightDiagnostic Lexer::emitError(const Twine &message) {
  assert(!reachedEOF() &&
         "lexer is out of range, you have to specify a location");
  return emitError(buffer.begin() + curPos, message);
}

//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//

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
template <typename Int>
LogicalResult Parser<Int>::parseSet(std::unique_ptr<SetExpr<Int>> &setExpr) {
  std::pair<SmallVector<StringRef, 8>, SmallVector<StringRef, 8>> dimSymPair;
  if (failed(parseDimAndSymbolIdLists(dimSymPair)))
    return failure();

  if (!lexer.peek().isa(Token::Kind::Colon))
    return emitErrorForToken(lexer.peek(),
                             "expected ':' but got: " + lexer.peek().string());
  lexer.next();

  std::unique_ptr<Expr> constraints;
  if (failed(parseOr(constraints)))
    return failure();

  setExpr = std::make_unique<SetExpr<Int>>(std::move(dimSymPair.first),
                                      std::move(dimSymPair.second),
                                      std::move(constraints));
  // checks that we are at the end of the string
  if (lexer.reachedEOF())
    return success();
  return emitErrorForToken(lexer.peek(),
                           "expected to be at the end of the set");
}

/// Parse a Presburger expression.
///
///  pb-expr ::= dim-and-symbol-use-list `->` piece (`;` piece)*
///  piece      ::= `(`pb-sum`) : (`pb-or-expr`)`
///
template <typename Int>
LogicalResult Parser<Int>::parseExpr(std::unique_ptr<PresburgerExprExpr> &expr) {
  std::pair<SmallVector<StringRef, 8>, SmallVector<StringRef, 8>> dimSymPair;
  SmallVector<std::unique_ptr<PieceExpr>, 4> pieces;
  if (failed(parseDimAndSymbolIdLists(dimSymPair)))
    return failure();

  if (!lexer.peek().isa(Token::Kind::Arrow))
    return emitErrorForToken(lexer.peek(), "expected \"->\" but got: " +
                                               lexer.peek().string());
  lexer.next();

  std::unique_ptr<PieceExpr> pieceExpr;

  if (failed(parsePiece(pieceExpr)))
    return failure();
  pieces.push_back(std::move(pieceExpr));

  while (lexer.peek().isa(Token::Kind::Semicolon)) {
    lexer.next();
    if (failed(parsePiece(pieceExpr)))
      return failure();
    pieces.push_back(std::move(pieceExpr));
  }

  expr = std::make_unique<PresburgerExprExpr>(std::move(dimSymPair.first),
                                              std::move(dimSymPair.second),
                                              std::move(pieces));

  if (lexer.reachedEOF())
    return success();
  return emitErrorForToken(lexer.peek(),
                           "expected to be at the end of the set");
}

/// This parses a string following the variable rule. It is only used to
/// allow PresburgerDialect::parseAttribute to check what kind of attribute
/// should be parsed
///
/// TODO find a nicer way to do this
template <typename Int>
LogicalResult Parser<Int>::parseKeyword(StringRef &keyword) {
  std::unique_ptr<VariableExpr> varExpr;
  if (failed(parseVariable(varExpr)))
    return failure();
  keyword = varExpr->getName();
  return success();
}

/// Parse a comma-separated list of elements, terminated with an arbitrary
/// token.  This allows empty lists if allowEmptyList is true.
///
///   abstract-list ::= rightToken           // if allowEmptyList == true
///   abstract-list ::= element (',' element)* rightToken
///
template <typename Int>
LogicalResult Parser<Int>::parseCommaSeparatedListUntil(SmallVector<StringRef, 8> &l,
                                                   Token::Kind rightToken,
                                                   bool allowEmpty) {
  Token token = lexer.peek();
  while (token.isa(Token::Kind::Identifier)) {
    l.push_back(token.string());
    lexer.next();
    if (lexer.peek().isa(rightToken)) {
      lexer.next();
      return success();
    }

    if (!lexer.peek().isa(Token::Kind::Comma))
      return emitErrorForToken(lexer.peek(),
                               "expected ',' or " + Token::name(rightToken));

    lexer.next();
    token = lexer.peek();
  }

  if (!allowEmpty)
    return emitErrorForToken(token, "expected non empty list");

  return lexer.consumeKindOrError(rightToken);
}

/// Parse the list of symbolic identifiers
///
/// dim-and-symbol-use-list is defined elsewhere
///
template <typename Int>
LogicalResult Parser<Int>::parseDimAndSymbolIdLists(
    std::pair<SmallVector<StringRef, 8>, SmallVector<StringRef, 8>>
        &dimSymPair) {
  if (failed(lexer.consumeKindOrError(Token::Kind::LeftParen)))
    return failure();

  if (failed(parseCommaSeparatedListUntil(dimSymPair.first,
                                          Token::Kind::RightParen, true)))
    return failure();

  if (lexer.peek().isa(Token::Kind::LeftSquare)) {
    lexer.next();
    if (failed(parseCommaSeparatedListUntil(dimSymPair.second,
                                            Token::Kind::RightSquare, true)))
      return failure();
  }

  return success();
}

/// Parse an or expression.
///
///  pb-or-expr ::= pb-and-expr (`or` pb-and-expr)*
///
template <typename Int>
LogicalResult Parser<Int>::parseOr(std::unique_ptr<Expr> &expr) {
  SmallVector<std::unique_ptr<Expr>, 8> exprs;
  std::unique_ptr<Expr> andExpr;

  if (lexer.peek().isa(Token::Kind::Empty)) {
    expr = std::make_unique<OrExpr<Int>>(std::move(exprs));
    return success();
  }

  if (failed(parseAnd(andExpr)))
    return failure();
  exprs.push_back(std::move(andExpr));
  while (lexer.peek().isa(Token::Kind::Or)) {
    lexer.next();
    if (failed(parseAnd(andExpr)))
      return failure();
    exprs.push_back(std::move(andExpr));
  }

  if (exprs.size() == 1) {
    expr = std::move(exprs[0]);
    return success();
  }

  expr = std::make_unique<OrExpr<Int>>(std::move(exprs));
  return success();
}

/// Parse an and expression.
///
///  pb-and-expr ::= pb-constraint (`and` pb-constraint)*
///
template <typename Int>
LogicalResult Parser<Int>::parseAnd(std::unique_ptr<Expr> &expr) {
  if (failed(lexer.consumeKindOrError(Token::Kind::LeftParen)))
    return failure();

  SmallVector<StringRef, 8> exists;
  SmallVector<StringRef, 8> divNames;
  SmallVector<std::unique_ptr<DivExpr<Int>>, 8> divs;

  if (lexer.peek().isa(Token::Kind::Exists)) {
    lexer.next();
    while (true) {
      std::unique_ptr<VariableExpr> varExpr;
      if (failed(parseVariable(varExpr)))
        return failure();
      StringRef name = varExpr->getName();

      if (lexer.peek().isa(Token::Kind::Equal)) {
        lexer.next();
        if (failed(lexer.consumeKindOrError(Token::Kind::LeftSquare)))
          return failure();
        if (failed(lexer.consumeKindOrError(Token::Kind::LeftParen)))
          return failure();

        std::unique_ptr<Expr> num;
        if (failed(parseSum(num)))
          return failure();

        if (failed(lexer.consumeKindOrError(Token::Kind::RightParen)))
          return failure();
        if (failed(lexer.consumeKindOrError(Token::Kind::Divide)))
          return failure();

        std::unique_ptr<IntegerExpr<Int>> den;
        if (failed(parseInteger(den, false)))
          return failure();

        if (failed(lexer.consumeKindOrError(Token::Kind::RightSquare)))
          return failure();

        divNames.emplace_back(name);
        divs.emplace_back(
            std::make_unique<DivExpr<Int>>(std::move(num), std::move(den)));
      } else {
        exists.emplace_back(name);
      }

      if (lexer.peek().isa(Token::Kind::Colon)) {
        lexer.next();
        break;
      }

      if (failed(lexer.consumeKindOrError(Token::Kind::Comma)))
        lexer.next();
    }
  }

  SmallVector<std::unique_ptr<ConstraintExpr<Int>>, 8> constraints;
  if (lexer.peek().isa(Token::Kind::RightParen)) {
    lexer.next();
    expr = std::make_unique<AndExpr<Int>>(std::move(constraints), std::move(exists),
                                     std::move(divNames), std::move(divs));
    return success();
  }

  std::unique_ptr<ConstraintExpr<Int>> c;
  if (failed(parseConstraint(c)))
    return failure();
  constraints.push_back(std::move(c));

  while (lexer.peek().isa(Token::Kind::And)) {
    lexer.next();
    if (failed(parseConstraint(c)))
      return failure();
    constraints.push_back(std::move(c));
  }

  if (failed(lexer.consumeKindOrError(Token::Kind::RightParen)))
    return failure();

  expr = std::make_unique<AndExpr<Int>>(std::move(constraints), std::move(exists),
                                   std::move(divNames), std::move(divs));
  return success();
}

/// Parse a piece of a Presburger expression.
///
///  piece ::= `(`pb-sum`) : (`pb-or-expr`)`
///
template <typename Int>
LogicalResult Parser<Int>::parsePiece(std::unique_ptr<PieceExpr> &piece) {
  if (failed(lexer.consumeKindOrError(Token::Kind::LeftParen)))
    return failure();

  std::unique_ptr<Expr> expr;
  if (failed(parseSum(expr)))
    return failure();

  if (failed(lexer.consumeKindOrError(Token::Kind::RightParen)) ||
      failed(lexer.consumeKindOrError(Token::Kind::Colon)))
    return failure();

  std::unique_ptr<Expr> constraints;

  parseOr(constraints);

  piece = std::make_unique<PieceExpr>(std::move(expr), std::move(constraints));

  return success();
}

/// Parse a Presburger constraint
///
///  pb-constraint ::= pb-expr (`>=` | `=` | `<=`) pb-expr
///
template <typename Int>
LogicalResult
Parser<Int>::parseConstraint(std::unique_ptr<ConstraintExpr<Int>> &constraint) {
  std::unique_ptr<Expr> leftExpr;
  if (failed(parseSum(leftExpr)))
    return failure();

  typename ConstraintExpr<Int>::Kind kind;
  Token cmpToken = lexer.next();
  if (cmpToken.isa(Token::Kind::GreaterEqual))
    kind = ConstraintExpr<Int>::Kind::GE;
  else if (cmpToken.isa(Token::Kind::Equal))
    kind = ConstraintExpr<Int>::Kind::EQ;
  else if (cmpToken.isa(Token::Kind::LessEqual))
    kind = ConstraintExpr<Int>::Kind::LE;
  else if (cmpToken.isa(Token::Kind::GreaterThan)) {
    return emitErrorForToken(cmpToken, "strict inequalities are not supported");
  } else if (cmpToken.isa(Token::Kind::LessThan)) {
    return emitErrorForToken(cmpToken, "strict inequalities are not supported");
  } else if (cmpToken.isa(Token::Kind::NotEqual)) {
    return emitErrorForToken(cmpToken, "!= constraints are not supported");
  } else {
    return emitErrorForToken(cmpToken, "expected comparison operator");
  }

  std::unique_ptr<Expr> rightExpr;
  if (failed(parseSum(rightExpr)))
    return failure();

  constraint = std::make_unique<ConstraintExpr<Int>>(kind, std::move(leftExpr),
                                                std::move(rightExpr));
  return success();
}

/// Parse a Presburger sum.
///
///  pb-sum ::= pb-term (('+' | '-') pb-term)*
///
template <typename Int>
LogicalResult Parser<Int>::parseSum(std::unique_ptr<Expr> &expr) {
  SmallVector<std::unique_ptr<TermExpr<Int>>, 8> terms;
  std::unique_ptr<TermExpr<Int>> term;

  if (failed(parseTerm(term)))
    return failure();
  terms.push_back(std::move(term));

  while (lexer.peek().isa(Token::Kind::Plus) ||
         lexer.peek().isa(Token::Kind::Minus)) {
    Token signToken = lexer.next();

    if (failed(parseTerm(term, signToken.isa(Token::Kind::Minus))))
      return failure();

    terms.push_back(std::move(term));
  }

  if (terms.size() == 1) {
    expr = std::move(terms[0]);
    return success();
  }

  expr = std::make_unique<SumExpr<Int>>(std::move(terms));
  return success();
}

/// Parse a Presburger term.
///
///  pb-term       ::= '-'? pb-int? pb-var
///                ::= '-'? pb-int
///
template <typename Int>
LogicalResult Parser<Int>::parseTerm(std::unique_ptr<TermExpr<Int>> &term,
                                bool isNegated) {
  std::unique_ptr<IntegerExpr<Int>> integer;
  if (lexer.peek().isa(Token::Kind::Minus)) {
    isNegated = !isNegated;
    lexer.next();
  }

  if (lexer.peek().isa(Token::Kind::Integer)) {
    if (failed(parseInteger(integer, isNegated)))
      return failure();
    lexer.consume(Token::Kind::Times);
  } else if (isNegated)
    integer = std::make_unique<IntegerExpr<Int>>(-1);

  std::unique_ptr<VariableExpr> identifier;
  if (lexer.peek().isa(Token::Kind::Identifier))
    if (failed(parseVariable(identifier)))
      return failure();

  if (!integer.get() && !identifier.get())
    return emitErrorForToken(lexer.peek(), "expected non empty term");

  term = std::make_unique<TermExpr<Int>>(std::move(integer), std::move(identifier));
  return success();
}

/// Parse a variable.
///
///  pb-var ::= letter (digit | letter)*
///
template <typename Int>
LogicalResult Parser<Int>::parseVariable(std::unique_ptr<VariableExpr> &vExpr) {
  Token t;
  if (failed(lexer.consumeKindOrError(Token::Kind::Identifier, t)))
    return failure();
  vExpr = std::make_unique<VariableExpr>(t.string());
  return success();
}

/// Parse a signless integer.
///
///  pb-int ::= digit+
///
template <typename Int>
LogicalResult Parser<Int>::parseInteger(std::unique_ptr<IntegerExpr<Int>> &iExpr,
                                   bool isNegated) {
  bool negativ = isNegated ^ lexer.peek().isa(Token::Kind::Minus);
  lexer.consume(Token::Kind::Minus);

  Token integerToken;
  if (failed(lexer.consumeKindOrError(Token::Kind::Integer, integerToken)))
    return failure();

  constexpr bool intIsGmp = std::is_same<Int, mpz_class>::value;

  typename std::conditional<intIsGmp, int64_t, UnderlyingInt<Int>>::type value;
  if (!llvm::to_integer(integerToken.string(), value)) {
    if constexpr (intIsGmp) {
      llvm::errs() << "parsing coefficients that don't fix in 64-bits is not yet implemented!";
      exit(1);
    }
    return emitErrorForToken(integerToken, "expected a valid integer");
  }
  if (negativ)
    value = -value;

  iExpr = std::make_unique<IntegerExpr<Int>>(value);
  return success();
}

template <typename Int>
InFlightDiagnostic Parser<Int>::emitErrorForToken(Token token,
                                             const Twine &message) {
  return lexer.emitError(token.string().begin(), message);
}

template <typename Int>
InFlightDiagnostic Parser<Int>::emitError(const Twine &message) {
  return lexer.emitErrorAtStart(message);
}

template <typename Int>
InFlightDiagnostic Parser<Int>::emitError(SMLoc loc, const Twine &message) {
  return lexer.emitError(loc, message);
}

//===----------------------------------------------------------------------===//
// PresburgerParser<Int>
//===----------------------------------------------------------------------===//
//
template <typename Int>
PresburgerParser<Int>::PresburgerParser(Parser<Int> parser) : parser(parser) {}

/// initializes a name to id mapping for variables
template <typename Int>
LogicalResult
PresburgerParser<Int>::initVariables(const SmallVector<StringRef, 8> &vars,
                                StringMap<size_t> &map) {
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

/// Parse a Presburger set into set
///
/// For the exact parsing rules, see Parser<Int>::parseSet
template <typename Int>
LogicalResult PresburgerParser<Int>::parsePresburgerSet(PresburgerSet<Int> &set) {
  std::unique_ptr<SetExpr<Int>> setExpr;
  if (failed(parser.parseSet(setExpr)))
    return failure();

  if (setExpr->getConstraints() == nullptr) {
    set = PresburgerSet<Int>(setExpr->getDims().size(), setExpr->getSyms().size());
    return success();
  }

  initVariables(setExpr->getDims(), dimNameToIndex);
  initVariables(setExpr->getSyms(), symNameToIndex);
  if (failed(parsePresburgerSet(setExpr->getConstraints(), set)))
    return failure();

  return success();
}

/// Creates a PresburgerSet<Int> instance from constraints
///
/// For each AndExpr<Int> contained in constraints it creates one
/// PresburgerBasicSet<Int> object
template <typename Int>
LogicalResult PresburgerParser<Int>::parsePresburgerSet(Expr *constraints,
                                                   PresburgerSet<Int> &set) {
  set = PresburgerSet<Int>(dimNameToIndex.size(), symNameToIndex.size());
  if (auto orConstraints = constraints->dyn_cast<OrExpr<Int>>()) {
    for (std::unique_ptr<Expr> &basicSet : orConstraints->getConstraints()) {
      PresburgerBasicSet<Int> bs;
      if (failed(parsePresburgerBasicSet(basicSet.get(), bs)))
        return failure();
      set.addBasicSet(bs);
    }
    return success();
  }

  PresburgerBasicSet<Int> bs;
  if (failed(parsePresburgerBasicSet(constraints, bs)))
    return failure();

  set.addBasicSet(bs);

  return success();
}

/// Creates a FlatAffineConstraint instance from constraints
///
/// Expects either a single ConstraintExpr<Int> or multiple of them combined in an
/// AndExpr<Int>
template <typename Int>
LogicalResult
PresburgerParser<Int>::parsePresburgerBasicSet(Expr *constraints,
                                          PresburgerBasicSet<Int> &cs) {
  if (constraints->dyn_cast<OrExpr<Int>>() != nullptr)
    return emitError("or conditions are not valid for basic sets");

  if (auto constraint = constraints->dyn_cast<ConstraintExpr<Int>>()) {
    llvm_unreachable("PresburgerBasicSets should only contain AndExprs.");
    PresburgerParser<Int>::Constraint c;
    if (failed(parseConstraint(constraint, c)))
      return failure();
    addConstraint(cs, c);
  } else if (auto andConstraints = constraints->dyn_cast<AndExpr<Int>>()) {
    initVariables(andConstraints->getExists(), existNameToIndex);
    initVariables(andConstraints->getDivNames(), divNameToIndex);
    cs = PresburgerBasicSet<Int>(dimNameToIndex.size(), symNameToIndex.size(),
                            existNameToIndex.size());

    unsigned offset =
        dimNameToIndex.size() + symNameToIndex.size() + existNameToIndex.size();
    SmallVector<DivisionConstraint<Int>, 8> divs;
    for (auto &divExpr : andConstraints->getDivs()) {
      std::pair<Int, SmallVector<Int, 8>> affineSum;
      parseSum(divExpr->num.get(), affineSum);
      Int denominator = divExpr->den->getValue();
      SmallVector<Int, 8> coeffs(affineSum.second);
      coeffs.push_back(affineSum.first);
      divs.emplace_back(coeffs, denominator, offset + divs.size());
    }
    cs.appendDivisionVariables(divs);

    for (std::unique_ptr<ConstraintExpr<Int>> &constraint :
         andConstraints->getConstraints()) {
      PresburgerParser<Int>::Constraint c;
      if (failed(parseConstraint(constraint.get(), c)))
        return failure();
      addConstraint(cs, c);
    }
  } else {
    return emitError("constraints expression should be one of"
                     "\"and\", \"or\", \"constraint\"");
  }
  return success();
}

/// Creates a constraint from a ConstraintExpr<Int>.
///
/// It either returns an equality (== 0) or an inequalitiy (>= 0).
/// As a ConstraintExpr<Int> contains sums on both sides, they are subtracted from
/// each other to get the desired form.
template <typename Int>
LogicalResult
PresburgerParser<Int>::parseConstraint(ConstraintExpr<Int> *constraint,
                                  PresburgerParser<Int>::Constraint &c) {
  if (constraint == nullptr)
    llvm_unreachable("constraint was nullptr!");

  std::pair<Int, SmallVector<Int, 8>> left;
  std::pair<Int, SmallVector<Int, 8>> right;
  if (failed(parseSum(constraint->getLeftSum(), left)) ||
      failed(parseSum(constraint->getRightSum(), right)))
    return failure();

  auto leftConst = left.first;
  auto leftCoeffs = left.second;
  auto rightConst = right.first;
  auto rightCoeffs = right.second;

  Int constant;
  SmallVector<Int, 8> coeffs;
  if (constraint->getKind() == ConstraintExpr<Int>::Kind::LE) {
    constant = rightConst - leftConst;
    for (size_t i = 0; i < leftCoeffs.size(); i++)
      coeffs.push_back(rightCoeffs[i] - leftCoeffs[i]);
  } else if (constraint->getKind() == ConstraintExpr<Int>::Kind::GE ||
             constraint->getKind() == ConstraintExpr<Int>::Kind::EQ) {
    constant = leftConst - rightConst;
    for (size_t i = 0; i < leftCoeffs.size(); i++)
      coeffs.push_back(leftCoeffs[i] - rightCoeffs[i]);
  } else {
    llvm_unreachable("invalid constraint kind");
  }

  Kind kind;
  if (constraint->getKind() == ConstraintExpr<Int>::Kind::EQ)
    kind = Kind::Equality;
  else
    kind = Kind::Inequality;

  coeffs.push_back(constant);
  c = {coeffs, kind};
  return success();
}

/// Creates a list of coefficients and a constant from a SumExpr<Int> or a TermExpr<Int>.
///
/// The list of coefficients corresponds to the coefficients of the dimensions
/// and after that the symbols.
///
template <typename Int>
LogicalResult PresburgerParser<Int>::parseSum(
    Expr *expr, std::pair<Int, SmallVector<Int, 8>> &r) {
  Int constant = 0;
  SmallVector<Int, 8> coeffs(
      dimNameToIndex.size() + symNameToIndex.size() + existNameToIndex.size() +
          divNameToIndex.size(),
      0);
  if (auto *term = expr->dyn_cast<TermExpr<Int>>()) {
    if (failed(parseAndAddTerm(term, constant, coeffs)))
      return failure();
  } else if (auto *sum = expr->dyn_cast<SumExpr<Int>>()) {
    for (std::unique_ptr<TermExpr<Int>> &term : sum->getTerms())
      if (failed(parseAndAddTerm(term.get(), constant, coeffs)))
        return failure();
  }
  r = {constant, coeffs};
  return success();
}

/// Takes a TermExpr<Int> and addapts the matchin coefficient or the constant to this
/// terms value. To determine the coefficient id it looks it up in the
/// nameToIndex mappings
///
/// Fails if the variable name is unknown
template <typename Int>
LogicalResult
PresburgerParser<Int>::parseAndAddTerm(TermExpr<Int> *term, Int &constant,
                                  SmallVector<Int, 8> &coeffs) {
  Int delta = 1;
  if (auto coeff = term->getCoeff())
    delta = coeff->getValue();

  auto var = term->getVar();
  if (!var) {
    constant += delta;
    return success();
  }

  unsigned offset = 0;
  auto it = dimNameToIndex.find(var->getName());
  if (it != dimNameToIndex.end()) {
    coeffs[offset + it->second] += delta;
    return success();
  }
  offset += dimNameToIndex.size();

  it = symNameToIndex.find(var->getName());
  if (it != symNameToIndex.end()) {
    coeffs[offset + it->second] += delta;
    return success();
  }
  offset += symNameToIndex.size();

  it = existNameToIndex.find(var->getName());
  if (it != existNameToIndex.end()) {
    coeffs[offset + it->second] += delta;
    return success();
  }
  offset += existNameToIndex.size();

  it = divNameToIndex.find(var->getName());
  if (it != divNameToIndex.end()) {
    coeffs[offset + it->second] += delta;
    return success();
  }
  offset += divNameToIndex.size();

  return emitError("encountered unknown variable name: " + var->getName());
}

template <typename Int>
void PresburgerParser<Int>::addConstraint(PresburgerBasicSet<Int> &bs,
                                     PresburgerParser<Int>::Constraint &constraint) {
  if (constraint.second == Kind::Equality)
    bs.addEquality(constraint.first);
  else
    bs.addInequality(constraint.first);
}

/// Parse a Presburger expression into expr
///
/// For the exact parsing rules, see Parser<Int>::parseExpr
template <typename Int>
LogicalResult PresburgerParser<Int>::parsePresburgerExpr(PresburgerExpr &res) {

  std::unique_ptr<PresburgerExprExpr> expr;
  if (failed(parser.parseExpr(expr)))
    return failure();

  initVariables(expr->getDims(), dimNameToIndex);
  initVariables(expr->getSyms(), symNameToIndex);

  res = PresburgerExpr(dimNameToIndex.size(), symNameToIndex.size());

  assert(expr->getPieces().size() > 0 &&
         "expect atleast one piece in a Presburger expression");

  for (std::unique_ptr<PieceExpr> &piece : expr->getPieces())
    if (failed(parseAndAddPiece(piece.get(), res)))
      return failure();

  return success();
}

/// Takes a PieceExpr and adds its corresponding representation to expr
template <typename Int>
LogicalResult PresburgerParser<Int>::parseAndAddPiece(PieceExpr *piece,
                                                 PresburgerExpr &expr) {
  std::pair<Int, SmallVector<Int, 8>> affineExpr;
  parseSum(piece->getExpr(), affineExpr);

  PresburgerSet<Int> set;

  if (piece->getConstraints() == nullptr)
    set = PresburgerSet<Int>(expr.getNumDims(), expr.getNumSyms());
  else
    parsePresburgerSet(piece->getConstraints(), set);

  llvm_unreachable("not yet implemented!");
  // expr.addPiece(affineExpr, set);
  return success();
}

/// Emits the provided error message at the start of the string to parse.
template <typename Int>
InFlightDiagnostic PresburgerParser<Int>::emitError(const Twine &message) {
  return parser.emitError(message);
}

/// Emits the provided error message at the location provided.
template <typename Int>
InFlightDiagnostic PresburgerParser<Int>::emitError(SMLoc loc,
                                               const Twine &message) {
  return parser.emitError(loc, message);
}
