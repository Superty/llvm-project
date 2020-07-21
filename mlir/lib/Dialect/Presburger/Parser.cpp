/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 * SPDX-FileCopyrightText: 2019 Tobias Grosser <tobias@grosser.es>
 * SPDX-FileCopyrightText: 2019 Arjun Pitchanathan <arjunpitchanathan@gmail.com>
 */

#include "mlir/Dialect/Presburger/Parser.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::presburger;

bool Token::isa(Kind mayKind) { return kind == mayKind; }

StringRef Token::string() { return content; }

std::string Token::name(Token::Kind kind) {
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
  case Token::Kind::Unknown:
    return "unknown";
  }
  llvm_unreachable("Unexpected token");
}

//===----------------------------------------------------------------------===//
// Lexer (from libint)
//===----------------------------------------------------------------------===//

Lexer::Lexer(StringRef buffer, ErrorCallback callback)
    : buffer(buffer), curPtr(buffer.begin()), callback(callback) {
  current = nextToken();
}

bool Lexer::isSpace(char c) { return std::isspace(c); }

bool Lexer::isDigit(char c) { return std::isdigit(c); }

bool Lexer::isAlpha(char c) { return std::isalpha(c); }

Token Lexer::atom(Token::Kind kind, const char *start) {
  curPtr++;
  return Token(kind, StringRef(start, curPtr - start));
}

Token Lexer::integer(const char *start) {
  while (isDigit(*curPtr))
    curPtr++;

  return Token(Token::Kind::Integer, StringRef(start, curPtr - start));
}

Token Lexer::identifierOrKeyword(const char *start) {
  char c = *curPtr;
  assert(isAlpha(c) && "identifier or keyword should begin with an alphabet");

  while (isDigit(c) || isAlpha(c)) {
    c = *++curPtr;
  }

  StringRef content(start, curPtr - start);

  // if we need more keyword, do something similar to the mlir parser
  Token::Kind kind = llvm::StringSwitch<Token::Kind>(content)
                         .Case("and", Token::Kind::And)
                         .Case("or", Token::Kind::Or)
                         .Default(Token::Kind::Identifier);

  return Token(kind, StringRef(start, curPtr - start));
}

Token Lexer::nextToken() {
  while (isSpace(*curPtr))
    curPtr++;

  const char *tokStart = curPtr;

  char c = *curPtr;

  if (isDigit(c))
    return integer(tokStart);

  if (isAlpha(c))
    return identifierOrKeyword(tokStart);

  switch (c) {
  case '(':
    return atom(Token::Kind::LeftParen, tokStart);
  case ')':
    return atom(Token::Kind::RightParen, tokStart);
  case '[':
    return atom(Token::Kind::LeftSquare, tokStart);
  case ']':
    return atom(Token::Kind::RightSquare, tokStart);
  case '{':
    return atom(Token::Kind::LeftCurly, tokStart);
  case '}':
    return atom(Token::Kind::RightCurly, tokStart);
  case '+':
    return atom(Token::Kind::Plus, tokStart);
  case '-':
    return atom(Token::Kind::Minus, tokStart);
  case '*':
    return atom(Token::Kind::Times, tokStart);
  case '/':
    return atom(Token::Kind::Divide, tokStart);
  case '%':
    return atom(Token::Kind::Modulo, tokStart);
  case '<':
    if (*(curPtr + 1) == '=') {
      curPtr++;
      return atom(Token::Kind::LessEqual, tokStart);
    }
    return atom(Token::Kind::LessThan, tokStart);
  case '>':
    if (*(curPtr + 1) == '=') {
      curPtr++;
      return atom(Token::Kind::GreaterEqual, tokStart);
    }
    return atom(Token::Kind::GreaterThan, tokStart);
  case '=':
    return atom(Token::Kind::Equal, tokStart);
  case '!':
    if (*(curPtr + 1) == '=') {
      curPtr++;
      return atom(Token::Kind::NotEqual, tokStart);
    }
    return atom(Token::Kind::Unknown, tokStart);
  case ':':
    return atom(Token::Kind::Colon, tokStart);
  case ',':
    return atom(Token::Kind::Comma, tokStart);
  default:
    return atom(Token::Kind::Unknown, tokStart);
  }
}

Token Lexer::peek() { return current; }

Token Lexer::next() {
  Token result = current;
  current = nextToken();
  return result;
}

void Lexer::consume(Token::Kind kind) {
  if (current.isa(kind))
    next();
}

LogicalResult Lexer::nextAssertKind(Token::Kind kind) {
  Token token = next();
  if (!token.isa(kind))
    return emitError("expected " + Token::name(kind));

  return success();
}

LogicalResult Lexer::nextAssertKind(Token::Kind kind, Token &token) {
  token = next();
  if (!token.isa(kind))
    return emitError("expected " + Token::name(kind));

  return success();
}

bool Lexer::reachedEOF() { return curPtr - 1 == buffer.end(); }

InFlightDiagnostic Lexer::emitError(const char *loc, const Twine &message) {
  return callback(SMLoc::getFromPointer(loc), message);
}

InFlightDiagnostic Lexer::emitError(SMLoc loc, const Twine &message) {
  return callback(loc, message);
}

InFlightDiagnostic Lexer::emitErrorAtStart(const Twine &message) {
  return emitError(SMLoc::getFromPointer(buffer.begin()), message);
}

InFlightDiagnostic Lexer::emitError(const Twine &message) {
  assert(!reachedEOF() &&
         "curPtr is out of range, you have to specify a location");
  return emitError(curPtr, message);
}

//===----------------------------------------------------------------------===//
// Parser (from libint)
//===----------------------------------------------------------------------===//

LogicalResult Parser::parse(std::unique_ptr<Expr> &expr) {
  return parseOr(expr);
}

LogicalResult Parser::parseSet(std::unique_ptr<SetExpr> &setExpr) {
  std::pair<SmallVector<StringRef, 8>, SmallVector<StringRef, 8>> dimSymPair;
  if (failed(parseDimAndOptionalSymbolIdList(dimSymPair)))
    return failure();

  if (!lexer.peek().isa(Token::Kind::Colon))
    return emitErrorForToken(lexer.peek(),
                             "expected ':' but got: " + lexer.peek().string());

  lexer.next();
  if (failed(lexer.nextAssertKind(Token::Kind::LeftParen)))
    return failure();

  if (lexer.peek().isa(Token::Kind::RightParen)) {
    lexer.next();
    setExpr = std::make_unique<SetExpr>(std::move(dimSymPair.first),
                                        std::move(dimSymPair.second), nullptr);
    // checks that we are at the end of the string
    if (lexer.reachedEOF())
      return success();
    return emitErrorForToken(lexer.peek(),
                             "expected to be at the end of the set");
  }

  std::unique_ptr<Expr> constraints;
  if (failed(parseOr(constraints)))
    return failure();

  if (failed(lexer.nextAssertKind(Token::Kind::RightParen)))
    return failure();

  setExpr = std::make_unique<SetExpr>(std::move(dimSymPair.first),
                                      std::move(dimSymPair.second),
                                      std::move(constraints));
  // checks that we are at the end of the string
  if (lexer.reachedEOF())
    return success();
  return emitErrorForToken(lexer.peek(),
                           "expected to be at the end of the set");
}

LogicalResult Parser::parseCommaSeparatedListUntil(SmallVector<StringRef, 8> &l,
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

  return lexer.nextAssertKind(rightToken);
}

LogicalResult Parser::parseDimAndOptionalSymbolIdList(
    std::pair<SmallVector<StringRef, 8>, SmallVector<StringRef, 8>>
        &dimSymPair) {
  if (failed(lexer.nextAssertKind(Token::Kind::LeftParen)))
    return failure();

  if (failed(parseCommaSeparatedListUntil(dimSymPair.first,
                                          Token::Kind::RightParen, false)))
    return failure();

  if (lexer.peek().isa(Token::Kind::LeftSquare)) {
    lexer.next();
    if (failed(parseCommaSeparatedListUntil(dimSymPair.second,
                                            Token::Kind::RightSquare, true)))
      return failure();
  }

  return success();
}

LogicalResult Parser::parseOr(std::unique_ptr<Expr> &expr) {
  SmallVector<std::unique_ptr<Expr>, 8> exprs;
  std::unique_ptr<Expr> andExpr;

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

  expr = std::make_unique<OrExpr>(std::move(exprs));
  return success();
}

LogicalResult Parser::parseAnd(std::unique_ptr<Expr> &expr) {
  SmallVector<std::unique_ptr<ConstraintExpr>, 8> constraints;
  std::unique_ptr<ConstraintExpr> c;
  if (failed(parseConstraint(c)))
    return failure();

  constraints.push_back(std::move(c));

  while (lexer.peek().isa(Token::Kind::And)) {
    lexer.next();
    if (failed(parseConstraint(c)))
      return failure();

    constraints.push_back(std::move(c));
  }

  if (constraints.size() == 1) {
    expr = std::move(constraints[0]);
    return success();
  }

  expr = std::make_unique<AndExpr>(std::move(constraints));
  return success();
}

LogicalResult
Parser::parseConstraint(std::unique_ptr<ConstraintExpr> &constraint) {
  std::unique_ptr<Expr> leftExpr;
  if (failed(parseSum(leftExpr)))
    return failure();

  ConstraintExpr::Kind kind;
  Token cmpToken = lexer.next();
  if (cmpToken.isa(Token::Kind::GreaterEqual))
    kind = ConstraintExpr::Kind::GE;
  else if (cmpToken.isa(Token::Kind::Equal))
    kind = ConstraintExpr::Kind::EQ;
  else if (cmpToken.isa(Token::Kind::LessEqual))
    kind = ConstraintExpr::Kind::LE;
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

  constraint = std::make_unique<ConstraintExpr>(kind, std::move(leftExpr),
                                                std::move(rightExpr));
  return success();
}

LogicalResult Parser::parseSum(std::unique_ptr<Expr> &expr) {
  SmallVector<std::unique_ptr<TermExpr>, 8> terms;
  std::unique_ptr<TermExpr> term;

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

  expr = std::make_unique<SumExpr>(std::move(terms));
  return success();
}

LogicalResult Parser::parseTerm(std::unique_ptr<TermExpr> &term,
                                bool isNegated) {
  std::unique_ptr<IntegerExpr> integer;
  if (lexer.peek().isa(Token::Kind::Minus)) {
    isNegated = !isNegated;
    lexer.next();
  }

  if (lexer.peek().isa(Token::Kind::Integer)) {
    if (failed(parseInteger(integer, isNegated)))
      return failure();
    lexer.consume(Token::Kind::Times);
  } else if (isNegated)
    integer = std::make_unique<IntegerExpr>(-1);

  std::unique_ptr<VariableExpr> identifier;
  if (lexer.peek().isa(Token::Kind::Identifier))
    if (failed(parseVariable(identifier)))
      return failure();

  if (!integer.get() && !identifier.get())
    return emitErrorForToken(lexer.peek(), "expected non empty term");

  term = std::make_unique<TermExpr>(std::move(integer), std::move(identifier));
  return success();
}

LogicalResult Parser::parseVariable(std::unique_ptr<VariableExpr> &vExpr) {
  Token t;
  if (failed(lexer.nextAssertKind(Token::Kind::Identifier, t)))
    return failure();
  vExpr = std::make_unique<VariableExpr>(t.string());
  return success();
}

LogicalResult Parser::parseInteger(std::unique_ptr<IntegerExpr> &iExpr,
                                   bool isNegated) {
  bool negativ = isNegated ^ lexer.peek().isa(Token::Kind::Minus);
  lexer.consume(Token::Kind::Minus);

  Token integerToken;
  if (failed(lexer.nextAssertKind(Token::Kind::Integer, integerToken)))
    return failure();
  int64_t value;
  if (!llvm::to_integer(integerToken.string(), value))
    return emitErrorForToken(integerToken, "expected a valid 64 bit integer");
  if (negativ)
    value = -value;

  iExpr = std::make_unique<IntegerExpr>(value);
  return success();
}

InFlightDiagnostic Parser::emitErrorForToken(Token token,
                                             const Twine &message) {
  return lexer.emitError(token.string().begin(), message);
}

InFlightDiagnostic Parser::emitError(const Twine &message) {
  return lexer.emitErrorAtStart(message);
}

InFlightDiagnostic Parser::emitError(SMLoc loc, const Twine &message) {
  return lexer.emitError(loc, message);
}

//===----------------------------------------------------------------------===//
// PresburgerSetParser
//===----------------------------------------------------------------------===//

PresburgerSetParser::PresburgerSetParser(Parser parser) : parser(parser) {}

LogicalResult
PresburgerSetParser::initVariables(const SmallVector<StringRef, 8> &vars,
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

LogicalResult PresburgerSetParser::parsePresburgerSet(PresburgerSet &set) {

  std::unique_ptr<SetExpr> setExpr;
  if (failed(parser.parseSet(setExpr)))
    return failure();

  if (setExpr->getConstraints() == nullptr) {
    set = PresburgerSet(setExpr->getDims().size(), setExpr->getSyms().size());
    return success();
  }

  initVariables(setExpr->getDims(), dimNameToIndex);
  initVariables(setExpr->getSyms(), symNameToIndex);
  if (failed(parsePresburgerSet(setExpr->getConstraints(), set)))
    return failure();

  return success();
}

LogicalResult PresburgerSetParser::parsePresburgerSet(Expr *constraints,
                                                      PresburgerSet &set) {
  set = PresburgerSet(dimNameToIndex.size(), symNameToIndex.size());
  if (auto orConstraints = constraints->dyn_cast<OrExpr>()) {
    for (std::unique_ptr<Expr> &basicSet : orConstraints->getConstraints()) {
      FlatAffineConstraints cs;
      if (failed(parseFlatAffineConstraints(basicSet.get(), cs)))
        return failure();
      set.addFlatAffineConstraints(cs);
    }
    return success();
  }

  FlatAffineConstraints cs;
  if (failed(parseFlatAffineConstraints(constraints, cs)))
    return failure();

  set.addFlatAffineConstraints(cs);

  return success();
}

LogicalResult
PresburgerSetParser::parseFlatAffineConstraints(Expr *constraints,
                                                FlatAffineConstraints &cs) {
  cs = FlatAffineConstraints(dimNameToIndex.size(), symNameToIndex.size());
  if (constraints->dyn_cast<OrExpr>() != nullptr)
    return emitError("or conditions are not valid for basic sets");

  if (auto constraint = constraints->dyn_cast<ConstraintExpr>()) {
    PresburgerSetParser::Constraint c;
    if (failed(parseConstraint(constraint, c)))
      return failure();
    addConstraint(cs, c);
  } else if (auto andConstraints = constraints->dyn_cast<AndExpr>()) {
    for (std::unique_ptr<ConstraintExpr> &constraint :
         andConstraints->getConstraints()) {
      PresburgerSetParser::Constraint c;
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

LogicalResult
PresburgerSetParser::parseConstraint(ConstraintExpr *constraint,
                                     PresburgerSetParser::Constraint &c) {
  if (constraint == nullptr)
    llvm_unreachable("constraint was nullptr!");

  std::pair<int64_t, SmallVector<int64_t, 8>> left;
  std::pair<int64_t, SmallVector<int64_t, 8>> right;
  if (failed(parseSum(constraint->getLeftSum(), left)) ||
      failed(parseSum(constraint->getRightSum(), right)))
    return failure();

  auto leftConst = left.first;
  auto leftCoeffs = left.second;
  auto rightConst = right.first;
  auto rightCoeffs = right.second;

  int64_t constant;
  SmallVector<int64_t, 8> coeffs;
  if (constraint->getKind() == ConstraintExpr::Kind::LE) {
    constant = rightConst - leftConst;
    for (size_t i = 0; i < leftCoeffs.size(); i++)
      coeffs.push_back(rightCoeffs[i] - leftCoeffs[i]);
  } else if (constraint->getKind() == ConstraintExpr::Kind::GE ||
             constraint->getKind() == ConstraintExpr::Kind::EQ) {
    constant = leftConst - rightConst;
    for (size_t i = 0; i < leftCoeffs.size(); i++)
      coeffs.push_back(leftCoeffs[i] - rightCoeffs[i]);
  } else {
    llvm_unreachable("invalid constraint kind");
  }

  Kind kind;
  if (constraint->getKind() == ConstraintExpr::Kind::EQ)
    kind = Kind::Equality;
  else
    kind = Kind::Inequality;

  coeffs.push_back(constant);
  c = {coeffs, kind};
  return success();
}

LogicalResult
PresburgerSetParser::parseSum(Expr *expr,
                              std::pair<int64_t, SmallVector<int64_t, 8>> &r) {
  int64_t constant = 0;
  SmallVector<int64_t, 8> coeffs(dimNameToIndex.size() + symNameToIndex.size(),
                                 0);
  if (auto *term = expr->dyn_cast<TermExpr>()) {
    if (failed(parseAndAddTerm(term, constant, coeffs)))
      return failure();
  } else if (auto *sum = expr->dyn_cast<SumExpr>()) {
    for (std::unique_ptr<TermExpr> &term : sum->getTerms())
      if (failed(parseAndAddTerm(term.get(), constant, coeffs)))
        return failure();
  }
  r = {constant, coeffs};
  return success();
}

LogicalResult
PresburgerSetParser::parseAndAddTerm(TermExpr *term, int64_t &constant,
                                     SmallVector<int64_t, 8> &coeffs) {
  int64_t delta = 1;
  if (auto coeff = term->getCoeff())
    delta = coeff->getValue();

  auto var = term->getVar();
  if (!var) {
    constant += delta;
    return success();
  }

  auto it = dimNameToIndex.find(var->getName());
  if (it != dimNameToIndex.end()) {
    coeffs[it->second] += delta;
    return success();
  }

  it = symNameToIndex.find(var->getName());
  if (it != symNameToIndex.end()) {
    coeffs[dimNameToIndex.size() + it->second] += delta;
    return success();
  }

  return emitError("encountered unknown variable name: " + var->getName());
}

void PresburgerSetParser::addConstraint(
    FlatAffineConstraints &cs, PresburgerSetParser::Constraint &constraint) {
  if (constraint.second == Kind::Equality)
    cs.addEquality(constraint.first);
  else
    cs.addInequality(constraint.first);
}

InFlightDiagnostic PresburgerSetParser::emitError(const Twine &message) {
  return parser.emitError(message);
}

InFlightDiagnostic PresburgerSetParser::emitError(SMLoc loc,
                                                  const Twine &message) {
  return parser.emitError(loc, message);
}
