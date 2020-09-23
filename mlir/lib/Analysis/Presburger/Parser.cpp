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

/// This is a baseclass for all AST nodes for the Presburger constructs.
class Expr {
public:
  enum class Type {
    Integer,
    Variable,
    Term,
    Sum,
    And,
    Or,
    Constraint,
    Set,
    None
  };

  template <class T>
  T *dyn_cast() {
    if (T::getStaticType() == getType())
      return (T *)this;

    return nullptr;
  }

  static Type getStaticType() { return Type::None; }
  virtual Type getType() { return Type::None; }
  virtual ~Expr() = default;
};

class IntegerExpr : public Expr {
public:
  explicit IntegerExpr(int64_t value) : value(value) {}

  int64_t getValue() { return value; }

  static Type getStaticType() { return Type::Integer; }
  virtual Type getType() { return Type::Integer; }

private:
  int64_t value;
};

class VariableExpr : public Expr {
public:
  explicit VariableExpr(StringRef name) : name(name) {}

  StringRef getName() { return name; }

  static Type getStaticType() { return Type::Variable; }
  virtual Type getType() { return Type::Variable; }

private:
  StringRef name;
};

class TermExpr : public Expr {
public:
  TermExpr(std::unique_ptr<IntegerExpr> oCoeff,
           std::unique_ptr<VariableExpr> oVar)
      : coeff(std::move(oCoeff)), var(std::move(oVar)) {}

  IntegerExpr *getCoeff() { return coeff.get(); }
  VariableExpr *getVar() { return var.get(); }

  static Type getStaticType() { return Type::Term; }
  virtual Type getType() { return Type::Term; }

private:
  std::unique_ptr<IntegerExpr> coeff;
  std::unique_ptr<VariableExpr> var;
};

class SumExpr : public Expr {
public:
  explicit SumExpr(SmallVector<std::unique_ptr<TermExpr>, 8> oTerms)
      : terms(std::move(oTerms)) {}

  TermExpr &getTerm(unsigned position) { return *terms[position]; }
  SmallVector<std::unique_ptr<TermExpr>, 8> &getTerms() { return terms; }

  static Type getStaticType() { return Type::Sum; }
  virtual Type getType() { return Type::Sum; }

private:
  SmallVector<std::unique_ptr<TermExpr>, 8> terms;
};

class ConstraintExpr : public Expr {
public:
  enum class Kind { LE, GE, EQ };

  ConstraintExpr(Kind oKind, std::unique_ptr<Expr> oLeftSum,
                 std::unique_ptr<Expr> oRightSum)
      : kind(oKind), leftSum(std::move(oLeftSum)),
        rightSum(std::move(oRightSum)) {}

  Expr *getLeftSum() { return leftSum.get(); }
  Expr *getRightSum() { return rightSum.get(); }

  static Type getStaticType() { return Type::Constraint; }
  virtual Type getType() { return Type::Constraint; }

  Kind getKind() { return kind; }

private:
  Kind kind;
  std::unique_ptr<Expr> leftSum;
  std::unique_ptr<Expr> rightSum;
};

class AndExpr : public Expr {
public:
  explicit AndExpr(SmallVector<std::unique_ptr<ConstraintExpr>, 8> oConstraints)
      : constraints(std::move(oConstraints)) {}

  unsigned getNumConstraints() { return constraints.size(); }
  ConstraintExpr &getConstraint(unsigned position) {
    return *constraints[position];
  }
  SmallVector<std::unique_ptr<ConstraintExpr>, 8> &getConstraints() {
    return constraints;
  }

  static Type getStaticType() { return Type::And; }
  virtual Type getType() { return Type::And; }

private:
  SmallVector<std::unique_ptr<ConstraintExpr>, 8> constraints;
};

class OrExpr : public Expr {
public:
  explicit OrExpr(SmallVector<std::unique_ptr<Expr>, 8> oExprs)
      : exprs(std::move(oExprs)) {}

  unsigned getNumChildren() { return exprs.size(); }
  SmallVector<std::unique_ptr<Expr>, 8> &getConstraints() { return exprs; }
  Expr &getChild(unsigned position) { return *exprs[position]; }

  static Type getStaticType() { return Type::Or; }
  virtual Type getType() { return Type::Or; }

private:
  SmallVector<std::unique_ptr<Expr>, 8> exprs;
};

class SetExpr : public Expr {
public:
  SetExpr(SmallVector<StringRef, 8> dims, SmallVector<StringRef, 8> syms,
          std::unique_ptr<Expr> oConstraints)
      : dims(std::move(dims)), syms(std::move(syms)),
        constraints(std::move(oConstraints)) {}

  SmallVector<StringRef, 8> &getDims() { return dims; }
  SmallVector<StringRef, 8> &getSyms() { return syms; }
  Expr *getConstraints() { return constraints.get(); }

  static Type getStaticType() { return Type::Set; }
  virtual Type getType() { return Type::Set; }

private:
  SmallVector<StringRef, 8> dims;
  SmallVector<StringRef, 8> syms;
  std::unique_ptr<Expr> constraints;
};

/// Uses the Lexer to transform a token stream into an AST representing
/// different Presburger constructs.
class PresburgerParser : public Parser {
public:
  // TODO this needs a rewrite
  enum class Kind { Equality, Inequality };
  using Constraint = std::pair<SmallVector<int64_t, 8>, Kind>;

  PresburgerParser(ParserState &state) : Parser(state){};

  /// Parse a Presburger set into set
  LogicalResult parsePresburgerSet(PresburgerSet &set);

  // TODO should this be public
  /// Parse a Presburger set and returns an AST corresponding to it.
  LogicalResult parseSet(std::unique_ptr<SetExpr> &setExpr);

private:
  // TODO some of these functions should not be called parse. Either we split
  // this up, or rename them

  // Helpers to transform the AST into a PresburgerSet
  LogicalResult parsePresburgerSet(Expr *constraints, PresburgerSet &set);
  LogicalResult parseFlatAffineConstraints(Expr *constraints,
                                           FlatAffineConstraints &cs);
  LogicalResult initVariables(const SmallVector<StringRef, 8> &vars,
                              StringMap<unsigned> &map);
  LogicalResult parseConstraint(ConstraintExpr *constraint, Constraint &c);
  LogicalResult parseSum(Expr *expr,
                         std::pair<int64_t, SmallVector<int64_t, 8>> &r);
  LogicalResult parseAndAddTerm(TermExpr *term, int64_t &constant,
                                SmallVector<int64_t, 8> &coeffs);
  void addConstraint(FlatAffineConstraints &cs, Constraint &constraint);

  StringMap<unsigned> dimNameToIndex;
  StringMap<unsigned> symNameToIndex;

  // Helpers for the AST generation
  LogicalResult parseDimAndOptionalSymbolIdList(
      std::pair<SmallVector<StringRef, 8>, SmallVector<StringRef, 8>>
          &dimSymPair);

  LogicalResult parseOr(std::unique_ptr<Expr> &expr);
  LogicalResult parseAnd(std::unique_ptr<Expr> &expr);

  LogicalResult parseConstraint(std::unique_ptr<ConstraintExpr> &constraint);
  LogicalResult parseSum(std::unique_ptr<Expr> &expr);
  LogicalResult parseTerm(std::unique_ptr<TermExpr> &term,
                          bool is_negated = false);
  LogicalResult parseInteger(std::unique_ptr<IntegerExpr> &iExpr,
                             bool is_negated = false);
  LogicalResult parseVariable(std::unique_ptr<VariableExpr> &vExpr);
};

//===----------------------------------------------------------------------===//
// PresburgerParser
//===----------------------------------------------------------------------===//

/// initializes a name to id mapping for variables
LogicalResult
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

/// Parse a Presburger set into set
LogicalResult PresburgerParser::parsePresburgerSet(PresburgerSet &set) {

  std::unique_ptr<SetExpr> setExpr;
  if (failed(parseSet(setExpr)))
    return failure();

  if (setExpr->getConstraints() == nullptr) {
    set = PresburgerSet::getUniverse(setExpr->getDims().size(),
                                     setExpr->getSyms().size());
    return success();
  }

  initVariables(setExpr->getDims(), dimNameToIndex);
  initVariables(setExpr->getSyms(), symNameToIndex);
  if (failed(parsePresburgerSet(setExpr->getConstraints(), set)))
    return failure();

  return success();
}

/// Creates a PresburgerSet instance from constraints
///
/// For each AndExpr contained in constraints it creates one
/// FlatAffineConstraints object
LogicalResult PresburgerParser::parsePresburgerSet(Expr *constraints,
                                                   PresburgerSet &set) {
  set =
      PresburgerSet::getUniverse(dimNameToIndex.size(), symNameToIndex.size());
  if (auto orConstraints = constraints->dyn_cast<OrExpr>()) {
    for (std::unique_ptr<Expr> &basicSet : orConstraints->getConstraints()) {
      FlatAffineConstraints cs;
      if (failed(parseFlatAffineConstraints(basicSet.get(), cs)))
        return failure();
      set.unionFACInPlace(cs);
    }
    return success();
  }

  FlatAffineConstraints cs;
  if (failed(parseFlatAffineConstraints(constraints, cs)))
    return failure();

  set.unionFACInPlace(cs);

  return success();
}

/// Creates a FlatAffineConstraint instance from constraints
///
/// Expects either a single ConstraintExpr or multiple of them combined in an
/// AndExpr
LogicalResult
PresburgerParser::parseFlatAffineConstraints(Expr *constraints,
                                             FlatAffineConstraints &cs) {
  cs = FlatAffineConstraints(dimNameToIndex.size(), symNameToIndex.size());
  if (constraints->dyn_cast<OrExpr>() != nullptr)
    return emitError("or conditions are not valid for basic sets");

  if (auto constraint = constraints->dyn_cast<ConstraintExpr>()) {
    PresburgerParser::Constraint c;
    if (failed(parseConstraint(constraint, c)))
      return failure();
    addConstraint(cs, c);
  } else if (auto andConstraints = constraints->dyn_cast<AndExpr>()) {
    for (std::unique_ptr<ConstraintExpr> &constraint :
         andConstraints->getConstraints()) {
      PresburgerParser::Constraint c;
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

/// Creates a constraint from a ConstraintExpr.
///
/// It either returns an equality (== 0) or an inequalitiy (>= 0).
/// As a ConstraintExpr contains sums on both sides, they are subtracted from
/// each other to get the desired form.
LogicalResult
PresburgerParser::parseConstraint(ConstraintExpr *constraint,
                                  PresburgerParser::Constraint &c) {
  assert(constraint != nullptr && "constraint was nullptr!");

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

  switch (constraint->getKind()) {
  case ConstraintExpr::Kind::LE:
    constant = rightConst - leftConst;
    for (unsigned i = 0; i < leftCoeffs.size(); i++)
      coeffs.push_back(rightCoeffs[i] - leftCoeffs[i]);
    break;

  case ConstraintExpr::Kind::GE:
  case ConstraintExpr::Kind::EQ:
    constant = leftConst - rightConst;
    for (unsigned i = 0; i < leftCoeffs.size(); i++)
      coeffs.push_back(leftCoeffs[i] - rightCoeffs[i]);
    break;
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

/// Creates a list of coefficients and a constant from a SumExpr or a
/// TermExpr.
///
/// The list of coefficients corresponds to the coefficients of the dimensions
/// and after that the symbols.
///
LogicalResult
PresburgerParser::parseSum(Expr *expr,
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
  // TODO add a struct for this?
  r = {constant, coeffs};
  return success();
}

/// Takes a TermExpr and addapts the matching coefficient or the constant to
/// this terms value. To determine the coefficient id it looks it up in the
/// nameToIndex mappings
///
/// Fails if the variable name is unknown
LogicalResult
PresburgerParser::parseAndAddTerm(TermExpr *term, int64_t &constant,
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

void PresburgerParser::addConstraint(FlatAffineConstraints &cs,
                                     PresburgerParser::Constraint &constraint) {
  if (constraint.second == Kind::Equality)
    cs.addEquality(constraint.first);
  else
    cs.addInequality(constraint.first);
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
LogicalResult PresburgerParser::parseSet(std::unique_ptr<SetExpr> &setExpr) {
  std::pair<SmallVector<StringRef, 8>, SmallVector<StringRef, 8>> dimSymPair;
  // TODO merge dimSymPair with the id list
  if (failed(parseDimAndOptionalSymbolIdList(dimSymPair)))
    return failure();

  if (parseToken(Token::colon, "expected ':'") ||
      parseToken(Token::l_paren, "expected '('"))
    return failure();

  if (consumeIf(Token::r_paren)) {
    setExpr = std::make_unique<SetExpr>(std::move(dimSymPair.first),
                                        std::move(dimSymPair.second), nullptr);
    // TODO
    // checks that we are at the end of the string
    // if (lexer.reachedEOF())
    //  return success();
    // return emitErrorForToken(lexer.peek(),
    //                         "expected to be at the end of the set");
    return success();
  }

  std::unique_ptr<Expr> constraints;
  if (failed(parseOr(constraints)))
    return failure();

  if (parseToken(Token::r_paren, "expected ')'"))
    return failure();

  setExpr = std::make_unique<SetExpr>(std::move(dimSymPair.first),
                                      std::move(dimSymPair.second),
                                      std::move(constraints));
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
/// TODO can we reuse parts of the AffineParser
LogicalResult PresburgerParser::parseDimAndOptionalSymbolIdList(
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
LogicalResult PresburgerParser::parseOr(std::unique_ptr<Expr> &expr) {
  SmallVector<std::unique_ptr<Expr>, 8> exprs;
  std::unique_ptr<Expr> andExpr;

  if (failed(parseAnd(andExpr)))
    return failure();
  exprs.push_back(std::move(andExpr));
  while (consumeIf(Token::kw_or)) {
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

/// Parse an and expression.
///
///  pb-and-expr ::= pb-constraint (`and` pb-constraint)*
///
LogicalResult PresburgerParser::parseAnd(std::unique_ptr<Expr> &expr) {
  SmallVector<std::unique_ptr<ConstraintExpr>, 8> constraints;
  std::unique_ptr<ConstraintExpr> c;
  if (failed(parseConstraint(c)))
    return failure();

  constraints.push_back(std::move(c));

  while (consumeIf(Token::kw_and)) {
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

/// Parse a Presburger constraint
///
///  pb-constraint ::= pb-expr (`>=` | `=` | `<=`) pb-expr
///
LogicalResult
PresburgerParser::parseConstraint(std::unique_ptr<ConstraintExpr> &constraint) {
  std::unique_ptr<Expr> leftExpr;
  if (failed(parseSum(leftExpr)))
    return failure();

  ConstraintExpr::Kind kind;
  switch (getToken().getKind()) {
  case Token::greater:
    consumeToken(Token::greater);
    if (consumeIf(Token::equal)) {
      kind = ConstraintExpr::Kind::GE;
      break;
    }
    return emitError("strict inequalities are not supported");
  case Token::equal:
    consumeToken(Token::equal);
    kind = ConstraintExpr::Kind::EQ;
    break;
  case Token::less:
    consumeToken(Token::less);
    if (consumeIf(Token::equal)) {
      kind = ConstraintExpr::Kind::LE;
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

  std::unique_ptr<Expr> rightExpr;
  if (failed(parseSum(rightExpr)))
    return failure();

  constraint = std::make_unique<ConstraintExpr>(kind, std::move(leftExpr),
                                                std::move(rightExpr));
  return success();
}

/// Parse a Presburger sum.
///
///  pb-sum ::= pb-term (('+' | '-') pb-term)*
///
LogicalResult PresburgerParser::parseSum(std::unique_ptr<Expr> &expr) {
  SmallVector<std::unique_ptr<TermExpr>, 8> terms;
  std::unique_ptr<TermExpr> term;

  if (failed(parseTerm(term)))
    return failure();
  terms.push_back(std::move(term));

  while (getToken().isAny(Token::plus, Token::minus)) {
    bool isMinus = getToken().is(Token::minus);
    consumeToken();

    if (failed(parseTerm(term, isMinus)))
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

/// Parse a Presburger term.
///
///  pb-term       ::= '-'? pb-int? pb-var
///                ::= '-'? pb-int
///
LogicalResult PresburgerParser::parseTerm(std::unique_ptr<TermExpr> &term,
                                          bool isNegated) {
  std::unique_ptr<IntegerExpr> integer;
  if (consumeIf(Token::minus))
    isNegated = !isNegated;

  if (getToken().is(Token::integer)) {
    if (failed(parseInteger(integer, isNegated)))
      return failure();
    consumeIf(Token::star);
  } else if (isNegated) {
    integer = std::make_unique<IntegerExpr>(-1);
  }

  std::unique_ptr<VariableExpr> identifier;
  if (getToken().is(Token::bare_identifier))
    if (failed(parseVariable(identifier)))
      return failure();

  if (!integer.get() && !identifier.get())
    return emitError("expected non empty term");

  term = std::make_unique<TermExpr>(std::move(integer), std::move(identifier));
  return success();
}

/// Parse a variable.
///
///  pb-var ::= letter (digit | letter)*
///
LogicalResult
PresburgerParser::parseVariable(std::unique_ptr<VariableExpr> &vExpr) {
  if (getToken().isNot(Token::bare_identifier))
    return failure();
  vExpr = std::make_unique<VariableExpr>(getTokenSpelling());
  consumeToken();
  return success();
}

/// Parse a signless integer.
///
///  pb-int ::= digit+
///
LogicalResult
PresburgerParser::parseInteger(std::unique_ptr<IntegerExpr> &iExpr,
                               bool isNegated) {
  bool negativ = isNegated ^ consumeIf(Token::minus);

  if (getToken().isNot(Token::integer))
    return failure();
  auto value = getToken().getUInt64IntegerValue();
  if (!value.hasValue() || (int64_t)value.getValue() < 0)
    return emitError("constant too large for index");

  consumeToken(Token::integer);
  iExpr = std::make_unique<IntegerExpr>(negativ ? -value.getValue()
                                                : value.getValue());
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
  if (failed(parser.parsePresburgerSet(set)))
    return failure();
  return set;
}
