/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 * SPDX-FileCopyrightText: 2019 Tobias Grosser <tobias@grosser.es>
 * SPDX-FileCopyrightText: 2019 Arjun Pitchanathan <arjunpitchanathan@gmail.com>
 */

#ifndef PRESBURGER_PARSER_H
#define PRESBURGER_PARSER_H

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Presburger/Expr.h"
#include "mlir/Analysis/Presburger/Set.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"

using llvm::SMLoc;
using llvm::StringMap;

namespace mlir {
namespace presburger {
using ErrorCallback = std::function<InFlightDiagnostic(SMLoc, const Twine &)>;

/// This class is used by the following Lexer to represent the lexical tokens
/// produced by it.
///
/// Stores a StringRef with the actual content of the token. This gives
/// additionally a location that enables easier error reporting.
class Token {
public:
  enum class Kind {
    Integer,
    Identifier,
    LeftParen,
    RightParen,
    LeftSquare,
    RightSquare,
    LeftCurly,
    RightCurly,
    LessThan,
    LessEqual,
    GreaterThan,
    GreaterEqual,
    Equal,
    NotEqual,
    Plus,
    Minus,
    Times,
    Divide,
    Modulo,
    Colon,
    Comma,
    And,
    Or,
    Empty,
    Exists,
    Arrow,
    Semicolon,
    Eof,
    Unknown
  };
  Token() : kind(Kind::Unknown) {}
  explicit Token(Kind kind) : kind(kind) {}
  Token(Kind kind, StringRef content) : kind(kind), content(content) {}

  /// Returns true if the token is of Kind kind.
  bool isa(Kind kind);

  /// Returns the StringRef corresponding to the content.
  StringRef string();

  /// Returns a StringRef with a name for the provided kind.
  static StringRef name(Token::Kind kind);

private:
  Kind kind;
  StringRef content;
};

/// Splits the buffer into Tokens and is able to report errors at certain
/// location by using the provided ErrorCallback.
class Lexer {
public:
  Lexer(StringRef buffer, ErrorCallback callback);

  /// Returns the next token without consuming it.
  Token peek();

  /// Returns the next token and consumes it.
  Token next();

  /// Consumes a token with specified kind. If the next token is not of the
  /// specified kind it is a no-op.
  void consume(Token::Kind kind);

  /// Consumes the next token and emits an error if it doesn't match the
  /// provided kind.
  LogicalResult consumeKindOrError(Token::Kind kind);
  LogicalResult consumeKindOrError(Token::Kind kind, Token &token);

  /// Returns true if the end of the buffer is reached.
  bool reachedEOF();

  /// Emits the provided error message at the location provided.
  InFlightDiagnostic emitError(const char *loc, const Twine &message = {});
  InFlightDiagnostic emitError(SMLoc loc, const Twine &message = {});

  /// Emits the provided error message at the current possition of the Lexer.
  InFlightDiagnostic emitError(const Twine &message = {});

  /// Emits the provided error message at the start of the buffer.
  InFlightDiagnostic emitErrorAtStart(const Twine &message = {});

private:
  bool isSpace(char c);
  bool isDigit(char c);
  bool isAlpha(char c);

  /// Returns a Token from start to curPtr
  Token getAtom(Token::Kind kind, unsigned start);

  /// Creates an integer token while consuming all digits
  Token consumeInteger(unsigned start);

  /// Create an identifier or keyword. An identifier has to start with an
  /// alphabetic char and after that contains a sequence of alphanumeric chars.
  Token consumeIdentifierOrKeyword(unsigned start);

  /// Determines the next token and consumes it.
  Token nextToken();

  /// Holds the current token.
  Token current;

  /// The buffer containing the string to lex.
  StringRef buffer;

  /// The current possition in the buffer.
  unsigned curPos;

  /// An error callback function. This is required as otherwise exact error
  /// messages aren't possible.
  ErrorCallback callback;
};

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
    Piece,
    PresburgerExpr,
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

template <typename Int>
class IntegerExpr : public Expr {
public:
  explicit IntegerExpr(Int value) : value(value) {}

  Int getValue() { return value; }

  static Type getStaticType() { return Type::Integer; }
  virtual Type getType() { return Type::Integer; }

private:
  Int value;
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

template <typename Int>
class TermExpr : public Expr {
public:
  TermExpr(std::unique_ptr<IntegerExpr<Int>> oCoeff,
           std::unique_ptr<VariableExpr> oVar)
      : coeff(std::move(oCoeff)), var(std::move(oVar)) {}

  IntegerExpr<Int> *getCoeff() { return coeff.get(); }
  VariableExpr *getVar() { return var.get(); }

  static Type getStaticType() { return Type::Term; }
  virtual Type getType() { return Type::Term; }

private:
  std::unique_ptr<IntegerExpr<Int>> coeff;
  std::unique_ptr<VariableExpr> var;
};

template <typename Int>
class SumExpr : public Expr {
public:
  explicit SumExpr(SmallVector<std::unique_ptr<TermExpr<Int>>, 8> oTerms)
      : terms(std::move(oTerms)) {}

  TermExpr<Int> &getTerm(size_t position) { return *terms[position]; }
  SmallVector<std::unique_ptr<TermExpr<Int>>, 8> &getTerms() { return terms; }

  static Type getStaticType() { return Type::Sum; }
  virtual Type getType() { return Type::Sum; }

private:
  SmallVector<std::unique_ptr<TermExpr<Int>>, 8> terms;
};

template <typename Int>
class ConstraintExpr : public Expr {
public:
  enum class Kind { LT, LE, GT, GE, EQ };

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

template <typename Int>
struct DivExpr {
  explicit DivExpr(std::unique_ptr<Expr> oNum,
                   std::unique_ptr<IntegerExpr<Int>> oDen)
      : num(std::move(oNum)), den(std::move(oDen)) {}
  std::unique_ptr<Expr> num;
  std::unique_ptr<IntegerExpr<Int>> den;
};

template <typename Int>
class AndExpr : public Expr {
public:
  explicit AndExpr(SmallVector<std::unique_ptr<ConstraintExpr<Int>>, 8> oConstraints,
                   SmallVector<StringRef, 8> oExists,
                   SmallVector<StringRef, 8> oDivNames,
                   SmallVector<std::unique_ptr<DivExpr<Int>>, 8> oDivs)
      : constraints(std::move(oConstraints)), exists(std::move(oExists)),
        divNames(std::move(oDivNames)), divs(std::move(oDivs)) {}

  size_t getNumConstraints() { return constraints.size(); }
  ConstraintExpr<Int> &getConstraint(size_t position) {
    return *constraints[position];
  }
  SmallVector<std::unique_ptr<ConstraintExpr<Int>>, 8> &getConstraints() {
    return constraints;
  }
  SmallVector<StringRef, 8> &getExists() { return exists; }
  SmallVector<std::unique_ptr<DivExpr<Int>>, 8> &getDivs() { return divs; }
  SmallVector<StringRef, 8> &getDivNames() { return divNames; }

  static Type getStaticType() { return Type::And; }
  virtual Type getType() { return Type::And; }

private:
  SmallVector<std::unique_ptr<ConstraintExpr<Int>>, 8> constraints;
  SmallVector<StringRef, 8> exists;
  SmallVector<StringRef, 8> divNames;
  SmallVector<std::unique_ptr<DivExpr<Int>>, 8> divs;
};

template <typename Int>
class OrExpr : public Expr {
public:
  explicit OrExpr(SmallVector<std::unique_ptr<Expr>, 8> oExprs)
      : exprs(std::move(oExprs)) {}

  size_t getNumChildren() { return exprs.size(); }
  SmallVector<std::unique_ptr<Expr>, 8> &getConstraints() { return exprs; }
  Expr &getChild(size_t position) { return *exprs[position]; }

  static Type getStaticType() { return Type::Or; }
  virtual Type getType() { return Type::Or; }

private:
  SmallVector<std::unique_ptr<Expr>, 8> exprs;
};

template <typename Int>
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

class PieceExpr : public Expr {
public:
  PieceExpr(std::unique_ptr<Expr> expr, std::unique_ptr<Expr> constraints)
      : expr(std::move(expr)), constraints(std::move(constraints)) {}

  Expr *getExpr() { return expr.get(); }
  Expr *getConstraints() { return constraints.get(); }

  static Type getStaticType() { return Type::Piece; }
  virtual Type getType() { return Type::Piece; }

private:
  std::unique_ptr<Expr> expr;
  std::unique_ptr<Expr> constraints;
};

class PresburgerExprExpr : public Expr {
public:
  PresburgerExprExpr(SmallVector<StringRef, 8> dims,
                     SmallVector<StringRef, 8> syms,
                     SmallVector<std::unique_ptr<PieceExpr>, 4> pieces)
      : dims(std::move(dims)), syms(std::move(syms)),
        pieces(std::move(pieces)) {}

  SmallVector<StringRef, 8> &getDims() { return dims; }
  SmallVector<StringRef, 8> &getSyms() { return syms; }
  SmallVector<std::unique_ptr<PieceExpr>, 4> &getPieces() { return pieces; }
  PieceExpr &getPieceAt(size_t i) { return *pieces[i]; }

  static Type getStaticType() { return Type::PresburgerExpr; }
  virtual Type getType() { return Type::PresburgerExpr; }

private:
  SmallVector<StringRef, 8> dims;
  SmallVector<StringRef, 8> syms;
  SmallVector<std::unique_ptr<PieceExpr>, 4> pieces;
};

/// Uses the Lexer to transform a token stream into an AST representing
/// different Presburger constructs.
template <typename Int>
class Parser {
public:
  Parser(StringRef buffer, ErrorCallback callback) : lexer(buffer, callback) {}

  /// Parse a Presburger set and returns an AST corresponding to it.
  LogicalResult parseSet(std::unique_ptr<SetExpr<Int>> &setExpr);

  /// Parse a Presburger expression and returns an AST corresponding
  /// to it.
  LogicalResult parseExpr(std::unique_ptr<PresburgerExprExpr> &expr);

  /// Parse a keyword that starts with a letter
  LogicalResult parseKeyword(StringRef &keyword);

  /// Emits the provided error message at the start of the buffer.
  InFlightDiagnostic emitError(const Twine &message = {});

  /// Emits the provided error message at the location provided.
  InFlightDiagnostic emitError(SMLoc loc, const Twine &message = {});

  /// Emits the provided error message at the start of the token.
  InFlightDiagnostic emitErrorForToken(Token token, const Twine &message = {});

private:
  // Helpers for the parsing

  LogicalResult parseCommaSeparatedListUntil(SmallVector<StringRef, 8> &l,
                                             Token::Kind rightToken,
                                             bool allowEmpty);
  LogicalResult parseDimAndSymbolIdLists(
      std::pair<SmallVector<StringRef, 8>, SmallVector<StringRef, 8>>
          &dimSymPair);

  LogicalResult parseOr(std::unique_ptr<Expr> &expr);
  LogicalResult parseAnd(std::unique_ptr<Expr> &expr);

  LogicalResult parseConstraint(std::unique_ptr<ConstraintExpr<Int>> &constraint);
  LogicalResult parsePiece(std::unique_ptr<PieceExpr> &piece);
  LogicalResult parseSum(std::unique_ptr<Expr> &expr);
  LogicalResult parseTerm(std::unique_ptr<TermExpr<Int>> &term,
                          bool is_negated = false);
  LogicalResult parseInteger(std::unique_ptr<IntegerExpr<Int>> &iExpr,
                             bool is_negated = false);
  LogicalResult parseVariable(std::unique_ptr<VariableExpr> &vExpr);

  Lexer lexer;
};

/// This class uses the Parser class to parse Presburger constructs. It uses the
/// Parser class to generate an AST and then transforms this AST to the
/// according datastructures.
///
/// At the moment it expects a Parser instance as this parser is already used to
/// determine the kind of object to parse. TODO change this
template <typename Int>
class PresburgerParser {
public:
  enum class Kind { Equality, Inequality };
  using Constraint = std::pair<SmallVector<Int, 8>, Kind>;

  PresburgerParser(Parser<Int> parser);

  /// Parse a Presburger expression into expr
  LogicalResult parsePresburgerExpr(PresburgerExpr &expr);

  /// Parse a Presburger set into set
  LogicalResult parsePresburgerSet(PresburgerSet<Int> &set);

private:
  // parsing helpers
  LogicalResult parsePresburgerSet(Expr *constraints, PresburgerSet<Int> &set);
  LogicalResult parseAndAddPiece(PieceExpr *piece, PresburgerExpr &expr);
  LogicalResult parsePresburgerBasicSet(Expr *constraints,
                                        PresburgerBasicSet<Int> &bs);
  LogicalResult initVariables(const SmallVector<StringRef, 8> &vars,
                              StringMap<size_t> &map);
  LogicalResult parseConstraint(ConstraintExpr<Int> *constraint, Constraint &c);
  LogicalResult
  parseSum(Expr *expr, std::pair<Int, SmallVector<Int, 8>> &r);
  LogicalResult parseAndAddTerm(TermExpr<Int> *term, Int &constant,
                                SmallVector<Int, 8> &coeffs);
  void addConstraint(PresburgerBasicSet<Int> &bs, Constraint &constraint);
  InFlightDiagnostic emitError(SMLoc loc, const Twine &message = {});
  InFlightDiagnostic emitError(const Twine &message = {});

  StringMap<size_t> dimNameToIndex;
  StringMap<size_t> symNameToIndex;
  StringMap<size_t> existNameToIndex;
  StringMap<size_t> divNameToIndex;
  Parser<Int> parser;
};

using TransprecParser = Parser<DefaultInt>;
using TransprecPresburgerParser = PresburgerParser<DefaultInt>;

} // namespace presburger
} // namespace mlir

#endif // PRESBURGER_PARSER_H
