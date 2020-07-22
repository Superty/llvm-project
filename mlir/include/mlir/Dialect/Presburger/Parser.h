/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 * SPDX-FileCopyrightText: 2019 Tobias Grosser <tobias@grosser.es>
 * SPDX-FileCopyrightText: 2019 Arjun Pitchanathan <arjunpitchanathan@gmail.com>
 */

#ifndef PRESBURGER_PARSER_H
#define PRESBURGER_PARSER_H

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Presburger/PwExpr.h"
#include "mlir/Analysis/Presburger/Set.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"

// TODO change to use the MLIR ADTs
#include <memory>

using llvm::SMLoc;
using llvm::StringMap;

namespace mlir {
namespace presburger {
using ErrorCallback = std::function<InFlightDiagnostic(SMLoc, const Twine &)>;

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
    Arrow,
    SemiColon,
    Unknown
  };
  Token() : kind(Kind::Unknown) {}
  explicit Token(Kind kind) : kind(kind) {}
  Token(Kind kind, StringRef content) : kind(kind), content(content) {}

  bool isa(Kind kind);
  StringRef string();
  static std::string name(Token::Kind kind);

private:
  Kind kind;
  StringRef content;
};

class Lexer {
public:
  Lexer(StringRef buffer, ErrorCallback callback);

  Token peek();
  Token next();
  void consume(Token::Kind kind);
  LogicalResult nextAssertKind(Token::Kind kind);
  LogicalResult nextAssertKind(Token::Kind kind, Token &token);
  bool reachedEOF();

  InFlightDiagnostic emitError(const char *loc, const Twine &message = {});
  InFlightDiagnostic emitError(SMLoc loc, const Twine &message = {});
  InFlightDiagnostic emitError(const Twine &message = {});
  InFlightDiagnostic emitErrorAtStart(const Twine &message = {});

private:
  bool isSpace(char c);
  bool isDigit(char c);
  bool isAlpha(char c);
  Token atom(Token::Kind kind, const char *start);
  Token integer(const char *start);
  Token identifierOrKeyword(const char *start);
  Token nextToken();

  Token current;

  StringRef buffer;
  const char *curPtr;
  ErrorCallback callback;
};

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
    PwExpr,
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

  TermExpr &getTerm(size_t position) { return *terms[position]; }
  SmallVector<std::unique_ptr<TermExpr>, 8> &getTerms() { return terms; }

  static Type getStaticType() { return Type::Sum; }
  virtual Type getType() { return Type::Sum; }

private:
  SmallVector<std::unique_ptr<TermExpr>, 8> terms;
};

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

class AndExpr : public Expr {
public:
  explicit AndExpr(SmallVector<std::unique_ptr<ConstraintExpr>, 8> oConstraints)
      : constraints(std::move(oConstraints)) {}

  size_t getNumConstraints() { return constraints.size(); }
  ConstraintExpr &getConstraint(size_t position) {
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

  size_t getNumChildren() { return exprs.size(); }
  SmallVector<std::unique_ptr<Expr>, 8> &getConstraints() { return exprs; }
  Expr &getChild(size_t position) { return *exprs[position]; }

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

class PwExprExpr : public Expr {
public:
  PwExprExpr(SmallVector<StringRef, 8> dims, SmallVector<StringRef, 8> syms)
      : dims(std::move(dims)), syms(std::move(syms)) {}

  SmallVector<StringRef, 8> &getDims() { return dims; }
  SmallVector<StringRef, 8> &getSyms() { return syms; }
  SmallVector<std::unique_ptr<PieceExpr>, 4> &getPieces() { return pieces; }
  PieceExpr *getPieceAt(unsigned i) {
    assert(i < pieces.size() && "out of bounds access");
    return pieces[i].get();
  }

  static Type getStaticType() { return Type::PwExpr; }
  virtual Type getType() { return Type::PwExpr; }

private:
  SmallVector<StringRef, 8> dims;
  SmallVector<StringRef, 8> syms;
  SmallVector<std::unique_ptr<PieceExpr>, 4> pieces;
};

class Parser {
public:
  Parser(StringRef buffer, ErrorCallback callback) : lexer(buffer, callback) {}

  LogicalResult parse(std::unique_ptr<Expr> &expr);
  LogicalResult parseSet(std::unique_ptr<SetExpr> &setExpr);
  LogicalResult parsePwExpr(std::unique_ptr<PwExprExpr> &pwExpr);
  LogicalResult parseCommaSeparatedListUntil(SmallVector<StringRef, 8> &l,
                                             Token::Kind rightToken,
                                             bool allowEmpty);

  LogicalResult parseDimAndOptionalSymbolIdList(
      std::pair<SmallVector<StringRef, 8>, SmallVector<StringRef, 8>>
          &dimSymPair);
  LogicalResult parseOr(std::unique_ptr<Expr> &expr);
  LogicalResult parseAnd(std::unique_ptr<Expr> &expr);
  LogicalResult parseConstraint(std::unique_ptr<ConstraintExpr> &constraint);
  LogicalResult parsePieces(SmallVector<std::unique_ptr<PieceExpr>, 4> &pieces);
  LogicalResult parseSum(std::unique_ptr<Expr> &expr);
  LogicalResult parseTerm(std::unique_ptr<TermExpr> &term,
                          bool is_negated = false);
  LogicalResult parseInteger(std::unique_ptr<IntegerExpr> &iExpr,
                             bool is_negated = false);
  LogicalResult parseVariable(std::unique_ptr<VariableExpr> &vExpr);

  InFlightDiagnostic emitError(const Twine &message = {});
  InFlightDiagnostic emitError(SMLoc loc, const Twine &message = {});
  InFlightDiagnostic emitErrorForToken(Token token, const Twine &message = {});

private:
  Lexer lexer;
};

class PresburgerParser {
public:
  enum class Kind { Equality, Inequality };
  using Constraint = std::pair<SmallVector<int64_t, 8>, Kind>;

  PresburgerParser(Parser parser);

  LogicalResult parsePresburgerPwExpr(PresburgerPwExpr &pwExpr);
  LogicalResult parsePresburgerSet(PresburgerSet &set);

protected:
  LogicalResult parsePresburgerSet(Expr *constraints, PresburgerSet &set);
  LogicalResult parseAndAddPiece(PieceExpr *piece, PresburgerPwExpr &pwExpr);
  LogicalResult parseFlatAffineConstraints(Expr *constraints,
                                           FlatAffineConstraints &cs);
  LogicalResult initVariables(const SmallVector<StringRef, 8> &vars,
                              StringMap<size_t> &map);
  LogicalResult parseConstraint(ConstraintExpr *constraint, Constraint &c);
  LogicalResult parseSum(Expr *expr,
                         std::pair<int64_t, SmallVector<int64_t, 8>> &r);
  LogicalResult parseAndAddTerm(TermExpr *term, int64_t &constant,
                                SmallVector<int64_t, 8> &coeffs);
  void addConstraint(FlatAffineConstraints &cs, Constraint &constraint);
  InFlightDiagnostic emitError(SMLoc loc, const Twine &message = {});
  InFlightDiagnostic emitError(const Twine &message = {});

  StringMap<size_t> dimNameToIndex;
  StringMap<size_t> symNameToIndex;
  Parser parser;
};

}; // namespace presburger
}; // namespace mlir

#endif // PRESBURGER_PARSER_H
