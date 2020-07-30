#include "mlir/Analysis/Presburger/Expr.h"
#include "mlir/Analysis/Presburger/Printer.h"

using namespace mlir;

unsigned PresburgerExpr::getNumDims() const { return nDim; }

unsigned PresburgerExpr::getNumSyms() const { return nSym; }

const SmallVector<PresburgerExpr::ExprType, 2> &
PresburgerExpr::getExprs() const {
  return exprs;
}
const SmallVector<PresburgerSet, 2> &PresburgerExpr::getDomains() const {
  return domains;
}

void PresburgerExpr::addPiece(const ExprType &expr,
                              const PresburgerSet &domain) {
  assert(exprs.size() == domains.size() &&
         "cannot have different amount of expressions and domains");
  exprs.push_back(expr);
  domains.push_back(domain);
}

void PresburgerExpr::print(raw_ostream &os) const {
  mlir::printPresburgerExpr(os, *this);
}

void PresburgerExpr::dump() const { print(llvm::errs()); }

llvm::hash_code PresburgerExpr::hash_value() const {
  // TODO
  return llvm::hash_combine(nDim, nSym);
}
