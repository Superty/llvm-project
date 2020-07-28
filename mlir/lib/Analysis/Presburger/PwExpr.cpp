#include "mlir/Analysis/Presburger/PwExpr.h"
#include "mlir/Analysis/Presburger/Printer.h"

using namespace mlir;

unsigned PresburgerPwExpr::getNumDims() const { return nDim; }

unsigned PresburgerPwExpr::getNumSyms() const { return nSym; }

const SmallVector<PresburgerPwExpr::ExprType, 2> &
PresburgerPwExpr::getExprs() const {
  return exprs;
}
const SmallVector<PresburgerSet, 2> &PresburgerPwExpr::getDomains() const {
  return domains;
}

void PresburgerPwExpr::addPiece(const ExprType &expr,
                                const PresburgerSet &domain) {
  assert(exprs.size() == domains.size() &&
         "cannot have different amount of expressions and domains");
  exprs.push_back(expr);
  domains.push_back(domain);
}

void PresburgerPwExpr::print(raw_ostream &os) const {
  PresburgerPrinter::print(os, *this);
}

void PresburgerPwExpr::dump() const { print(llvm::errs()); }

llvm::hash_code PresburgerPwExpr::hash_value() const {
  // TODO
  return llvm::hash_combine(nDim, nSym);
}
