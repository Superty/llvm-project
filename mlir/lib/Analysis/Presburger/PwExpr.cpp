#include "mlir/Analysis/Presburger/PwExpr.h"

using namespace mlir;

unsigned PresburgerPwExpr::getNumDims() const { return nDim; }

unsigned PresburgerPwExpr::getNumSyms() const { return nSym; }

void PresburgerPwExpr::addPiece(const ExprType &expr,
                                const PresburgerSet &domain) {
  assert(exprs.size() == domains.size() &&
         "cannot have different amount of expressions and domains");
  exprs.push_back(expr);
  domains.push_back(domain);
}

void PresburgerPwExpr::print(raw_ostream &os) const {
  assert(exprs.size() == domains.size() &&
         "cannot have different amount of expressions and domains");
  printVariableList(os);

  os << " -> ";

  printExpr(exprs[0], os);
  os << " : ";
  domains[0].printConstraints(os);
  for (unsigned i = 1, e = exprs.size(); i < e; ++i) {
    os << " ; ";
    printExpr(exprs[i], os);
    os << " : ";
    domains[i].printConstraints(os);
  }
}

void PresburgerPwExpr::dump() const { print(llvm::errs()); }

void PresburgerPwExpr::printVar(raw_ostream &os, int64_t val, unsigned i,
                                unsigned &countNonZero) const {
  if (val == 0) {
    return;
  } else if (val > 0) {
    if (countNonZero > 0) {
      os << " + ";
    }
    if (val > 1)
      os << val;
  } else {
    if (countNonZero > 0) {
      os << " - ";
      if (val != -1)
        os << -val;
    } else {
      if (val == -1)
        os << "-";
      else
        os << val;
    }
  }

  if (i < getNumDims()) {
    os << 'd' << i;
  } else if (i < getNumDims() + getNumSyms()) {
    os << 's' << (i - getNumDims());
  }
  countNonZero++;
}

void PresburgerPwExpr::printExpr(ExprType expr, raw_ostream &os) const {
  os << "(";
  unsigned countNonZero = 0;
  for (unsigned i = 0, e = expr.second.size(); i < e; ++i) {
    printVar(os, expr.second[i], i, countNonZero);
  }
  int64_t c = expr.first;
  if (countNonZero > 0) {
    if (c > 0)
      os << " + " << c;
    else if (c < 0)
      os << " - " << -c;
  } else {
    os << c;
  }

  os << ")";
}

// TODO somehow merge this with PresburgerSet
void PresburgerPwExpr::printVariableList(raw_ostream &os) const {
  assert(!domains.empty() && "pwExpr needs atleast one piece");
  domains[0].printVariableList(os);
}

llvm::hash_code PresburgerPwExpr::hash_value() const {
  // TODO
  return llvm::hash_combine(nDim, nSym);
}
