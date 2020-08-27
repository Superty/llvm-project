#include "mlir/Analysis/Presburger/Printer.h"

using namespace mlir;
using namespace analysis::presburger;

namespace {

void printFlatAffineConstraints(raw_ostream &os,
                                const FlatAffineConstraints &cs);

void printConstraints(
    raw_ostream &os,
    const SmallVectorImpl<FlatAffineConstraints> &flatAffineConstraints);
void printVariableList(raw_ostream &os, unsigned nDim, unsigned nSym);
void printExpr(raw_ostream &os, ArrayRef<int64_t> coeffs, int64_t constant,
               unsigned nDim);
bool printCoeff(raw_ostream &os, int64_t val, bool first);
void printVarName(raw_ostream &os, int64_t i, unsigned nDim);
void printConst(raw_ostream &os, int64_t c, bool first);

/// Prints the '(d0, ..., dN)[s0, ... ,sM]' dimension and symbol list.
///
void printVariableList(raw_ostream &os, unsigned nDim, unsigned nSym) {
  os << "(";
  for (unsigned i = 0; i < nDim; i++)
    os << (i != 0 ? ", " : "") << 'd' << i;
  os << ")";

  if (nSym > 0) {
    os << "[";
    for (unsigned i = 0; i < nSym; i++)
      os << (i != 0 ? ", " : "") << 's' << i;
    os << "]";
  }
}

/// Prints the constraints of each `FlatAffineConstraints`.
///
void printConstraints(
    raw_ostream &os,
    const SmallVectorImpl<FlatAffineConstraints> &flatAffineConstraints) {
  os << "(";
  bool fst = true;
  for (auto &c : flatAffineConstraints) {
    if (fst)
      fst = false;
    else
      os << " or ";
    printFlatAffineConstraints(os, c);
  }
  os << ")";
}

/// Prints the constraints of the `FlatAffineConstraints`. Each constraint is
/// printed separately and the are conjuncted with 'and'.
///
void printFlatAffineConstraints(raw_ostream &os,
                                const FlatAffineConstraints &cs) {
  unsigned numIds = cs.getNumIds();
  for (unsigned i = 0, e = cs.getNumEqualities(); i < e; ++i) {
    if (i != 0)
      os << " and ";
    ArrayRef<int64_t> eq = cs.getEquality(i);
    printExpr(os, eq.take_front(numIds), eq[numIds], cs.getNumDimIds());
    os << " = 0";
  }

  if (cs.getNumEqualities() > 0 && cs.getNumInequalities() > 0)
    os << " and ";

  for (unsigned i = 0, e = cs.getNumInequalities(); i < e; ++i) {
    if (i != 0)
      os << " and ";
    ArrayRef<int64_t> ineq = cs.getInequality(i);
    printExpr(os, ineq.take_front(numIds), ineq[numIds], cs.getNumDimIds());
    os << " >= 0";
  }
}

/// Prints the coefficient of the i'th variable with an additional '+' or '-' is
/// first = false. First indicates if this is the first summand of an
/// expression.
///
/// Returns false if the coefficient value is 0 and therefore is not printed.
///
bool printCoeff(raw_ostream &os, int64_t val, bool first) {
  if (val == 0)
    return false;

  if (val > 0) {
    if (!first) {
      os << " + ";
    }
    if (val > 1)
      os << val;
  } else {
    if (!first) {
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
  return true;
}

/// Prints the identifier of the i'th variable. The first nDim variables are
/// dimensions and therefore prefixed with 'd', everything afterwards is a
/// symbol with prefix 's'.
///
void printVarName(raw_ostream &os, int64_t i, unsigned nDim) {
  if (i < nDim) {
    os << 'd' << i;
  } else {
    os << 's' << (i - nDim);
  }
}

/// Prints a constant with an additional '+' or '-' is first = false. First
/// indicates if this is the first summand of an expression.
void printConst(raw_ostream &os, int64_t c, bool first) {
  if (first) {
    os << c;
  } else {
    if (c > 0)
      os << " + " << c;
    else if (c < 0)
      os << " - " << -c;
  }
}

/// Prints an affine expression. `coeffs` contains all the coefficients:
/// dimensions followed by symbols.
///
void printExpr(raw_ostream &os, ArrayRef<int64_t> coeffs, int64_t constant,
               unsigned nDim) {
  bool first = true;
  for (unsigned i = 0, e = coeffs.size(); i < e; ++i) {
    if (printCoeff(os, coeffs[i], first)) {
      first = false;
      printVarName(os, i, nDim);
    }
  }

  printConst(os, constant, first);
}

} // namespace

void mlir::analysis::presburger::printPresburgerSet(raw_ostream &os,
                                                    const PresburgerSet &set) {
  printVariableList(os, set.getNumDims(), set.getNumSyms());
  os << " : ";
  printConstraints(os, set.getFlatAffineConstraints());
}

void mlir::analysis::presburger::printPresburgerExpr(
    raw_ostream &os, const PresburgerExpr &expr) {
  unsigned nDim = expr.getNumDims(), nSym = expr.getNumSyms();
  printVariableList(os, nDim, nSym);

  os << " -> ";

  for (unsigned i = 0, e = expr.getExprs().size(); i < e; ++i) {
    if (i != 0)
      os << " ; ";

    os << "(";
    // TODO change this as soon as we have a Constraint class. Try to unify the
    // handling of expression
    auto eI = expr.getExprs()[i];
    printExpr(os, eI.second, eI.first, nDim);
    os << ")";
    os << " : ";
    printConstraints(os, expr.getDomains()[i].getFlatAffineConstraints());
  }
}

