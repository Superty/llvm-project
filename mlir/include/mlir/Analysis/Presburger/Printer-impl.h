#ifndef MLIR_ANALYSIS_PRESBURGER_PRINTER_IMPL_H
#define MLIR_ANALYSIS_PRESBURGER_PRINTER_IMPL_H

#include "mlir/Analysis/Presburger/Printer.h"

using namespace mlir;
using namespace analysis::presburger;

namespace {

template <typename Int>
void printConstraints(raw_ostream &os, const PresburgerBasicSet<Int> &bs);

template <typename Int>
void printConstraints(raw_ostream &os, const PresburgerSet<Int> &set);
// void printVariableList(raw_ostream &os, unsigned nDim, unsigned nSym);
template <typename Int>
void printExpr(raw_ostream &os, ArrayRef<Int> coeffs,
               Int constant, const PresburgerBasicSet<Int> &bs);
template <typename Int>
bool printCoeff(raw_ostream &os, Int val, bool first);
template <typename Int>
void printVarName(raw_ostream &os, unsigned i, const PresburgerBasicSet<Int> &bs);
template <typename Int>
void printConst(raw_ostream &os, Int c, bool first);

/// Prints the '(d0, ..., dN)[s0, ... ,sM]' dimension and symbol list.
///
// void printVariableList(raw_ostream &os, unsigned nDim, unsigned nSym) {
//   os << "(";
//   for (unsigned i = 0; i < nDim; i++)
//     os << (i != 0 ? ", " : "") << 'd' << i;
//   os << ")";

//   if (nSym > 0) {
//     os << "[";
//     for (unsigned i = 0; i < nSym; i++)
//       os << (i != 0 ? ", " : "") << 's' << i;
//     os << "]";
//   }
// }

/// Prints the constraints of each `PresburgerBasicSet<Int>`.
///
template <typename Int>
void printConstraints(raw_ostream &os, const PresburgerSet<Int> &set) {
  bool fst = true;
  for (auto &c : set.getBasicSets()) {
    if (fst)
      fst = false;
    else
      os << " or ";
    printConstraints(os, c);
  }
}

/// Prints the constraints of the `PresburgerBasicSet<Int>`. Each constraint is
/// printed separately and the are conjuncted with 'and'.
///
template <typename Int>
void printConstraints(raw_ostream &os, const PresburgerBasicSet<Int> &bs) {
  os << '(';
  unsigned numTotalDims = bs.getNumTotalDims();

  if (bs.getNumExists() > 0 || bs.getNumDivs() > 0) {
    os << "exists ";
    bool fst = true;
    for (unsigned i = 0, e = bs.getNumExists(); i < e; ++i) {
      if (fst)
        fst = false;
      else
        os << ", ";
      os << "e" << i;
    }
    for (unsigned i = 0, e = bs.getNumDivs(); i < e; ++i) {
      if (fst)
        fst = false;
      else
        os << ", ";
      os << "q" << i << " = [(";
      auto &div = bs.getDivisions()[i];
      printExpr(os, div.getCoeffs().take_front(numTotalDims),
                div.getCoeffs()[numTotalDims], bs);
      os << ")/" << div.getDenominator() << "]";
    }
    os << " : ";
  }

  for (unsigned i = 0, e = bs.getNumEqualities(); i < e; ++i) {
    if (i != 0)
      os << " and ";
    ArrayRef<Int> eq = bs.getEquality(i).getCoeffs();
    printExpr(os, eq.take_front(numTotalDims), eq[numTotalDims], bs);
    os << " = 0";
  }

  if (bs.getNumEqualities() > 0 && bs.getNumInequalities() > 0)
    os << " and ";

  for (unsigned i = 0, e = bs.getNumInequalities(); i < e; ++i) {
    if (i != 0)
      os << " and ";
    ArrayRef<Int> ineq = bs.getInequality(i).getCoeffs();
    printExpr(os, ineq.take_front(numTotalDims), ineq[numTotalDims], bs);
    os << " >= 0";
  }
  os << ')';
}

/// Prints the coefficient of the i'th variable with an additional '+' or '-' is
/// first = false. First indicates if this is the first summand of an
/// expression.
///
/// Returns false if the coefficient value is 0 and therefore is not printed.
template <typename Int>
bool printCoeff(raw_ostream &os, Int val, bool first) {
  if (val == Int(0))
    return false;

  if (val > Int(0)) {
    if (!first) {
      os << " + ";
    }
    if (val > Int(1))
      os << val;
  } else {
    if (!first) {
      os << " - ";
      if (val != Int(-1))
        os << -val;
    } else {
      if (val == Int(-1))
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
template <typename Int>
void printVarName(raw_ostream &os, unsigned i,
                  const PresburgerBasicSet<Int> &bs) {
  if (i < bs.getNumDims()) {
    os << 'd' << i;
    return;
  }
  i -= bs.getNumDims();

  if (i < bs.getNumParams()) {
    os << 's' << i;
    return;
  }
  i -= bs.getNumParams();

  if (i < bs.getNumExists()) {
    os << 'e' << i;
    return;
  }
  i -= bs.getNumExists();

  if (i < bs.getNumDivs()) {
    os << 'q' << i;
    return;
  }
  i -= bs.getNumDivs();

  llvm_unreachable("Unknown variable index!");
}

/// Prints a constant with an additional '+' or '-' is first = false. First
/// indicates if this is the first summand of an expression.
template <typename Int>
void printConst(raw_ostream &os, Int c, bool first) {
  if (first) {
    os << c;
  } else {
    if (c > Int(0))
      os << " + " << c;
    else if (c < Int(0))
      os << " - " << -c;
  }
}

/// Prints an affine expression. `coeffs` contains all the coefficients:
/// dimensions followed by symbols.
///
template <typename Int>
void printExpr(raw_ostream &os, ArrayRef<Int> coeffs,
               Int constant, const PresburgerBasicSet<Int> &bs) {
  bool first = true;
  for (unsigned i = 0, e = coeffs.size(); i < e; ++i) {
    if (printCoeff(os, coeffs[i], first)) {
      first = false;
      printVarName(os, i, bs);
    }
  }

  printConst(os, constant, first);
}

} // namespace

template <typename Int>
void mlir::analysis::presburger::printPresburgerSet(raw_ostream &os,
                                                    const PresburgerSet<Int> &set) {
  printVariableList(os, set.getNumDims(), set.getNumSyms());
  os << " : ";
  if (set.isMarkedEmpty()) {
    os << "empty";
    return;
  }
  printConstraints(os, set);
}

template <typename Int>
void mlir::analysis::presburger::printPresburgerBasicSet(
    raw_ostream &os, const PresburgerBasicSet<Int> &bs) {
  printVariableList(os, bs.getNumDims(), bs.getNumParams());
  os << " : ";
  printConstraints(os, bs);
}

// void mlir::analysis::presburger::printPresburgerExpr(
//     raw_ostream &os, const PresburgerExpr &expr) {
//   unsigned nDim = expr.getNumDims(), nSym = expr.getNumSyms();

//   // Terrible hack to pass nDim, nSym to printExpr until the domains of Exprs
//   // become basic sets.
//   PresburgerBasicSet<int64_t> bs(nDim, nSym);
//   printVariableList(os, nDim, nSym);

//   os << " -> ";

//   for (unsigned i = 0, e = expr.getExprs().size(); i < e; ++i) {
//     if (i != 0)
//       os << " ; ";

//     os << "(";
//     // TODO change this as soon as we have a Constraint class. Try to unify the
//     // handling of expression
//     auto eI = expr.getExprs()[i];
//     llvm_unreachable("not yet implemented!");
//     // printExpr(os, eI.second, eI.first, bs);
//     os << ")";
//     os << " : ";
//     printConstraints(os, expr.getDomains()[i]);
//   }
// }
#endif // MLIR_ANALYSIS_PRESBURGER_PRINTER_IMPL_H
