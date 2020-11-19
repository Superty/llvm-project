#include "mlir/Analysis/Presburger/ISLPrinter.h"

using namespace mlir;
using namespace analysis::presburger;

namespace {

void printConstraints(raw_ostream &os, ArrayRef<std::string> paramNames, const PresburgerBasicSet &bs);

void printConstraints(raw_ostream &os, ArrayRef<std::string> paramNames, const PresburgerSet &set);
void printVariableList(raw_ostream &os, ArrayRef<std::string> paramNames, unsigned nDim, unsigned nSym);
void printExpr(raw_ostream &os, ArrayRef<std::string> paramNames, ArrayRef<int64_t> coeffs, int64_t constant,
               const PresburgerBasicSet &bs);
bool printCoeff(raw_ostream &os, int64_t val, bool first);
void printVarName(raw_ostream &os, ArrayRef<std::string> paramNames, int64_t i, const PresburgerBasicSet &bs);
void printConst(raw_ostream &os, int64_t c, bool first);

/// Prints the '(d0, ..., dN)[s0, ... ,sM]' dimension and symbol list.
///
void printVariableList(raw_ostream &os, ArrayRef<std::string> paramNames, unsigned nDim, unsigned nSym) {
  if (nSym > 0) {
    os << "[";
    for (unsigned i = 0; i < nSym; i++) {
      os << (i != 0 ? ", " : "");
      if (paramNames.empty())
        os << "mlirs" << i;
      else
        os << paramNames[i];
    }
    os << "] -> ";
  }
    
  os << "{ [";
  for (unsigned i = 0; i < nDim; i++) {
    os << (i != 0 ? ", " : "") << "mlird" << i;
  }
  os << "]";
}

/// Prints the constraints of each `PresburgerBasicSet`.
///
void printConstraints(
    raw_ostream &os, ArrayRef<std::string> paramNames,
    const PresburgerSet &set) {
  bool fst = true;
  for (auto &c : set.getBasicSets()) {
    if (fst)
      fst = false;
    else
      os << " or ";
    printConstraints(os, paramNames, c);
  }
}

/// Prints the constraints of the `PresburgerBasicSet`. Each constraint is
/// printed separately and the are conjuncted with 'and'.
///
void printConstraints(raw_ostream &os, ArrayRef<std::string> paramNames,
                                const PresburgerBasicSet &bs) {
  os << '(';
  unsigned numTotalDims = bs.getNumTotalDims();

  if (bs.getNumExists() > 0 || bs.getNumDivs() > 0) {
    os << "exists (";
    bool fst = true;
    for (unsigned i = 0, e = bs.getNumExists(); i < e; ++i) {
      if (fst)
        fst = false;
      else
        os << ", ";
      os << "mlire" << i;
    }
    for (unsigned i = 0, e = bs.getNumDivs(); i < e; ++i) {
      if (fst)
        fst = false;
      else
        os << ", ";
      os << "mlirq" << i << " = [(";
      auto &div = bs.getDivisions()[i];
      printExpr(os, paramNames, div.getCoeffs().take_front(numTotalDims), div.getCoeffs()[numTotalDims], bs);
      os << ")/" << div.getDenominator() << "]";
    }
    os << " : ";
  }

  for (unsigned i = 0, e = bs.getNumEqualities(); i < e; ++i) {
    if (i != 0)
      os << " and ";
    ArrayRef<int64_t> eq = bs.getEquality(i).getCoeffs();
    printExpr(os, paramNames, eq.take_front(numTotalDims), eq[numTotalDims], bs);
    os << " = 0";
  }

  if (bs.getNumEqualities() > 0 && bs.getNumInequalities() > 0)
    os << " and ";

  for (unsigned i = 0, e = bs.getNumInequalities(); i < e; ++i) {
    if (i != 0)
      os << " and ";
    ArrayRef<int64_t> ineq = bs.getInequality(i).getCoeffs();
    printExpr(os, paramNames, ineq.take_front(numTotalDims), ineq[numTotalDims], bs);
    os << " >= 0";
  }

  if (bs.getNumExists() > 0 || bs.getNumDivs() > 0)
    os << ')';
  os << ')';
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
void printVarName(raw_ostream &os, ArrayRef<std::string> paramNames, int64_t i, const PresburgerBasicSet &bs) {
  if (i < bs.getNumDims()) {
    os << "mlird" << i;
    return;
  }
  i -= bs.getNumDims();
  
  if (i < bs.getNumParams()) {
    if (paramNames.empty())
      os << "mlirs" << i;
    else
      os << paramNames[i];
    return;
  }
  i -= bs.getNumParams();

  if (i < bs.getNumExists()) {
    os << "mlire" << i;
    return;
  }
  i -= bs.getNumExists();

  if (i < bs.getNumDivs()) {
    os << "mlirq" << i;
    return;
  }
  i -= bs.getNumDivs();

  llvm_unreachable("Unknown variable index!");
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
void printExpr(raw_ostream &os, ArrayRef<std::string> paramNames, ArrayRef<int64_t> coeffs, int64_t constant,
               const PresburgerBasicSet &bs) {
  bool first = true;
  for (unsigned i = 0, e = coeffs.size(); i < e; ++i) {
    if (printCoeff(os, coeffs[i], first)) {
      first = false;
      printVarName(os, paramNames, i, bs);
    }
  }

  printConst(os, constant, first);
}
} // namespace

void mlir::analysis::presburger::printPresburgerSetISL(raw_ostream &os,
                                                    const PresburgerSet &set) {
  printVariableList(os, set.getParamNames(), set.getNumDims(), set.getNumSyms());
  if (set.isUniverse()) {
    os << "}";
    return;
  }
  os << " : ";
  if (set.isMarkedEmpty()) {
    os << "false";
  } else {
    printConstraints(os, set.getParamNames(), set);
  }
  os << "}";
}
