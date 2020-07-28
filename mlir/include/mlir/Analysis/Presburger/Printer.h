#ifndef PRESBURGER_PRINTER_H
#define PRESBURGER_PRINTER_H

#include "mlir/Analysis/Presburger/PwExpr.h"
#include "mlir/Analysis/Presburger/Set.h"

namespace mlir {
/// This class provides printing functions for the different Presburger
/// datastructures.
class PresburgerPrinter {
public:
  /// Prints the set into the stream
  static void print(raw_ostream &os, const PresburgerSet &set);

  /// Prints the piecewise Presburger expression into the stream
  static void print(raw_ostream &os, const PresburgerPwExpr &expr);

private:
  // print helpers
  static void printFlatAffineConstraints(raw_ostream &os,
                                         const FlatAffineConstraints &cs);

  static void printConstraints(
      raw_ostream &os,
      const SmallVectorImpl<FlatAffineConstraints> &flatAffineConstraints);
  static void printVariableList(raw_ostream &os, unsigned nDim, unsigned nSym);
  static void printExpr(raw_ostream &os, ArrayRef<int64_t> coeffs,
                        int64_t constant, unsigned nDim);
  static LogicalResult printCoef(raw_ostream &os, int64_t val, bool first);
  static void printVarName(raw_ostream &os, int64_t i, unsigned nDim);
  static void printConst(raw_ostream &os, int64_t c, bool first);
};

} // namespace mlir
#endif // PRESBURGER_PRINTER_H
