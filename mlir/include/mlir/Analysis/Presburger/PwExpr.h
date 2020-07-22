#ifndef MLIR_ANALYSIS_PRESBURGER_PW_EXPR_H
#define MLIR_ANALYSIS_PRESBURGER_PW_EXPR_H

#include "mlir/Analysis/Presburger/Set.h"

namespace mlir {
class PresburgerPwExpr {
public:
  using ExprType = std::pair<int64_t, SmallVector<int64_t, 8>>;
  PresburgerPwExpr(unsigned nDim = 0, unsigned nSym = 0)
      : nDim(nDim), nSym(nSym) {}

  unsigned getNumDims() const;
  unsigned getNumSyms() const;
  void addPiece(const ExprType &expr, const PresburgerSet &domain);
  void print(raw_ostream &os) const;
  void dump() const;
  llvm::hash_code hash_value() const;

private:
  void printVariableList(raw_ostream &os) const;
  void printExpr(ExprType expr, raw_ostream &os) const;
  void printVar(raw_ostream &os, int64_t val, unsigned i,
                unsigned &countNonZero) const;

  SmallVector<ExprType, 2> exprs;
  SmallVector<PresburgerSet, 2> domains;

  unsigned nDim, nSym;
};
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_PW_EXPR_H
