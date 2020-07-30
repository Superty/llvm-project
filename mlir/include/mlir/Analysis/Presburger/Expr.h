#ifndef MLIR_ANALYSIS_PRESBURGER_EXPR_H
#define MLIR_ANALYSIS_PRESBURGER_EXPR_H

#include "mlir/Analysis/Presburger/Set.h"

namespace mlir {
namespace analysis {
namespace presburger {

/// This class implements a data structure that represents a Presburger
/// expression, i.e a piecewise affine expression. Each piece is define by a
/// domain and an expression that is applied to points lying in this domain.
/// Domains are represented as parameteric Presburger sets, expressions by
/// affine expressions.
///
/// As the domains can be parametric, a Presburger expression not only takes
/// dimension parameters, but symbols as well.
///
/// TODO: Add support for quasi-affine expressions as soon as the land.
///
class PresburgerExpr {
public:
  using ExprType = std::pair<int64_t, SmallVector<int64_t, 8>>;
  PresburgerExpr(unsigned nDim = 0, unsigned nSym = 0)
      : nDim(nDim), nSym(nSym) {}

  unsigned getNumDims() const;
  unsigned getNumSyms() const;

  const SmallVector<ExprType, 2> &getExprs() const;
  const SmallVector<PresburgerSet, 2> &getDomains() const;

  /// Adds a piece that applies expr on the speciefied domain.
  void addPiece(const ExprType &expr, const PresburgerSet &domain);

  void print(raw_ostream &os) const;
  void dump() const;
  llvm::hash_code hash_value() const;

private:
  SmallVector<ExprType, 2> exprs;
  SmallVector<PresburgerSet, 2> domains;

  unsigned nDim, nSym;
};

} // namespace presburger
} // namespace analysis
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_EXPR_H
