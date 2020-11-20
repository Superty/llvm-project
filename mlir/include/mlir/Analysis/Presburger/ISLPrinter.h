#ifndef PRESBURGER_ISL_PRINTER_H
#define PRESBURGER_ISL_PRINTER_H

#include "mlir/Analysis/Presburger/Expr.h"
#include "mlir/Analysis/Presburger/Set.h"

namespace mlir {
namespace analysis {
namespace presburger {

/// This file provides printing functions for the different Presburger
/// datastructures.
/// TODO: move this to the head documentation

/// Prints the set into the stream
void printPresburgerSetISL(raw_ostream &os, const PresburgerSet &set);

/// Prints the basic set into the stream
void printPresburgerBasicSetISL(raw_ostream &os, const PresburgerBasicSet &set);
} // namespace presburger
} // namespace analysis
} // namespace mlir

#endif // PRESBURGER_PRINTER_H
