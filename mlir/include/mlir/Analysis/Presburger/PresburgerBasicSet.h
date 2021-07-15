//===- PresburgerBasicSet.h - MLIR PresburgerBasicSet Class -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functionality to perform analysis on FlatAffineConstraints. In particular,
// support for performing emptiness checks.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_PRESBURGERBASICSET_H
#define MLIR_ANALYSIS_PRESBURGER_PRESBURGERBASICSET_H

#include "mlir/Analysis/Presburger/Constraint.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Analysis/Presburger/Simplex.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace analysis {
namespace presburger {

class PresburgerSet;

class PresburgerBasicSet {
public:
  friend class PresburgerSet;
  friend class PresburgerBasicMap;

  PresburgerBasicSet(unsigned oNDim = 0, unsigned oNParam = 0, unsigned oNExist = 0)
    : nDim(oNDim), nParam(oNParam), nExist(oNExist) {}

  PresburgerBasicSet(unsigned oNDim, unsigned oNParam, unsigned oNExist,
                     ArrayRef<DivisionConstraint> oDivs);

  PresburgerBasicSet(unsigned oNDim, unsigned oNParam, unsigned oNExist,
                     ArrayRef<InequalityConstraint> oIneqs,
                     ArrayRef<EqualityConstraint> oEqs,
                     ArrayRef<DivisionConstraint> oDivs);

  unsigned getNumDims() const { return nDim; }
  unsigned getNumTotalDims() const { return nParam + nDim + nExist + divs.size(); }
  unsigned getNumParams() const { return nParam; }
  unsigned getNumExists() const { return nExist; }
  unsigned getNumDivs() const { return divs.size(); }
  unsigned getNumInequalities() const { return ineqs.size(); }
  unsigned getNumEqualities() const { return eqs.size(); }

  void intersect(PresburgerBasicSet bs);

  void appendDivisionVariable(ArrayRef<int64_t> coeffs, int64_t denom);

  static void toCommonSpace(PresburgerBasicSet &a, PresburgerBasicSet &b);
  void appendDivisionVariables(ArrayRef<DivisionConstraint> newDivs);
  void prependDivisionVariables(ArrayRef<DivisionConstraint> newDivs);

  const InequalityConstraint &getInequality(unsigned i) const;
  const EqualityConstraint &getEquality(unsigned i) const;
  ArrayRef<InequalityConstraint> getInequalities() const;
  ArrayRef<EqualityConstraint> getEqualities() const;
  ArrayRef<DivisionConstraint> getDivisions() const;

  void addInequality(ArrayRef<int64_t> coeffs);
  void addEquality(ArrayRef<int64_t> coeffs);

  void removeLastInequality();
  void removeLastEquality();
  void removeLastDivision();

  void removeInequality(unsigned i);
  void removeEquality(unsigned i);

  /// Find a sample point satisfying the constraints. This uses a branch and
  /// bound algorithm with generalized basis reduction, which always works if
  /// the set is bounded. This should not be called for unbounded sets.
  ///
  /// Returns such a point if one exists, or an empty Optional otherwise.
  Optional<SmallVector<int64_t, 8>> findIntegerSample() const;

  bool isIntegerEmpty();

  /// Get a {denominator, sample} pair representing a rational sample point in
  /// this basic set.
  Optional<std::pair<int64_t, SmallVector<int64_t, 8>>>
  findRationalSample() const;

  PresburgerBasicSet makeRecessionCone() const;

  /// Uses simplex to remove redundant constraints
  void removeRedundantConstraints();

  /// Align equivalent divs PresburgerBasicSets bs1 and bs2
  /// Converts non matching divisions to existentials
  static void alignDivs(PresburgerBasicSet &bs1, PresburgerBasicSet &bs2);

  void simplify();

  void dumpCoeffs() const;

  void dump() const;
  void print(raw_ostream &os) const;

  void printISL(raw_ostream &os) const;
  void dumpISL() const;

private:
  void substitute(ArrayRef<int64_t> values);

  /// Find a sample point in this basic set, when it is known that this basic
  /// set has no unbounded directions.
  ///
  /// \returns the sample point or an empty llvm::Optional if the set is empty.
  Optional<SmallVector<int64_t, 8>> findSampleBounded() const;

  /// Find a sample for only the bounded dimensions of this basic set.
  ///
  /// \param cone should be the recession cone of this basic set.
  ///
  /// \returns the sample or an empty std::optional if no sample exists.
  Optional<SmallVector<int64_t, 8>>
  findBoundedDimensionsSample(const PresburgerBasicSet &cone) const;

  /// Find a sample for this basic set, which is known to be a full-dimensional
  /// cone.
  ///
  /// \returns the sample point or an empty std::optional if the set is empty.
  Optional<SmallVector<int64_t, 8>> findSampleFullCone();

  /// Project this basic set to its bounded dimensions. It is assumed that the
  /// unbounded dimensions occupy the last \p unboundedDims dimensions.
  void projectOutUnboundedDimensions(unsigned unboundedDims);

  /// Find a sample point in this basic set, which has unbounded directions.
  ///
  /// \param cone should be the recession cone of this basic set.
  ///
  /// \returns the sample point or an empty llvm::Optional if the set
  /// is empty.
  Optional<SmallVector<int64_t, 8>>
  findSampleUnbounded(PresburgerBasicSet &cone, bool onlyEmptiness) const;

  /// Factor out greatest commond divisor from each equality and inequality
  void normalizeConstraints();

  /// 1. Converts each coefficient c in the division numerator to
  /// be in the range -denominator < 2 * c <= denominator
  /// 2. Divides numerator and denominator by their gcd
  /// Assumes that divisions are ordered before this function is called.
  void normalizeDivisions();

  /// Orders divisions such that a division only depends on division
  /// before it.
  void orderDivisions();

  /// Return true if the variable is redundant in the set
  bool redundantVar(unsigned var);

  /// Creates new coeffs from ogCoeffs, keeping only the exists in 
  /// nrExists and divs in nrDiv and removing the rest
  SmallVector<int64_t, 8> copyWithNonRedundant(std::vector<unsigned> &nrExists,
                                         std::vector<unsigned> &nrDiv,
                                         const ArrayRef<int64_t> &ogCoeffs);

  /// Remove divisions and existentials that do not occur in any constraint
  void removeRedundantVars();

  /// Convert existentials to divisions using inequalities
  /// Also converts inequalities that can from an equality to equalities
  void recoverDivisionsFromInequalities();

  /// Convert existentials to divisions using equalities
  void recoverDivisionsFromEqualities();

  /// Remove duplicate divisions
  void removeDuplicateDivs();

  /// Swap division variables at indexes vari and varj
  /// vari and varj are indexes in the divs vector
  void swapDivisions(unsigned vari, unsigned varj);

  /// Get the index of first division variable.
  unsigned getDivOffset();

  /// Get the index of first existential variable.
  unsigned getExistOffset();

  /// Convert dimensions between range [l, r) to existentials
  void convertDimsToExists(unsigned l, unsigned r);

  Matrix coefficientMatrixFromEqs() const;

  void insertDimensions(unsigned pos, unsigned count);
  void prependExistentialDimensions(unsigned count);
  void appendExistentialDimensions(unsigned count);

  PresburgerBasicSet makePlainBasicSet() const;
  bool isPlainBasicSet() const;

  void updateFromSimplex(const Simplex &simplex);

  SmallVector<InequalityConstraint, 8> ineqs;
  SmallVector<EqualityConstraint, 8> eqs;
  SmallVector<DivisionConstraint, 8> divs;
  unsigned nDim, nParam, nExist;
};
} // namespace presburger
} // namespace analysis
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_PRESBURGERBASICSET_H
