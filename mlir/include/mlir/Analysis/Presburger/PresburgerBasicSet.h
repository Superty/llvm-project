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

template <typename Int>
class PresburgerSet;

template <typename Int>
class PresburgerMap;

template <typename Int>
class PresburgerBasicSet {
public:
  friend class PresburgerSet<Int>;
  friend class PresburgerMap<Int>;

  PresburgerBasicSet(unsigned oNDim = 0, unsigned oNParam = 0,
                     unsigned oNExist = 0)
      : nDim(oNDim), nParam(oNParam), nExist(oNExist) {}

  PresburgerBasicSet(unsigned oNDim, unsigned oNParam, unsigned oNExist,
                     ArrayRef<DivisionConstraint<Int>> divs);

  template <typename OInt>
  PresburgerBasicSet(const PresburgerBasicSet<OInt> &o);

  PresburgerBasicSet(const FlatAffineConstraints &fac);

  unsigned getNumDims() const { return nDim; }
  unsigned getNumTotalDims() const {
    return nParam + nDim + nExist + divs.size();
  }
  unsigned getNumParams() const { return nParam; }
  unsigned getNumExists() const { return nExist; }
  unsigned getNumDivs() const { return divs.size(); }
  unsigned getNumInequalities() const { return ineqs.size(); }
  unsigned getNumEqualities() const { return eqs.size(); }

  void intersect(PresburgerBasicSet bs);

  void appendDivisionVariable(ArrayRef<Int> coeffs, Int denom);

  static void toCommonSpace(PresburgerBasicSet &a, PresburgerBasicSet &b);
  void appendDivisionVariables(ArrayRef<DivisionConstraint<Int>> newDivs);
  void prependDivisionVariables(ArrayRef<DivisionConstraint<Int>> newDivs);

  const InequalityConstraint<Int> &getInequality(unsigned i) const;
  const EqualityConstraint<Int> &getEquality(unsigned i) const;
  ArrayRef<InequalityConstraint<Int>> getInequalities() const;
  ArrayRef<EqualityConstraint<Int>> getEqualities() const;
  ArrayRef<DivisionConstraint<Int>> getDivisions() const;

  void addInequality(ArrayRef<Int> coeffs);
  void addEquality(ArrayRef<Int> coeffs);

  void removeLastInequality();
  void removeLastEquality();
  void removeLastDivision();

  void removeDivision(unsigned i);
  void removeInequality(unsigned i);
  void removeEquality(unsigned i);

  Optional<SmallVector<Int, 8>>
  findIntegerSampleRemoveEqs(bool onlyEmptiness = false) const;

  /// Find a sample point satisfying the constraints. This uses a branch and
  /// bound algorithm with generalized basis reduction, which always works if
  /// the set is bounded. This should not be called for unbounded sets.
  ///
  /// Returns such a point if one exists, or an empty Optional otherwise.
  Optional<SmallVector<Int, 8>>
  findIntegerSample(bool onlyEmptiness = false) const;

  bool isIntegerEmpty() const;

  /// Get a {denominator, sample} pair representing a rational sample point in
  /// this basic set.
  Optional<std::pair<Int, SmallVector<Int, 8>>>
  findRationalSample() const;

  PresburgerBasicSet makeRecessionCone() const;

  /// Uses simplex to remove redundant constraints
  void removeRedundantConstraints();

  /// Align equivalent divs PresburgerBasicSets bs1 and bs2
  /// Converts non matching divisions to existentials
  /// 
  /// If preserve is set, does not add additional existentials
  ///
  /// Returns true if the alignment was succesfull.
  static bool alignDivs(PresburgerBasicSet<Int> &bs1,
                        PresburgerBasicSet<Int> &bs2, 
                        bool preserve);


  /// Return +1 if the basic set is trivially non-empty;
  ///        -1 if the basic set is empty;
  ///         0 otherwise.
  int isTrivial() const;

  /// Simplify the basic set. If aggressive is true, then uses more expensive 
  /// (in terms of performance) hueristics to simplify the basicset.
  void simplify(bool aggressive = true);

  PresburgerBasicSet<Int> makePlainBasicSet() const;
  bool isPlainBasicSet() const;

  void dumpCoeffs() const;

  void dump() const;
  void print(raw_ostream &os) const;

  void printISL(raw_ostream &os) const;
  void dumpISL() const;

  template <typename OInt>
  friend class PresburgerBasicSet;

private:
  void substitute(ArrayRef<Int> values);

  /// Find a sample point in this basic set, when it is known that this basic
  /// set has no unbounded directions.
  ///
  /// \returns the sample point or an empty llvm::Optional if the set is empty.
  Optional<SmallVector<Int, 8>> findSampleBounded(bool onlyEmptiness) const;

  /// Find a sample for only the bounded dimensions of this basic set.
  ///
  /// \param cone should be the recession cone of this basic set.
  ///
  /// \returns the sample or an empty std::optional if no sample exists.
  Optional<SmallVector<Int, 8>>
  findBoundedDimensionsSample(const PresburgerBasicSet &cone,
                              bool onlyEmptiness) const;

  /// Find a sample for this basic set, which is known to be a full-dimensional
  /// cone.
  ///
  /// \returns the sample point or an empty std::optional if the set is empty.
  Optional<SmallVector<Int, 8>> findSampleFullCone();

  /// Project this basic set to its bounded dimensions. It is assumed that the
  /// unbounded dimensions occupy the last \p unboundedDims dimensions.
  void projectOutUnboundedDimensions(unsigned unboundedDims);

  /// Find a sample point in this basic set, which has unbounded directions.
  ///
  /// \param cone should be the recession cone of this basic set.
  ///
  /// \returns the sample point or an empty llvm::Optional if the set
  /// is empty.
  Optional<SmallVector<Int, 8>>
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
  SmallVector<Int, 8> copyWithNonRedundant(std::vector<unsigned> &nrExists,
                                         std::vector<unsigned> &nrDiv,
                                         const ArrayRef<Int> &ogCoeffs);

  /// Remove divisions and existentials that do not occur in any constraint
  void removeRedundantVars();

  /// Convert existentials to divisions using inequalities
  /// Also converts inequalities that can from an equality to equalities
  void recoverDivisionsFromInequalities();

  /// Convert existentials to divisions using equalities
  void recoverDivisionsFromEqualities();

  /// Remove duplicate divisions
  void removeDuplicateDivs();

  /// Remove constraints that are trivially redundant
  void removeTriviallyRedundantConstraints();

  /// Remove duplicate equalities and inequalities
  void removeDuplicateConstraints();

  /// Remove constant divs
  void removeConstantDivs();

  /// Swap division variables at indexes vari and varj
  /// vari and varj are indexes in the divs vector
  void swapDivisions(unsigned vari, unsigned varj);

  /// Get the index of first division variable.
  unsigned getDivOffset();

  /// Get the index of first existential variable.
  unsigned getExistOffset();

  Matrix<Int> coefficientMatrixFromEqs() const;

  void insertDimensions(unsigned pos, unsigned count);
  void prependExistentialDimensions(unsigned count);
  void appendExistentialDimensions(unsigned count);

  void updateFromSimplex(const Simplex<Int> &simplex);

  SmallVector<InequalityConstraint<Int>, 8> ineqs;
  SmallVector<EqualityConstraint<Int>, 8> eqs;
  SmallVector<DivisionConstraint<Int>, 8> divs;
  unsigned nDim, nParam, nExist;
};
} // namespace presburger
} // namespace analysis
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_PRESBURGERBASICSET_H
