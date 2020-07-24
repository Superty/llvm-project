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

class PresburgerBasicSet {
public:
  PresburgerBasicSet() = delete;
  PresburgerBasicSet(unsigned oNDim, unsigned oNParam, unsigned oNExist)
    : nDim(oNDim), nParam(oNParam), nExist(oNExist) {}

  unsigned getNumDims() const { return nDim; }
  unsigned getNumTotalDims() const { return nParam + nDim + nExist + divs.size(); }
  unsigned getNumParams() const { return nParam; }
  unsigned getNumExists() const { return nExist; }
  unsigned getNumDivs() const { return divs.size(); }
  unsigned getNumInequalities() const { return ineqs.size(); }
  unsigned getNumEqualities() const { return eqs.size(); }
  void appendDivisionVariable(ArrayRef<int64_t> coeffs, int64_t denom);


  const InequalityConstraint &getInequality(unsigned i) const;
  const EqualityConstraint &getEquality(unsigned i) const;
  ArrayRef<InequalityConstraint> getInequalities() const;
  ArrayRef<EqualityConstraint> getEqualities() const;

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

  /// Get a {denominator, sample} pair representing a rational sample point in
  /// this basic set.
  Optional<std::pair<int64_t, SmallVector<int64_t, 8>>>
  findRationalSample() const;

  PresburgerBasicSet makeRecessionCone() const;


  void dump() const;

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
  findSampleUnbounded(PresburgerBasicSet &cone) const;

  Matrix coefficientMatrixFromEqs() const;
  void assertPlainSet() const;

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
