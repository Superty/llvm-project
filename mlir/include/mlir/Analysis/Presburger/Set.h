//===- Set.h - MLIR PresburgerSet Class -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A class to represent unions of FlatAffineConstraints.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_SET_H
#define MLIR_ANALYSIS_PRESBURGER_SET_H

#include "mlir/Analysis/AffineStructures.h"

namespace mlir {

/// This class can represent a union of FlatAffineConstraints, with support for
/// union, intersection, subtraction and complement operations, as well as
/// sampling.
///
/// The FlatAffineConstraints (FACs) are stored in a vector, and the set
/// represents the union of these FACs.
class PresburgerSet {
public:
  PresburgerSet(unsigned nDim = 0, unsigned nSym = 0)
      : nDim(nDim), nSym(nSym) {}
  PresburgerSet(FlatAffineConstraints cs);

  /// Return the number of FACs in the union.
  unsigned getNumFACs() const;

  /// Return the number of real dimensions.
  unsigned getNumDims() const;

  /// Return the number of symbolic dimensions.
  unsigned getNumSyms() const;

  /// Returns a reference to the list of FlatAffineConstraints.
  ArrayRef<FlatAffineConstraints> getFlatAffineConstraints() const;

  /// Returns the FlatAffineConsatraints at the specified index.
  const FlatAffineConstraints &getFlatAffineConstraints(unsigned index) const;

  /// Add the given FlatAffineConstraints to the union.
  void addFlatAffineConstraints(FlatAffineConstraints cs);

  /// Intersect the given set with the current set.
  void unionSet(const PresburgerSet &set);

  /// Intersect the given set with the current set.
  void intersectSet(const PresburgerSet &set);

  /// Returns true if the set contains the given point, or false otherwise.
  bool containsPoint(ArrayRef<int64_t> point) const;

  /// Print the set's internal state.
  void print(raw_ostream &os) const;
  void dump() const;

  /// Returns the complement of the given set.
  static PresburgerSet complement(const PresburgerSet &set);

  /// Subtract the given set from the current set.
  void subtract(const PresburgerSet &set);

  /// Return the set difference c - set.
  static PresburgerSet subtract(FlatAffineConstraints &fac,
                                const PresburgerSet &set);

  /// Return the set difference c - set.
  static PresburgerSet subtract(FlatAffineConstraints &&fac,
                                const PresburgerSet &set);

  /// Return a universe set of the specified type that contains all points.
  static PresburgerSet makeUniverse(unsigned nDim = 0, unsigned nSym = 0);

  /// Returns true if all the sets in the union are known to be integer empty
  /// false otherwise.
  bool isIntegerEmpty() const;

  /// Find an integer sample from the given set. This should not be called if
  /// any of the FACs in the union are unbounded.
  llvm::Optional<SmallVector<int64_t, 8>> findIntegerSample();

private:
  /// Number of identifiers corresponding to real dimensions.
  unsigned nDim;

  /// Number of symbolic dimensions, unknown but constant for analysis, as in
  /// FlatAffineConstraints.
  unsigned nSym;

  /// The list of flatAffineConstraints that this set is the union of.
  SmallVector<FlatAffineConstraints, 2> flatAffineConstraints;
};

} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_SET_H
