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
  /// Construct an empty PresburgerSet.
  PresburgerSet(unsigned nDim = 0, unsigned nSym = 0)
      : nDim(nDim), nSym(nSym) {}

  explicit PresburgerSet(const FlatAffineConstraints &fac);

  /// Return the number of FACs in the union.
  unsigned getNumFACs() const;

  /// Return the number of real dimensions.
  unsigned getNumDims() const;

  /// Return the number of symbolic dimensions.
  unsigned getNumSyms() const;

  /// Return a reference to the list of FlatAffineConstraints.
  ArrayRef<FlatAffineConstraints> getAllFlatAffineConstraints() const;

  /// Return the FlatAffineConsatraints at the specified index.
  const FlatAffineConstraints &getFlatAffineConstraints(unsigned index) const;

  /// Add the given FlatAffineConstraints to the union.
  void addFlatAffineConstraints(const FlatAffineConstraints &fac);

  /// Return the union of this set and the given set.
  PresburgerSet unionSet(const PresburgerSet &set) const;

  /// Return the intersection of this set and the given set.
  PresburgerSet intersect(const PresburgerSet &set) const;

  /// Return true if the set contains the given point, or false otherwise.
  bool containsPoint(ArrayRef<int64_t> point) const;

  /// Print the set's internal state.
  void print(raw_ostream &os) const;
  void dump() const;

  /// Return the complement of this set.
  PresburgerSet complement() const;

  /// Return the set difference facA \ facB.
  static PresburgerSet getSetDifference(const FlatAffineConstraints &facA,
                                        const FlatAffineConstraints &facB);

  /// Return the set difference of this set and the given set, i.e.,
  /// return `this \ set`.
  PresburgerSet subtract(const PresburgerSet &set) const;

  /// Return a universe set of the specified type that contains all points.
  static PresburgerSet makeUniverse(unsigned nDim = 0, unsigned nSym = 0);

  /// Return true if all the sets in the union are known to be integer empty
  /// false otherwise.
  bool isIntegerEmpty() const;

  /// Find an integer sample from the given set. This should not be called if
  /// any of the FACs in the union are unbounded.
  bool findIntegerSample(SmallVectorImpl<int64_t> &sample);

private:
  /// Return the set difference fac \ set.
  static PresburgerSet getSetDifference(FlatAffineConstraints fac,
                                        const PresburgerSet &set);

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
