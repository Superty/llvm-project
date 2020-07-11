#include "mlir/Analysis/Presburger/Coalesce.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Presburger/Set.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include <iostream>

using namespace mlir;

// TODO: look at MutableArrayRef

// struct for classified constraints
// redundant and cut are for constraints that are typed as REDUNDANT or CUT
// respectively. adjIneq is for any constraint, that is adjacent to the other
// polytope. t is for any constraint, that is part of an equality constraint and
// adjacent to the other polytope.
// TODO: find better name than t
// TODO: possibly change up structure of Info, if simplex manages to classify
// adjacent to equality
struct Info {
  SmallVector<ArrayRef<int64_t>, 8> redundant;
  SmallVector<ArrayRef<int64_t>, 8> cut;
  Optional<ArrayRef<int64_t>> adjIneq;
  Optional<ArrayRef<int64_t>> t;
  void dump() {
    std::cout << "red:" << std::endl;
    for (ArrayRef<int64_t> curr : this->redundant) {
      mlir::dump(curr);
    }
    std::cout << "cut:" << std::endl;
    for (ArrayRef<int64_t> curr : this->cut) {
      mlir::dump(curr);
    }
    if (this->adjIneq) {
      std::cout << "adjIneq:" << std::endl;
      mlir::dump(this->adjIneq.getValue());
    }
    if (this->t) {
      std::cout << "t:" << std::endl;
      mlir::dump(this->t.getValue());
    }
  }
};

// computes the complement of t
// i.e. for a given constraint t(x) >= 0 returns -t(x) -1 >= 0
SmallVector<int64_t, 8> complement(ArrayRef<int64_t> t);

// shifts t by amount
// i.e. for a given constraint t(x) >= 0 return t(x) + amount >= 0
void shift(SmallVectorImpl<int64_t> &t, int amount);

// add eq as two inequalities to target
void AddEqualitiesAsInequalities(const ArrayRef<ArrayRef<int64_t>> eq,
                                 SmallVectorImpl<ArrayRef<int64_t>> &target);

// adds all Equalities to bs
void addEqualities(FlatAffineConstraints &bs,
                   const SmallVector<ArrayRef<int64_t>, 8> &equalities);

// adds all Inequalities to bs
void addInequalities(FlatAffineConstraints &bs,
                     const SmallVector<ArrayRef<int64_t>, 8> &inequalities);

// only gets called by classify
// classifies all constraints into redundant, cut or adjIneq according to the
// ineqType that the simplex returns.
//
// returns true if it has neither encountered a separate constraint nor more
// than one adjacent inequality. This has a similar idea as short-circuiting
// behind it. The moment either a separate or two adjacent constraints are
// encountered, the polytope cannot be coalesced anymore, so we can move to the
// next tuple.
bool classifyIneq(Simplex &simp,
                  const SmallVector<ArrayRef<int64_t>, 8> &constraints,
                  Info &info);

// classifies all constraints into redundant, cut, adjIneq or t, where t stands
// for a constraint adjacent to a the other polytope
//
// returns true if it has neither encountered a separate constraint nor more
// than one adjacent inequality. This has a similar idea as short-circuiting
// behind it. The moment either a separate or two adjacent constraints are
// encountered, the polytope cannot be coalesced anymore, so we can move to the
// next tuple.
bool classify(Simplex &simp,
              const SmallVector<ArrayRef<int64_t>, 8> &inequalities,
              const SmallVector<ArrayRef<int64_t>, 8> &equalities, Info &info);

// compute the protrusionCase and return whether it has worked
//
// In the protusionCase, only CUT constraints and REDUNDANT constraints can
// exist. This case differs from the CutCase in that it can still coalesce
// polytopes, as long as they are not sticking out of each other by too much
// (i.e. by less than 2). As we are considering an integer setting, the convex
// hull of the union of those two polytopes actually is the same thing as the
// union of the two polytope.
//      _____          _____
//    _|___  |        /     |
//   | |   | |   ==> |      |
//   | |   | |       |      |
//   |_|___|_|       |______|
//
// TODO: find better example
bool protrusionCase(SmallVectorImpl<FlatAffineConstraints> &basicSetVector,
                    Info &infoA, const Info &infoB, unsigned i, unsigned j);

// compute, whether a constraint of cut sticks out of bs by more than 2
bool stickingOut(const SmallVector<ArrayRef<int64_t>, 8> &cut,
                 const FlatAffineConstraints &bs);

// adds a FlatAffineConstraints and removes the sets at i and j.
void addCoalescedBasicSet(
    SmallVectorImpl<FlatAffineConstraints> &basicSetVector, unsigned i,
    unsigned j, const FlatAffineConstraints &bs);

// compute the cut case and return whether it has worked.
//
// The cut case is the case, for which a polytope only has REDUNDANT and CUT
// constraints. If all the facets of such cut constraints are contained within
// the other polytope, the polytopes can be combined to a polytope only
// limited by all the REDUNDANT constraints.
//    ___________        ___________
//   /   /  |   /       /          /
//   \   \  |  /   ==>  \         /
//    \   \ | /          \       /
//     \___\|/            \_____/
//
//
bool cutCase(SmallVector<FlatAffineConstraints, 4> &basicSetVector, unsigned i,
             unsigned j, const Info &infoA, const Info &infoB);

// compute adjIneq pure Case and return whether it has worked.
//
// The adjIneq pure case can be viewed as a single polytope originally (below:
// on the right), that was cut into two parts by a strip with width 1.
// This is computed by just using all the REDUNDANT constraints.
//  ________ ____            ______________
// |       //    |          |              |
// |      //     |          |              |
// |     //      |    ==>   |              |
// |    //       |          |              |
// |___//________|          |______________|
//
bool adjIneqPureCase(SmallVector<FlatAffineConstraints, 4> &basicSetVector,
                     unsigned i, unsigned j, const Info &infoA,
                     const Info &infoB);

// compute the non-pure adjIneq case and return whether it has worked.
// Constraint t is the adjIneq.
//
// In the non-pure adjIneq case, one of the polytopes is like an extension of
// the other one. This can be computed, by inverting the adjIneq and checking
// whether all constraints are stll valid for this new polytope.
//
// In the example below, Polytope A is the big one on the left, Polytope B is
// the small line at the right side. that gets included after coalescing.
//   ____          ____
//  |    \        |    \
//  |     \   ==> |     \
//  |_____|_      |______\
//
bool adjIneqCase(SmallVector<FlatAffineConstraints, 4> &basicSetVector,
                 unsigned i, unsigned j, const Info &infoA, const Info &infoB);

// compute the pure adjEqCase and return whether it has worked.
//
// The pure case consists of two equalities, that are adjacent. Such equalities
// can always be coalesced by finding the two constraints, that make them become
// a trapezoid. This is done by wrapping the limiting constraints of the
// equalities around each other to include the other one.
//                _
//    / /        / /
//   / /   ==>  / /
//  / /        /_/
//
bool adjEqCasePure(SmallVectorImpl<FlatAffineConstraints> &basicSetVector,
                   unsigned i, unsigned j, const Info &infoA,
                   const Info &infoB);

// compute the adjEq Case for no CUT constraints.
//
// The adjEq no cut case is simply an extension case, where the constraint,
// that is adjacent to the equality, can be relaxed by 1 to include the other
// polytope.
//   ______           _______
//  |      ||        |       |
//  |      ||        |       |
//  |      ||   ==>  |       |
//  |______||        |_______|
//
bool adjEqCaseNoCut(SmallVectorImpl<FlatAffineConstraints> &basicSetVector,
                    unsigned i, unsigned j, SmallVector<int64_t, 8> t);

// compute the non-pure adjEqCase and return whether it has worked.
//
// The non-pure case has cut constraints such that it is not a simple extension
// like the no cut case. It is computed by wrapping those cut constraints and
// checking, whether everything stilly holds.
//     ________           ________
//    |        |         |        \
//    |        | |  ==>  |         |
//    |        |         |        /
//    |        /         |       /
//    |_______/          |______/
//
bool adjEqCaseNonPure(SmallVectorImpl<FlatAffineConstraints> &basicSetVector,
                      unsigned i, unsigned j, const Info &infoA,
                      const Info &infoB);

void shift(SmallVectorImpl<int64_t> &t, int amount) {
  t.push_back(t.pop_back_val() + amount);
}

SmallVector<int64_t, 8> complement(ArrayRef<int64_t> t) {
  SmallVector<int64_t, 8> complement;
  for (size_t k = 0; k < t.size() - 1; k++) {
    complement.push_back(-t[k]);
  }
  complement.push_back(-t.back() - 1);
  return complement;
}

// helperfuncton to convert arrayRefs to SmallVectors
static SmallVector<int64_t, 8> arrayRefToSmallVector(ArrayRef<int64_t> ref) {
  SmallVector<int64_t, 8> res;
  for (const int64_t &curr : ref)
    res.push_back(curr);
  return res;
}

// returns all Equalities of a BasicSet as a SmallVector of ArrayRefs
void getBasicSetEqualities(const FlatAffineConstraints &bs,
                           SmallVector<ArrayRef<int64_t>, 8> &eqs) {
  for (unsigned k = 0; k < bs.getNumEqualities(); k++) {
    eqs.push_back(bs.getEquality(k));
  }
}

// returns all Inequalities of a BasicSet as a SmallVector of ArrayRefs
void getBasicSetInequalities(const FlatAffineConstraints &bs,
                             SmallVector<ArrayRef<int64_t>, 8> &ineqs) {
  for (unsigned k = 0; k < bs.getNumInequalities(); k++) {
    ineqs.push_back(bs.getInequality(k));
  }
}

PresburgerSet mlir::coalesce(PresburgerSet &set) {
  PresburgerSet newSet(set.getNumDims(), set.getNumSyms());
  SmallVector<FlatAffineConstraints, 4> basicSetVector =
      set.getFlatAffineConstraints();
  // TODO: find better looping strategy
  // redefine coalescing function on two BasicSets, return a BasicSet and do the
  // looping strategy in a different function?
  for (size_t i = 0; i < basicSetVector.size(); i++) {
    for (size_t j = 0; j < basicSetVector.size(); j++) {
      if (j == i)
        continue;
      FlatAffineConstraints bs1 = basicSetVector[i];
      Simplex simplex1(bs1);
      SmallVector<ArrayRef<int64_t>, 8> equalities1, inequalities1;
      getBasicSetEqualities(bs1, equalities1);
      getBasicSetInequalities(bs1, inequalities1);

      FlatAffineConstraints bs2 = basicSetVector[j];
      Simplex simplex2(bs2);
      SmallVector<ArrayRef<int64_t>, 8> equalities2, inequalities2;
      getBasicSetEqualities(bs2, equalities2);
      getBasicSetInequalities(bs2, inequalities2);

      Info info1, info2;
      if (!classify(simplex2, inequalities1, equalities1, info1))
        continue;
      if (!classify(simplex1, inequalities2, equalities2, info2))
        continue;

      // TODO: find better strategy than i--; break;
      // change indices when changing vector?
      if (!info1.redundant.empty() && info1.cut.empty() && !info1.adjIneq &&
          !info2.t) {
        // contained 2 in 1
        basicSetVector.erase(basicSetVector.begin() + j);
        j--;
      } else if (!info2.redundant.empty() && info2.cut.empty() &&
                 !info2.adjIneq && !info1.t) {
        // contained 1 in 2
        basicSetVector.erase(basicSetVector.begin() + i);
        i--;
        break;
      } else if (!info1.redundant.empty() && !info1.cut.empty() &&
                 !info1.adjIneq && !info2.t) {
        // cut or protrusion case 1
        if (cutCase(basicSetVector, i, j, info1, info2)) {
          i--;
          break;
        } else if (stickingOut(info1.cut, bs2) &&
                   protrusionCase(basicSetVector, info1, info2, i, j)) {
          // protrusion
          i--;
          break;
        }
      } else if (!info2.redundant.empty() && !info2.cut.empty() &&
                 !info2.adjIneq && !info1.t) {
        // cut or protrusion case 2
        if (cutCase(basicSetVector, j, i, info2, info1)) {
          i--;
          break;
        } else if (stickingOut(info2.cut, bs1) &&
                   protrusionCase(basicSetVector, info2, info1, j, i)) {
          // protrusion case
          i--;
          break;
        }
      } else if (!info1.redundant.empty() && info1.adjIneq &&
                 info1.cut.empty() && !info2.t && !info2.redundant.empty() &&
                 info2.adjIneq && info2.cut.empty() && !info1.t) {
        // adjIneq, pure case
        if (adjIneqPureCase(basicSetVector, i, j, info1, info2)) {
          i--;
          break;
        }
      } else if (!info1.redundant.empty() && info1.adjIneq &&
                 info1.cut.empty() && !info1.t && !info2.t) {
        // adjIneq complex case 1
        if (adjIneqCase(basicSetVector, i, j, info1, info2)) {
          i--;
          break;
        }
      } else if (!info2.redundant.empty() && info2.adjIneq &&
                 info2.cut.empty() && !info2.t && !info1.t) {
        // adjIneq complex case 2
        if (adjIneqCase(basicSetVector, j, i, info2, info1)) {
          i--;
          break;
        }
      } else if (info1.t && info2.t) {
        // adjEq for two equalities
        if (adjEqCasePure(basicSetVector, i, j, info1, info2)) {
          i--;
          break;
        }
      } else if (info1.t && info2.cut.empty()) {
        // adjEq Case for one equality 1
        // compute the inequality, that is adjacent to an equality by computing
        // the complement of the inequality part of an equality
        SmallVector<int64_t, 8> adjEq = complement(info1.t.getValue());
        if (adjEqCaseNoCut(basicSetVector, i, j, adjEq)) {
          // adjEq noCut case
          i--;
          break;
        } else if (info1.t &&
                   adjEqCaseNonPure(basicSetVector, j, i, info2, info1)) {
          // adjEq case
          i--;
          break;
        }
      } else if (info2.t && info1.cut.empty()) {
        // adjEq Case for one equality 2
        // compute the inequality, that is adjacent to an equality by computing
        // the complement of the inequality part of an equality
        SmallVector<int64_t, 8> adjEq = complement(info2.t.getValue());
        if (adjEqCaseNoCut(basicSetVector, j, i, adjEq)) {
          // adjEq noCut case
          i--;
          break;
        } else if (info2.t &&
                   adjEqCaseNonPure(basicSetVector, i, j, info1, info2)) {
          // adjEq case
          i--;
          break;
        }
      }
    }
  }
  for (const FlatAffineConstraints curr : basicSetVector) {
    newSet.addFlatAffineConstraints(curr);
  }
  return newSet;
}

void addInequalities(FlatAffineConstraints &bs,
                     const SmallVector<ArrayRef<int64_t>, 8> &inequalities) {
  for (size_t k = 0; k < inequalities.size(); k++) {
    bs.addInequality(inequalities[k]);
  }
}

void addEqualities(FlatAffineConstraints &bs,
                   const SmallVector<ArrayRef<int64_t>, 8> &equalities) {
  for (size_t k = 0; k < equalities.size(); k++) {
    bs.addEquality(equalities[k]);
  }
}

bool adjIneqPureCase(SmallVector<FlatAffineConstraints, 4> &basicSetVector,
                     unsigned i, unsigned j, const Info &infoA,
                     const Info &infoB) {
  FlatAffineConstraints newSet(basicSetVector[i].getNumIds(),
                               basicSetVector[i].getNumSymbolIds());
  addInequalities(newSet, infoA.redundant);
  addInequalities(newSet, infoB.redundant);
  addCoalescedBasicSet(basicSetVector, i, j, newSet);
  return true;
}

// Currently erases both i and j and the pushes the new BasicSet to the back of
// the vector.
// TODO: probably change when changing looping strategy
void addCoalescedBasicSet(
    SmallVectorImpl<FlatAffineConstraints> &basicSetVector, unsigned i,
    unsigned j, const FlatAffineConstraints &bs) {
  if (i < j) {
    basicSetVector.erase(basicSetVector.begin() + j);
    basicSetVector.erase(basicSetVector.begin() + i);
  } else {
    basicSetVector.erase(basicSetVector.begin() + i);
    basicSetVector.erase(basicSetVector.begin() + j);
  }
  basicSetVector.push_back(bs);
}

bool protrusionCase(SmallVectorImpl<FlatAffineConstraints> &basicSetVector,
                    Info &infoA, const Info &infoB, unsigned i, unsigned j) {
  FlatAffineConstraints &a = basicSetVector[i];
  FlatAffineConstraints &b = basicSetVector[j];
  SmallVector<ArrayRef<int64_t>, 8> constraintsB, equalitiesB;
  getBasicSetEqualities(b, equalitiesB);
  getBasicSetInequalities(b, constraintsB);
  AddEqualitiesAsInequalities(equalitiesB, constraintsB);

  // For every cut constraint t of a, bPrime is computed as b intersected with
  // t(x) + 1 = 0. For this, t is shifted by 1.
  SmallVector<SmallVector<int64_t, 8>, 8> wrapped;
  for (size_t l = 0; l < infoA.cut.size(); l++) {
    SmallVector<int64_t, 8> t = arrayRefToSmallVector(infoA.cut[l]);
    FlatAffineConstraints bPrime =
        FlatAffineConstraints(b.getNumDimIds(), b.getNumSymbolIds());
    addInequalities(bPrime, constraintsB);
    shift(t, 1);
    Simplex simp(bPrime);
    simp.addEquality(t);
    if (simp.isEmpty()) {
      // If bPrime is empty, t is shifted back and reclassified as redundant.
      shift(t, -1);
      infoA.redundant.push_back(t);
      infoA.cut.erase(infoA.cut.begin() + l);
    } else {
      // Otherwise, all cut constraints of B are considered. Of those, only the
      // ones, that actually define bPrime are considered. So the ones, that
      // actually "touche" the polytope bPrime. Those are wrapped around t(x) +
      // 1 to include a.
      for (ArrayRef<int64_t> currCut : infoB.cut) {
        SmallVector<int64_t, 8> curr1 = arrayRefToSmallVector(currCut);
        Simplex simp2(bPrime);
        simp2.addEquality(curr1);
        // TODO: "touching" the polytope is currently defined by adding the
        // constraint to be checked as an equality and then looking whether this
        // polytope is empty or not. This requires, that the point at which the
        // constraint touches the polytope, is an integer point. It is not
        // entirely clear yet, whether this is sufficient.
        if (!simp2.isEmpty()) {
          auto result = wrapping(a, t, curr1);
          if (!result) {
            return false;
          }
          wrapped.push_back(result.getValue());
        }
      }
    }
  }

  FlatAffineConstraints newSet(b.getNumDimIds(), b.getNumSymbolIds());
  // If all the wrappings were succesfull, the two polytopes can be replaced by
  // a polytope with all of the redundant constraints and the wrapped
  // constraints.
  addInequalities(newSet, infoA.redundant);
  addInequalities(newSet, infoB.redundant);
  for (const SmallVector<int64_t, 8> curr : wrapped) {
    newSet.addInequality(curr);
  }
  // Additionally for every remaining cut constraint t of a, t + 1 >= 0 is
  // added.
  for (ArrayRef<int64_t> currRef : infoA.cut) {
    SmallVector<int64_t, 8> curr = arrayRefToSmallVector(currRef);
    shift(curr, 1);
    newSet.addInequality(curr);
  }
  addCoalescedBasicSet(basicSetVector, i, j, newSet);
  return true;
}

bool stickingOut(const SmallVector<ArrayRef<int64_t>, 8> &cut,
                 const FlatAffineConstraints &bs) {
  Simplex simp(bs);
  // for every cut constraint t, compute the optimum in the direction of cut. If
  // the optimum is < -2, the polytopee doesn't stick too much out of the other
  // one.
  for (ArrayRef<int64_t> curr : cut) {
    auto result = simp.computeOptimum(Simplex::Direction::Down, curr);
    if (!result) {
      return false;
    }
    auto res = result.getValue();
    if (res <= Fraction(-2, 1)) {
      return false;
    }
  }
  return true;
}

bool mlir::sameConstraint(ArrayRef<int64_t> c1, ArrayRef<int64_t> c2) {
  Fraction ratio(0, 1);
  assert(c1.size() == c1.size() && "the constraints have different dimensions");
  // if c1 = a*c2, this iterates over the vector trying to find a as soon as
  // possible and then comparing with a.
  for (size_t i = 0; i < c1.size(); i++) {
    // TODO: is ratio.num == 0 really sufficient for checking whether fraction
    // was set yet
    if (c2[i] != 0 && ratio.num == 0) {
      ratio.num = c1[i];
      ratio.den = c2[i];
    } else if (c2[i] != 0 && ratio.num != 0) {
      if (ratio.den * c1[i] != ratio.num * c2[i]) {
        return false;
      }
    } else {
      if (c1[i] != c2[i]) {
        return false;
      }
    }
  }
  return true;
}

bool adjEqCaseNoCut(SmallVectorImpl<FlatAffineConstraints> &basicSetVector,
                    unsigned i, unsigned j, SmallVector<int64_t, 8> t) {
  FlatAffineConstraints &A = basicSetVector[j];
  FlatAffineConstraints &B = basicSetVector[i];
  // relax t by 1 and add all other constraints of A to newSet.
  // TODO: is looping only over the inequalities sufficient here?
  // Reasoning: if A had an equality, we would be in a different case.
  SmallVector<SmallVector<int64_t, 8>, 8> newSetInequalities;
  for (size_t k = 0; k < A.getNumInequalities(); k++) {
    SmallVector<int64_t, 8> curr = arrayRefToSmallVector(A.getInequality(k));
    if (!sameConstraint(t, curr)) {
      newSetInequalities.push_back(arrayRefToSmallVector(A.getInequality(k)));
    }
  }
  shift(t, 1);
  newSetInequalities.push_back(t);
  FlatAffineConstraints newSet(A.getNumDimIds(), A.getNumSymbolIds());
  for (size_t k = 0; k < newSetInequalities.size(); k++) {
    newSet.addInequality(newSetInequalities[k]);
  }

  // for every constraint c of B, check if it is redundant for the newSet with
  // t added as an equality. This makes sure, that B actually is a simple
  // extension of A.
  Simplex simp(newSet);
  simp.addEquality(t);
  SmallVector<ArrayRef<int64_t>, 8> constraintsB, equalitiesB;
  for (size_t k = 0; k < B.getNumEqualities(); k++) {
    equalitiesB.push_back(B.getEquality(k));
  }
  for (size_t k = 0; k < B.getNumInequalities(); k++) {
    constraintsB.push_back(B.getInequality(k));
  }
  AddEqualitiesAsInequalities(equalitiesB, constraintsB);
  for (size_t k = 0; k < constraintsB.size(); k++) {
    if (simp.ineqType(constraintsB[k]) != Simplex::IneqType::REDUNDANT) {
      return false;
    }
  }
  addCoalescedBasicSet(basicSetVector, i, j, newSet);
  return true;
}

bool adjEqCaseNonPure(SmallVectorImpl<FlatAffineConstraints> &basicSetVector,
                      unsigned i, unsigned j, const Info &infoA,
                      const Info &infoB) {
  FlatAffineConstraints &a = basicSetVector[i];
  FlatAffineConstraints &b = basicSetVector[j];
  SmallVector<SmallVector<int64_t, 8>, 8> wrapped;
  SmallVector<int64_t, 8> minusT;
  // the constraint of a adjacent to an equality, it the complement of the
  // constraint f b, that is part of an equality and adjacent to an inequality.
  SmallVector<int64_t, 8> t =
      complement(arrayRefToSmallVector(infoB.t.getValue()));
  for (size_t k = 0; k < t.size(); k++) {
    minusT.push_back(-t[k]);
  }

  // TODO: can only cut be non-redundant?
  // The cut constraints of a are wrapped around -t to include B.
  for (size_t k = 0; k < infoA.cut.size(); k++) {
    // TODO: why does the pure case differ here in that it doesn't wrap t?
    SmallVector<int64_t, 8> curr = arrayRefToSmallVector(infoA.cut[k]);
    auto result = wrapping(b, minusT, curr);
    if (!result)
      return false;
    wrapped.push_back(result.getValue());
  }

  // Some of the wrapped constraints can now be non redudant.
  Simplex simp(b);
  for (size_t k = 0; k < wrapped.size(); k++) {
    if (simp.ineqType(wrapped[k]) != Simplex::IneqType::REDUNDANT) {
      return false;
    }
  }

  shift(t, 1);
  shift(minusT, -1);
  // TODO: can only cut be non-redundant?
  // the cut constraints of b (except -t - 1) are wrapped around t + 1 to
  // include a.
  for (size_t k = 0; k < infoB.cut.size(); k++) {
    if (!sameConstraint(minusT, infoB.cut[k])) {
      SmallVector<int64_t, 8> curr = arrayRefToSmallVector(infoB.cut[k]);
      auto result = wrapping(a, t, curr);
      if (!result)
        return false;
      wrapped.push_back(result.getValue());
    }
  }

  FlatAffineConstraints newSet(b.getNumIds(), b.getNumSymbolIds());
  // The new polytope consists of all the wrapped constraints and all the
  // redundant constraints
  for (size_t k = 0; k < wrapped.size(); k++) {
    newSet.addInequality(wrapped[k]);
  }
  addInequalities(newSet, infoA.redundant);
  addInequalities(newSet, infoB.redundant);
  // additionally, t + 1 is added.
  newSet.addInequality(t);

  addCoalescedBasicSet(basicSetVector, i, j, newSet);
  return true;
}

bool adjEqCasePure(SmallVectorImpl<FlatAffineConstraints> &basicSetVector,
                   unsigned i, unsigned j, const Info &infoA,
                   const Info &infoB) {
  FlatAffineConstraints &a = basicSetVector[i];
  FlatAffineConstraints &b = basicSetVector[j];
  SmallVector<SmallVector<int64_t, 8>, 8> wrapped;
  SmallVector<int64_t, 8> minusT;
  SmallVector<int64_t, 8> t = arrayRefToSmallVector(infoA.t.getValue());
  for (size_t k = 0; k < t.size(); k++) {
    minusT.push_back(-t[k]);
  }

  // TODO: can only cut be non-redundant?
  // The cut constraints of a (except t) are wrapped around -t to include B.
  for (size_t k = 0; k < infoA.cut.size(); k++) {
    if (!sameConstraint(t, infoA.cut[k])) {
      SmallVector<int64_t, 8> curr = arrayRefToSmallVector(infoA.cut[k]);
      auto result = wrapping(b, minusT, curr);
      if (!result)
        return false;
      wrapped.push_back(result.getValue());
    }
  }

  shift(t, 1);
  shift(minusT, -1);
  // TODO: can only cut be non-redundant?
  // the cut constraints of b (except -t -1) are wrapped around t+1 to include
  // a.
  for (size_t k = 0; k < infoB.cut.size(); k++) {
    if (!sameConstraint(minusT, infoB.cut[k])) {
      SmallVector<int64_t, 8> curr = arrayRefToSmallVector(infoB.cut[k]);
      auto result = wrapping(a, t, curr);
      if (!result)
        return false;
      wrapped.push_back(result.getValue());
    }
  }

  FlatAffineConstraints newSet(b.getNumIds(), b.getNumSymbolIds());
  // The new polytope consists of all the wrapped constraints and all the
  // redundant constraints
  for (size_t k = 0; k < wrapped.size(); k++) {
    newSet.addInequality(wrapped[k]);
  }
  addInequalities(newSet, infoA.redundant);
  addInequalities(newSet, infoB.redundant);
  // Additionally, the constraint t + 1 is added to the new BasicSet.
  newSet.addInequality(t);
  addCoalescedBasicSet(basicSetVector, i, j, newSet);
  return true;
}

SmallVector<int64_t, 8> mlir::combineConstraint(ArrayRef<int64_t> c1,
                                                ArrayRef<int64_t> c2,
                                                Fraction &ratio) {
  int64_t n = ratio.num;
  int64_t d = ratio.den;
  SmallVector<int64_t, 8> result;
  for (size_t i = 0; i < c1.size(); i++) {
    result.push_back(-n * c1[i] + d * c2[i]);
  }
  return result;
}

Optional<SmallVector<int64_t, 8>>
mlir::wrapping(const FlatAffineConstraints &bs, SmallVectorImpl<int64_t> &valid,
               SmallVectorImpl<int64_t> &invalid) {
  size_t n = bs.getNumDimIds();
  Simplex simplex(n + 1);

  // for every constraint t(x) + c of bs, make it become t(x) + c*lambda
  for (size_t k = 0; k < bs.getNumEqualities(); k++) {
    SmallVector<int64_t, 8> curr = arrayRefToSmallVector(bs.getEquality(k));
    curr.push_back(0);
    simplex.addEquality(curr);
  }
  for (size_t k = 0; k < bs.getNumInequalities(); k++) {
    SmallVector<int64_t, 8> curr = arrayRefToSmallVector(bs.getInequality(k));
    curr.push_back(0);
    simplex.addInequality(curr);
  }

  // add lambda >= 0
  // TODO: why does it work with n here? Isn't this constraint actually n+2
  // long?
  SmallVector<int64_t, 8> lambda(n, 0);
  lambda.push_back(1);
  lambda.push_back(0);
  simplex.addInequality(lambda);

  // make the valid constraint be equal to 1 and add it as an equality
  valid.push_back(-1);
  simplex.addEquality(valid);

  // transform the invalid constraint into lambda space
  invalid.push_back(0);
  Optional<Fraction> result =
      simplex.computeOptimum(Simplex::Direction::Down, invalid);
  if (!result) {
    return {};
  }

  // retransform valid and invalid into normal space before combining them.
  valid.pop_back();
  invalid.pop_back();
  return combineConstraint(valid, invalid, result.getValue());
}

bool classify(Simplex &simp,
              const SmallVector<ArrayRef<int64_t>, 8> &inequalities,
              const SmallVector<ArrayRef<int64_t>, 8> &equalities, Info &info) {
  Optional<SmallVector<int64_t, 8>> dummy = {};
  if (!classifyIneq(simp, inequalities, info))
    return false;
  SmallVector<ArrayRef<int64_t>, 8> eqAsIneq;
  AddEqualitiesAsInequalities(equalities, eqAsIneq);
  for (ArrayRef<int64_t> currentConstraint : eqAsIneq) {
    Simplex::IneqType ty = simp.ineqType(currentConstraint);
    switch (ty) {
    case Simplex::IneqType::REDUNDANT:
      info.redundant.push_back(currentConstraint);
      break;
    case Simplex::IneqType::CUT:
      info.cut.push_back(currentConstraint);
      break;
    case Simplex::IneqType::AdjIneq:
      if (info.adjIneq)
        // if two adjacent constraints are found, we can surely not coalesce
        // this town is too small for two adjIneq
        return false;
      info.adjIneq = currentConstraint;
      info.t = currentConstraint;
      break;
    case Simplex::IneqType::AdjEq:
      // TODO: possibly needs to change if simplex can handle adjacent to
      // equality
      info.t = currentConstraint;
      break;
    case Simplex::IneqType::SEPARATE:
      // coalescing always failes when a separate constraint is encountered.
      return false;
    }
  }
  return true;
}

bool classifyIneq(Simplex &simp,
                  const SmallVector<ArrayRef<int64_t>, 8> &constraints,
                  Info &info) {
  for (ArrayRef<int64_t> currentConstraint : constraints) {
    Simplex::IneqType ty = simp.ineqType(currentConstraint);
    switch (ty) {
    case Simplex::IneqType::REDUNDANT:
      info.redundant.push_back(currentConstraint);
      break;
    case Simplex::IneqType::CUT:
      info.cut.push_back(currentConstraint);
      break;
    case Simplex::IneqType::AdjIneq:
      if (info.adjIneq)
        // if two adjacent constraints are found, we can surely not coalesce
        // this town is too small for two adjIneq
        return false;
      info.adjIneq = currentConstraint;
      break;
    case Simplex::IneqType::AdjEq:
      // TODO: possibly needs to change if simplex can handle adjacent to
      // equality
      break;
    case Simplex::IneqType::SEPARATE:
      // coalescing always failes when a separate constraint is encountered.
      return false;
    }
  }
  return true;
}

bool adjIneqCase(SmallVector<FlatAffineConstraints, 4> &basicSetVector,
                 unsigned i, unsigned j, const Info &infoA, const Info &infoB) {
  ArrayRef<int64_t> t = infoA.adjIneq.getValue();
  FlatAffineConstraints bs(basicSetVector[i].getNumDimIds(),
                           basicSetVector[i].getNumSymbolIds());
  addInequalities(bs, infoA.redundant);
  addInequalities(bs, infoA.cut);
  addInequalities(bs, infoB.redundant);

  // If all constraints of a are added but the adjacent one and all the
  // REDUNDANT ones from b, are all cut constraints of b now REDUNDANT?
  // If so, all REDUNDANT constraints of a and b together define the new
  // polytope
  Simplex comp(bs);
  comp.addInequality(complement(t));
  for (size_t k = 0; k < infoB.cut.size(); k++) {
    if (comp.ineqType(infoB.cut[k]) != Simplex::IneqType::REDUNDANT) {
      return false;
    }
  }
  if (infoB.adjIneq) {
    if (comp.ineqType(infoB.adjIneq.getValue()) !=
        Simplex::IneqType::REDUNDANT) {
      return false;
    }
  }
  FlatAffineConstraints newSet(basicSetVector[i].getNumDimIds(),
                               basicSetVector[i].getNumSymbolIds());
  addInequalities(newSet, infoA.redundant);
  addInequalities(newSet, infoB.redundant);
  addCoalescedBasicSet(basicSetVector, i, j, newSet);
  return true;
}

bool cutCase(SmallVector<FlatAffineConstraints, 4> &basicSetVector, unsigned i,
             unsigned j, const Info &infoA, const Info &infoB) {
  // if all facets are contained, the redundant constraints of both a and b
  // define the new polytope
  for (size_t k = 0; k < infoA.cut.size(); k++) {
    if (!containedFacet(infoA.cut[k], basicSetVector[i], infoB.cut)) {
      return false;
    }
  }
  FlatAffineConstraints newSet(basicSetVector[i].getNumIds(),
                               basicSetVector[i].getNumSymbolIds());
  addInequalities(newSet, infoA.redundant);
  addInequalities(newSet, infoB.redundant);
  addCoalescedBasicSet(basicSetVector, i, j, newSet);
  return true;
}

bool mlir::containedFacet(ArrayRef<int64_t> ineq,
                          const FlatAffineConstraints &bs,
                          const SmallVector<ArrayRef<int64_t>, 8> &cut) {
  Simplex simp(bs);
  simp.addEquality(ineq);
  for (ArrayRef<int64_t> curr : cut) {
    if (simp.ineqType(curr) != Simplex::IneqType::REDUNDANT) {
      return false;
    }
  }
  return true;
}

void AddEqualitiesAsInequalities(ArrayRef<ArrayRef<int64_t>> eq,
                                 SmallVectorImpl<ArrayRef<int64_t>> &target) {
  for (ArrayRef<int64_t> curr : eq) {
    target.push_back(curr);
    // TODO: fix this memory leak
    SmallVector<int64_t, 8> *inverted = new SmallVector<int64_t, 8>();
    for (int64_t n : curr) {
      inverted->push_back(-n);
    }
    ArrayRef<int64_t> invertedRef(*inverted);
    target.push_back(invertedRef);
  }
}

void mlir::dump(const ArrayRef<int64_t> cons) {
  std::cout << cons[cons.size() - 1] << " + ";
  for (size_t i = 1; i < cons.size(); i++) {
    std::cout << cons[i - 1] << "x" << i - 1;
    if (i == cons.size() - 1) {
      break;
    }
    std::cout << " + ";
  }
  std::cout << " >= 0" << std::endl;
}
/*This function removes the BasicSet at position pos
 */
/*template <typename T> void removeElement(int pos, std::vector<T> &vec) {
  vec.erase(vec.begin() + pos);
}*/
