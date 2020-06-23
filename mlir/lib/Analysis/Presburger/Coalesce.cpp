#include "mlir/Analysis/Presburger/Coalesce.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Presburger/Set.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include <iostream>

using namespace mlir;

// struct for classified constraints
// redundant and cut are for constraints that are typed as REDUNDANT or CUT
// respectively. adj_ineq is for any constraint, that is adjacent to the other
// polytope. t is for any constraint, that is part of an equality constraint and
// adjacent to the other polytope.
// TODO: find better name than t
// TODO: possibly change up structure of Info, if simplex manages to classify
// adjacent to equality
struct Info {
  SmallVector<ArrayRef<int64_t>, 8> redundant;
  SmallVector<ArrayRef<int64_t>, 8> cut;
  Optional<ArrayRef<int64_t>> adj_ineq;
  Optional<ArrayRef<int64_t>> t;
};

// computes the complement of t
// i.e. for a given constraint t(x) >= 0 returns -t(x) -1 >= 0
SmallVector<int64_t, 8> complement(ArrayRef<int64_t> t);

// shifts t by amount
// i.e. for a given constraint t(x) >= 0 return t(x) + amount >= 0
void shift(SmallVectorImpl<int64_t> &t, int amount);

// dumps an Info struct
void dumpInfo(const Info &info);

// add eq as two inequalities to target
void addAsIneq(const ArrayRef<ArrayRef<int64_t>> eq,
               SmallVectorImpl<ArrayRef<int64_t>> &target);

// adds all Equalities to bs
void addEqualities(FlatAffineConstraints &bs,
                   const SmallVector<ArrayRef<int64_t>, 8> &equalities);

// adds all Inequalities to bs
void addInequalities(FlatAffineConstraints &bs,
                     const SmallVector<ArrayRef<int64_t>, 8> &inequalities);

// only gets called by classify
// classifies all constraints into redundant, cut or adj_ineq according to the
// ineqType that the simplex returns
//
// returns true if it has neither encountered a separate constraint nor more
// than one adjacent inequality
bool classify_ineq(Simplex &simp,
                   const SmallVector<ArrayRef<int64_t>, 8> &constraints,
                   Info &info);

// classifies all constraints into redundant, cut, adj_ineq or t, where t stands
// for a constraint adjacent to a the other polytope
//
// returns true if it has not encountered a separate constraint or more than one
// adjcacent constraint
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
                    Info &info_a, const Info &info_b, unsigned i, unsigned j);

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
// _________          _________
// \ \  |  /          \       /
//  \ \ | /    ==>     \     /
//   \_\|/              \___/
//
//
bool cutCase(SmallVector<FlatAffineConstraints, 4> &basicSetVector, unsigned i,
             unsigned j, const Info &info_a, const Info &info_b);

// compute adj_ineq pure Case and return whether it has worked.
//
// The adj_ineq pure case can be viewed as a single polytope originally (below:
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
                     unsigned i, unsigned j, const Info &info_a,
                     const Info &info_b);

// compute the non-pure adj_ineq case and return whether it has worked.
// Constraint t is the adj_ineq.
//
// In the non-pure adj_ineq case, one of the polytopes is like an extension of
// the other one. This can be computed, by inverting the adj_ineq and checking
// whether all constraints are stll valid for this new polytope.
//   ____          ____
//  |    \        |    \
//  |     \   ==> |     \
//  |_____|_      |______\
//
bool adjIneqCase(SmallVector<FlatAffineConstraints, 4> &basicSetVector,
                 unsigned i, unsigned j, const Info &info_a,
                 const Info &info_b);

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
                   unsigned i, unsigned j, const Info &info_a,
                   const Info &info_b);

// compute the adj_eq Case for no CUT constraints.
//
// The adj_eq no cut case is simply an extension case, where the constraint,
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
                      unsigned i, unsigned j, const Info &info_a,
                      const Info &info_b);

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
  PresburgerSet new_set(set.getNumDims(), set.getNumSyms());
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
      Simplex simplex_1(bs1);
      SmallVector<ArrayRef<int64_t>, 8> equalities1, inequalities1;
      getBasicSetEqualities(bs1, equalities1);
      getBasicSetInequalities(bs1, inequalities1);

      FlatAffineConstraints bs2 = basicSetVector[j];
      Simplex simplex_2(bs2);
      SmallVector<ArrayRef<int64_t>, 8> equalities2, inequalities2;
      getBasicSetEqualities(bs2, equalities2);
      getBasicSetInequalities(bs2, inequalities2);

      Info info_1, info_2;
      if (!classify(simplex_2, inequalities1, equalities1, info_1))
        continue;
      if (!classify(simplex_1, inequalities2, equalities2, info_2))
        continue;

      // TODO: find better strategy than i--; break;
      // change indices when changing vector?
      if (!info_1.redundant.empty() && info_1.cut.empty() && !info_1.adj_ineq &&
          !info_2.t) {
        // contained 2 in 1
        basicSetVector.erase(basicSetVector.begin() + j);
        j--;
      } else if (!info_2.redundant.empty() && info_2.cut.empty() &&
                 !info_2.adj_ineq && !info_1.t) {
        // contained 1 in 2
        basicSetVector.erase(basicSetVector.begin() + i);
        i--;
        break;
      } else if (!info_1.redundant.empty() && !info_1.cut.empty() &&
                 !info_1.adj_ineq && !info_2.t) {
        // cut or protrusion case 1
        if (cutCase(basicSetVector, i, j, info_1, info_2)) {
          i--;
          break;
        } else if (stickingOut(info_1.cut, bs2) &&
                   protrusionCase(basicSetVector, info_1, info_2, i, j)) {
          // protrusion
          i--;
          break;
        }
      } else if (!info_2.redundant.empty() && !info_2.cut.empty() &&
                 !info_2.adj_ineq && !info_1.t) {
        // cut or protrusion case 2
        if (cutCase(basicSetVector, j, i, info_2, info_1)) {
          i--;
          break;
        } else if (stickingOut(info_2.cut, bs1) &&
                   protrusionCase(basicSetVector, info_2, info_1, j, i)) {
          // protrusion case
          i--;
          break;
        }
      } else if (!info_1.redundant.empty() && info_1.adj_ineq &&
                 info_1.cut.empty() && !info_2.t && !info_2.redundant.empty() &&
                 info_2.adj_ineq && info_2.cut.empty() && !info_1.t) {
        // adj_ineq, pure case
        if (adjIneqPureCase(basicSetVector, i, j, info_1, info_2)) {
          i--;
          break;
        }
      } else if (!info_1.redundant.empty() && info_1.adj_ineq &&
                 info_1.cut.empty() && !info_1.t && !info_2.t) {
        // adj_ineq complex case 1
        if (adjIneqCase(basicSetVector, i, j, info_1, info_2)) {
          i--;
          break;
        }
      } else if (!info_2.redundant.empty() && info_2.adj_ineq &&
                 info_2.cut.empty() && !info_2.t && !info_1.t) {
        // adj_ineq complex case 2
        if (adjIneqCase(basicSetVector, j, i, info_2, info_1)) {
          i--;
          break;
        }
      } else if (info_1.t && info_2.t) {
        // adj_eq for two equalities
        if (adjEqCasePure(basicSetVector, i, j, info_1, info_2)) {
          i--;
          break;
        }
      } else if (info_1.t && info_2.cut.empty()) {
        // adj_eq Case for one equality 1
        // compute the inequality, that is adjacent to an equality by computing
        // the complement of the inequality part of an equality
        SmallVector<int64_t, 8> adjEq = complement(info_1.t.getValue());
        if (adjEqCaseNoCut(basicSetVector, i, j, adjEq)) {
          // adj_eq noCut case
          i--;
          break;
        } else if (info_1.t &&
                   adjEqCaseNonPure(basicSetVector, j, i, info_2, info_1)) {
          // adjEq case
          i--;
          break;
        }
      } else if (info_2.t && info_1.cut.empty()) {
        // adj_eq Case for one equality 2
        // compute the inequality, that is adjacent to an equality by computing
        // the complement of the inequality part of an equality
        SmallVector<int64_t, 8> adjEq = complement(info_2.t.getValue());
        if (adjEqCaseNoCut(basicSetVector, j, i, adjEq)) {
          // adj_eq noCut case
          i--;
          break;
        } else if (info_2.t &&
                   adjEqCaseNonPure(basicSetVector, i, j, info_1, info_2)) {
          // adjEq case
          i--;
          break;
        }
      }
    }
  }
  for (size_t i = 0; i < basicSetVector.size(); i++) {
    new_set.addFlatAffineConstraints(basicSetVector[i]);
  }
  return new_set;
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
                     unsigned i, unsigned j, const Info &info_a,
                     const Info &info_b) {
  FlatAffineConstraints newSet(basicSetVector[i].getNumIds(),
                               basicSetVector[i].getNumSymbolIds());
  addInequalities(newSet, info_a.redundant);
  addInequalities(newSet, info_b.redundant);
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
                    Info &info_a, const Info &info_b, unsigned i, unsigned j) {
  FlatAffineConstraints &a = basicSetVector[i];
  FlatAffineConstraints &b = basicSetVector[j];
  SmallVector<ArrayRef<int64_t>, 8> constraints_b, equalities_b;
  getBasicSetEqualities(b, equalities_b);
  getBasicSetInequalities(b, constraints_b);
  addAsIneq(equalities_b, constraints_b);

  // For every cut constraint t of a, bPrime is computed as b intersected with
  // t(x) + 1 = 0. For this, t is shifted by 1.
  SmallVector<SmallVector<int64_t, 8>, 8> wrapped;
  for (size_t l = 0; l < info_a.cut.size(); l++) {
    SmallVector<int64_t, 8> t = arrayRefToSmallVector(info_a.cut[l]);
    FlatAffineConstraints bPrime =
        FlatAffineConstraints(b.getNumDimIds(), b.getNumSymbolIds());
    addInequalities(bPrime, constraints_b);
    shift(t, 1);
    Simplex simp(bPrime);
    simp.addEquality(t);
    if (simp.isEmpty()) {
      // If bPrime is empty, t is shifted back and reclassified as redundant.
      shift(t, -1);
      info_a.redundant.push_back(t);
      info_a.cut.erase(info_a.cut.begin() + l);
    } else {
      // Otherwise, all cut constraints of B are considered. Of those, only the
      // ones, that actually define bPrime are considered. So the ones, that
      // actually "touche" the polytope bPrime. Those are wrapped around t(x) +
      // 1 to include a.
      for (size_t k = 0; k < info_b.cut.size(); k++) {
        SmallVector<int64_t, 8> curr1 = arrayRefToSmallVector(info_b.cut[k]);
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

  FlatAffineConstraints new_set(b.getNumDimIds(), b.getNumSymbolIds());
  // If all the wrappings were succesfull, the two polytopes can be replaced by
  // a polytope with all of the redundant constraints and the wrapped
  // constraints.
  addInequalities(new_set, info_a.redundant);
  addInequalities(new_set, info_b.redundant);
  for (size_t k = 0; k < wrapped.size(); k++) {
    new_set.addInequality(wrapped[k]);
  }
  // Additionally for every remaining cut constraint t of a, t + 1 >= 0 is
  // added.
  for (size_t k = 0; k < info_a.cut.size(); k++) {
    SmallVector<int64_t, 8> curr = arrayRefToSmallVector(info_a.cut[k]);
    shift(curr, 1);
    new_set.addInequality(curr);
  }
  addCoalescedBasicSet(basicSetVector, i, j, new_set);
  return true;
}

bool stickingOut(const SmallVector<ArrayRef<int64_t>, 8> &cut,
                 const FlatAffineConstraints &bs) {
  Simplex simp(bs);
  // for every cut constraint t, compute the optimum in the direction of cut. If
  // the optimum is < -2, the polytopee doesn't stick too much out of the other
  // one.
  for (size_t k = 0; k < cut.size(); k++) {
    ArrayRef<int64_t> curr = cut[k];
    auto result = simp.computeOptimum(Simplex::Direction::Down, curr);
    if (!result) {
      return false;
    }
    auto res = result.getValue();
    if (res.num <= -2 * res.den) {
      return false;
    }
  }
  return true;
}

bool mlir::sameConstraint(ArrayRef<int64_t> c1, ArrayRef<int64_t> c2) {
  Fraction ratio(0, 1);
  if (c1.size() != c1.size()) {
    return false;
  }
  // if c1 = a*c2, this iterates over the vector trying to find a as soon as
  // possible and then comparing with a.
  for (size_t i = 0; i < c1.size(); i++) {
    // TODO: is ratio.num == 0 really sufficient for checking whether fraction
    // was set yet
    if (c2[i] != 0 && ratio.num == 0) {
      ratio.num = c1[i];
      ratio.den = c2[i];
      if (ratio.den * c1[i] != ratio.num * c2[i]) {
        return false;
      }
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
  // relax t by 1 and add all other constraints of A to new_set.
  // TODO: is looping only over the inequalities sufficient here?
  // Reasoning: if A had an equality, we would be in a different case.
  SmallVector<SmallVector<int64_t, 8>, 8> new_set_inequalities;
  for (size_t k = 0; k < A.getNumInequalities(); k++) {
    SmallVector<int64_t, 8> curr = arrayRefToSmallVector(A.getInequality(k));
    if (!sameConstraint(t, curr)) {
      new_set_inequalities.push_back(arrayRefToSmallVector(A.getInequality(k)));
    }
  }
  shift(t, 1);
  new_set_inequalities.push_back(t);
  FlatAffineConstraints new_set(A.getNumDimIds(), A.getNumSymbolIds());
  for (size_t k = 0; k < new_set_inequalities.size(); k++) {
    new_set.addInequality(new_set_inequalities[k]);
  }

  // for every constraint c of B, check if it is redundant for the new_set with
  // t added as an equality. This makes sure, that B actually is a simple
  // extension of A.
  Simplex simp(new_set);
  simp.addEquality(t);
  SmallVector<ArrayRef<int64_t>, 8> constraints_b, equalities_b;
  for (size_t k = 0; k < B.getNumEqualities(); k++) {
    equalities_b.push_back(B.getEquality(k));
  }
  for (size_t k = 0; k < B.getNumInequalities(); k++) {
    constraints_b.push_back(B.getInequality(k));
  }
  addAsIneq(equalities_b, constraints_b);
  for (size_t k = 0; k < constraints_b.size(); k++) {
    if (simp.ineqType(constraints_b[k]) != Simplex::IneqType::REDUNDANT) {
      return false;
    }
  }
  addCoalescedBasicSet(basicSetVector, i, j, new_set);
  return true;
}

bool adjEqCaseNonPure(SmallVectorImpl<FlatAffineConstraints> &basicSetVector,
                      unsigned i, unsigned j, const Info &info_a,
                      const Info &info_b) {
  FlatAffineConstraints &a = basicSetVector[i];
  FlatAffineConstraints &b = basicSetVector[j];
  SmallVector<SmallVector<int64_t, 8>, 8> wrapped;
  SmallVector<int64_t, 8> minusT;
  // the constraint of a adjacent to an equality, it the complement of the
  // constraint f b, that is part of an equality and adjacent to an inequality.
  SmallVector<int64_t, 8> t =
      complement(arrayRefToSmallVector(info_b.t.getValue()));
  for (size_t k = 0; k < t.size(); k++) {
    minusT.push_back(-t[k]);
  }

  // TODO: can only cut be non_redundant?
  // The cut constraints of a are wrapped around -t to include B.
  for (size_t k = 0; k < info_a.cut.size(); k++) {
    // TODO: why does the pure case differ here in that it doesn't wrap t?
    SmallVector<int64_t, 8> curr = arrayRefToSmallVector(info_a.cut[k]);
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
  // TODO: can only cut be non_redundant?
  // the cut constraints of b (except -t - 1) are wrapped around t + 1 to
  // include a.
  for (size_t k = 0; k < info_b.cut.size(); k++) {
    if (!sameConstraint(minusT, info_b.cut[k])) {
      SmallVector<int64_t, 8> curr = arrayRefToSmallVector(info_b.cut[k]);
      auto result = wrapping(a, t, curr);
      if (!result)
        return false;
      wrapped.push_back(result.getValue());
    }
  }

  FlatAffineConstraints new_set(b.getNumIds(), b.getNumSymbolIds());
  // The new polytope consists of all the wrapped constraints and all the
  // redundant constraints
  for (size_t k = 0; k < wrapped.size(); k++) {
    new_set.addInequality(wrapped[k]);
  }
  addInequalities(new_set, info_a.redundant);
  addInequalities(new_set, info_b.redundant);
  // additionally, t + 1 is added.
  new_set.addInequality(t);

  addCoalescedBasicSet(basicSetVector, i, j, new_set);
  return true;
}

bool adjEqCasePure(SmallVectorImpl<FlatAffineConstraints> &basicSetVector,
                   unsigned i, unsigned j, const Info &info_a,
                   const Info &info_b) {
  FlatAffineConstraints &a = basicSetVector[i];
  FlatAffineConstraints &b = basicSetVector[j];
  SmallVector<SmallVector<int64_t, 8>, 8> wrapped;
  SmallVector<int64_t, 8> minusT;
  SmallVector<int64_t, 8> t = arrayRefToSmallVector(info_a.t.getValue());
  for (size_t k = 0; k < t.size(); k++) {
    minusT.push_back(-t[k]);
  }

  // TODO: can only cut be non_redundant?
  // The cut constraints of a (except t) are wrapped around -t to include B.
  for (size_t k = 0; k < info_a.cut.size(); k++) {
    if (!sameConstraint(t, info_a.cut[k])) {
      SmallVector<int64_t, 8> curr = arrayRefToSmallVector(info_a.cut[k]);
      auto result = wrapping(b, minusT, curr);
      if (!result)
        return false;
      wrapped.push_back(result.getValue());
    }
  }

  shift(t, 1);
  shift(minusT, -1);
  // TODO: can only cut be non_redundant?
  // the cut constraints of b (except -t -1) are wrapped around t+1 to include
  // a.
  for (size_t k = 0; k < info_b.cut.size(); k++) {
    if (!sameConstraint(minusT, info_b.cut[k])) {
      SmallVector<int64_t, 8> curr = arrayRefToSmallVector(info_b.cut[k]);
      auto result = wrapping(a, t, curr);
      if (!result)
        return false;
      wrapped.push_back(result.getValue());
    }
  }

  FlatAffineConstraints new_set(b.getNumIds(), b.getNumSymbolIds());
  // The new polytope consists of all the wrapped constraints and all the
  // redundant constraints
  for (size_t k = 0; k < wrapped.size(); k++) {
    new_set.addInequality(wrapped[k]);
  }
  addInequalities(new_set, info_a.redundant);
  addInequalities(new_set, info_b.redundant);
  // Additionally, the constraint t + 1 is added to the new BasicSet.
  new_set.addInequality(t);
  addCoalescedBasicSet(basicSetVector, i, j, new_set);
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
  if (!classify_ineq(simp, inequalities, info))
    return false;
  SmallVector<ArrayRef<int64_t>, 8> eqAsIneq;
  addAsIneq(equalities, eqAsIneq);
  for (ArrayRef<int64_t> current_constraint : eqAsIneq) {
    Simplex::IneqType ty = simp.ineqType(current_constraint);
    switch (ty) {
    case Simplex::IneqType::REDUNDANT:
      info.redundant.push_back(current_constraint);
      break;
    case Simplex::IneqType::CUT:
      info.cut.push_back(current_constraint);
      break;
    case Simplex::IneqType::ADJ_INEQ:
      if (info.adj_ineq)
        // if two adjacent constraints are found, we can surely not coalesce
        // this town is too small for two adj_ineq
        return false;
      info.adj_ineq = current_constraint;
      info.t = current_constraint;
      break;
    case Simplex::IneqType::ADJ_EQ:
      // TODO: possibly needs to change if simplex can handle adjacent to
      // equality
      info.t = current_constraint;
      break;
    case Simplex::IneqType::SEPARATE:
      // coalescing always failes when a separate constraint is encountered.
      return false;
    }
  }
  return true;
}

bool classify_ineq(Simplex &simp,
                   const SmallVector<ArrayRef<int64_t>, 8> &constraints,
                   Info &info) {
  for (ArrayRef<int64_t> current_constraint : constraints) {
    Simplex::IneqType ty = simp.ineqType(current_constraint);
    switch (ty) {
    case Simplex::IneqType::REDUNDANT:
      info.redundant.push_back(current_constraint);
      break;
    case Simplex::IneqType::CUT:
      info.cut.push_back(current_constraint);
      break;
    case Simplex::IneqType::ADJ_INEQ:
      if (info.adj_ineq)
        // if two adjacent constraints are found, we can surely not coalesce
        // this town is too small for two adj_ineq
        return false;
      info.adj_ineq = current_constraint;
      break;
    case Simplex::IneqType::ADJ_EQ:
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
                 unsigned i, unsigned j, const Info &info_a,
                 const Info &info_b) {
  ArrayRef<int64_t> t = info_a.adj_ineq.getValue();
  FlatAffineConstraints bs(basicSetVector[i].getNumDimIds(),
                           basicSetVector[i].getNumSymbolIds());
  addInequalities(bs, info_a.redundant);
  addInequalities(bs, info_a.cut);
  addInequalities(bs, info_b.redundant);

  // If all constraints of a are added but the adjacent one and all the
  // REDUNDANT ones from b, are all cut constraints of b now REDUNDANT?
  // If so, all REDUNDANT constraints of a and b together define the new
  // polytope
  Simplex comp(bs);
  comp.addInequality(complement(t));
  for (size_t k = 0; k < info_b.cut.size(); k++) {
    if (comp.ineqType(info_b.cut[k]) != Simplex::IneqType::REDUNDANT) {
      return false;
    }
  }
  if (info_b.adj_ineq) {
    if (comp.ineqType(info_b.adj_ineq.getValue()) !=
        Simplex::IneqType::REDUNDANT) {
      return false;
    }
  }
  FlatAffineConstraints newSet(basicSetVector[i].getNumDimIds(),
                               basicSetVector[i].getNumSymbolIds());
  addInequalities(newSet, info_a.redundant);
  addInequalities(newSet, info_b.redundant);
  addCoalescedBasicSet(basicSetVector, i, j, newSet);
  return true;
}

bool cutCase(SmallVector<FlatAffineConstraints, 4> &basicSetVector, unsigned i,
             unsigned j, const Info &info_a, const Info &info_b) {
  // if all facets are contained, the redundant constraints of both a and b
  // define the new polytope
  for (size_t k = 0; k < info_a.cut.size(); k++) {
    if (!containedFacet(info_a.cut[k], basicSetVector[i], info_b.cut)) {
      return false;
    }
  }
  FlatAffineConstraints new_set(basicSetVector[i].getNumIds(),
                                basicSetVector[i].getNumSymbolIds());
  addInequalities(new_set, info_a.redundant);
  addInequalities(new_set, info_b.redundant);
  addCoalescedBasicSet(basicSetVector, i, j, new_set);
  return true;
}

bool mlir::containedFacet(ArrayRef<int64_t> ineq,
                          const FlatAffineConstraints &bs,
                          const SmallVector<ArrayRef<int64_t>, 8> &cut) {
  Simplex simp(bs);
  simp.addEquality(ineq);
  for (size_t i = 0; i < cut.size(); i++) {
    if (simp.ineqType(cut[i]) != Simplex::IneqType::REDUNDANT) {
      return false;
    }
  }
  return true;
}

void addAsIneq(ArrayRef<ArrayRef<int64_t>> eq,
               SmallVectorImpl<ArrayRef<int64_t>> &target) {
  for (size_t i = 0; i < eq.size(); i++) {
    ArrayRef<int64_t> curr = eq[i];
    target.push_back(curr);
    // TODO: fix this memory leak
    SmallVector<int64_t, 8> *inverted = new SmallVector<int64_t, 8>();
    for (size_t j = 0; j < curr.size(); j++) {
      inverted->push_back(-curr[j]);
    }
    ArrayRef<int64_t> inverted_ref(*inverted);
    target.push_back(inverted_ref);
  }
}

void dumpInfo(const Info &info) {
  std::cout << "red:" << std::endl;
  for (size_t k = 0; k < info.redundant.size(); k++) {
    dump(info.redundant[k]);
  }
  std::cout << "cut:" << std::endl;
  for (size_t k = 0; k < info.cut.size(); k++) {
    dump(info.cut[k]);
  }
  if (info.adj_ineq) {
    std::cout << "adj_ineq:" << std::endl;
    dump(info.adj_ineq.getValue());
  }
  if (info.t) {
    std::cout << "t:" << std::endl;
    dump(info.t.getValue());
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
