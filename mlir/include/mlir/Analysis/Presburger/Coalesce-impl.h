#include "mlir/Analysis/Presburger/Coalesce.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Presburger/Set.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include <iostream>

#ifndef MLIR_ANALYSIS_PRESBURGER_COALESCE_IMPL_H
#define MLIR_ANALYSIS_PRESBURGER_COALESCE_IMPL_H

using namespace mlir;

template <typename Int>
bool containedFacet(ArrayRef<Int> ineq, const PresburgerBasicSet<Int> &bs,
                    const SmallVectorImpl<ArrayRef<Int>> &cut) {
  return containedFacet(ineq, bs, makeArrayRef(cut));
}

namespace mlir {
template <typename Int>
SmallVector<Int, 8> combineConstraint(const SmallVectorImpl<Int> &c1,
                                              const SmallVectorImpl<Int> &c2,
                                              Fraction<Int> &ratio) {
  return combineConstraint(makeArrayRef(c1), makeArrayRef(c2), ratio);
}

template <typename Int>
bool sameConstraint(const SmallVectorImpl<Int> &c1, const SmallVectorImpl<Int> &c2) {
  return sameConstraint(makeArrayRef(c1), makeArrayRef(c2));
}

template <typename Int>
bool sameConstraint(const SmallVectorImpl<Int> &c1, ArrayRef<Int> c2) {
  return sameConstraint(makeArrayRef(c1), c2);
}
}

/// TODO: look at MutableArrayRef
/// struct for classified constraints
/// redundant and cut are for constraints that are typed as Redundant or Cut
/// respectively. adjIneq is for any constraint, that is adjacent to the other
/// polytope. t is for any constraint, that is part of an equality constraint
/// and adjacent to the other polytope.
/// TODO: find better name than t
/// TODO: possibly change up structure of Info<Int>, if simplex manages to classify
/// adjacent to equality
/// TODO: rebuild the struct as soon as adjEq can be typed. Two sets can be
/// coalescable as long as the number of adjEqs isn't greater than the number of
/// dimensions.
template <typename Int>
struct Info {
  // This is a hack to fix the memory leak
  SmallVector<SmallVector<Int, 8>, 8> hack;
  SmallVector<ArrayRef<Int>, 8> redundant;
  SmallVector<ArrayRef<Int>, 8> cut;
  Optional<ArrayRef<Int>> adjIneq;
  Optional<ArrayRef<Int>> t;

  void dump() {
    std::cout << "red:" << std::endl;
    for (ArrayRef<Int> curr : this->redundant) {
      mlir::dump(curr);
    }
    std::cout << "cut:" << std::endl;
    for (ArrayRef<Int> curr : this->cut) {
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

/// computes the complement of t
/// i.e. for a given constraint t(x) >= 0 returns -t(x) -1 >= 0
template <typename Int>
SmallVector<Int, 8> complement(ArrayRef<Int> t);

/// shifts t by amount
/// i.e. for a given constraint t(x) >= 0 return t(x) + amount >= 0
template <typename Int>
void shift(SmallVectorImpl<Int> &t, int amount);

/// add eq as two inequalities to target
template <typename Int>
void addEqualitiesAsInequalities(const ArrayRef<ArrayRef<Int>> eq,
                                 SmallVectorImpl<ArrayRef<Int>> &target,
                                 Info<Int> &info);

template <typename Int>
void addEqualitiesAsInequalities(const SmallVectorImpl<ArrayRef<Int>> &eq,
                                 SmallVectorImpl<ArrayRef<Int>> &target,
                                 Info<Int> &info) {
  return addEqualitiesAsInequalities(makeArrayRef(eq), target, info);
}

/// adds all Equalities to bs
template <typename Int>
void addEqualities(PresburgerBasicSet<Int> &bs,
                   ArrayRef<ArrayRef<Int>> equalities);

/// adds all Inequalities to bs
template <typename Int>
void addInequalities(PresburgerBasicSet<Int> &bs,
                     ArrayRef<ArrayRef<Int>> inequalities);

template <typename Int>
void addInequalities(PresburgerBasicSet<Int> &bs,
                     const SmallVectorImpl<ArrayRef<Int>> &inequalities) {
  return addInequalities(bs, makeArrayRef(inequalities));
}

/// only gets called by classify
/// classifies all constraints into redundant, cut or adjIneq according to the
/// ineqType that the simplex returns.
///
/// returns true if it has neither encountered a separate constraint nor more
/// than one adjacent inequality. This has a similar idea as short-circuiting
/// behind it. The moment either a separate or two adjacent constraints are
/// encountered, the polytope cannot be coalesced anymore, so we can move to the
/// next tuple.
template <typename Int>
bool classifyIneq(Simplex<Int> &simp, ArrayRef<ArrayRef<Int>> constraints,
                  Info<Int> &info);

/// classifies all constraints into redundant, cut, adjIneq or t, where t stands
/// for a constraint adjacent to a the other polytope
///
/// returns true if it has neither encountered a separate constraint nor more
/// than one adjacent inequality. This has a similar idea as short-circuiting
/// behind it. The moment either a separate or two adjacent constraints are
/// encountered, the polytope cannot be coalesced anymore, so we can move to the
/// next tuple.
template <typename Int>
bool classify(Simplex<Int> &simp, ArrayRef<ArrayRef<Int>> inequalities,
              ArrayRef<ArrayRef<Int>> equalities, Info<Int> &info);

template <typename Int>
bool classify(Simplex<Int> &simp, const SmallVector<ArrayRef<Int>, 8> &inequalities,
              const SmallVector<ArrayRef<Int>, 8> &equalities, Info<Int> &info) {
  return classify(simp, makeArrayRef(inequalities), makeArrayRef(equalities), info);
}
/// compute the protrusionCase and return whether it has worked
///
/// In the protusionCase, only Cut constraints and Redundant constraints can
/// exist. This case differs from the CutCase in that it can still coalesce
/// polytopes, as long as they are not sticking out of each other by too much
/// (i.e. by less than 2). As we are considering an integer setting, the convex
/// hull of the union of those two polytopes actually is the same thing as the
/// union of the two polytope.
///      _____          _____
///    _|___  |        /     |
///   | |   | |   ==> |      |
///   | |   | |       |      |
///   |_|___|_|       |______|
///
/// TODO: find better example
template <typename Int>
bool protrusionCase(SmallVectorImpl<PresburgerBasicSet<Int>> &basicSetVector,
                    Info<Int> &infoA, const Info<Int> &infoB, unsigned i, unsigned j);

/// compute, whether a constraint of cut sticks out of bs by more than 2
template <typename Int>
bool stickingOut(ArrayRef<ArrayRef<Int>> cut,
                 const PresburgerBasicSet<Int> &bs);

template <typename Int>
bool stickingOut(const SmallVectorImpl<ArrayRef<Int>> &cut,
                 const PresburgerBasicSet<Int> &bs) {
  return stickingOut(makeArrayRef(cut), bs);
}

/// adds a PresburgerBasicSet<Int> and removes the sets at i and j.
template <typename Int>
void addCoalescedBasicSet(SmallVectorImpl<PresburgerBasicSet<Int>> &basicSetVector,
                          unsigned i, unsigned j, const PresburgerBasicSet<Int> &bs);

/// compute the cut case and return whether it has worked.
///
/// The cut case is the case, for which a polytope only has Redundant and Cut
/// constraints. If all the facets of such cut constraints are contained within
/// the other polytope, the polytopes can be combined to a polytope only
/// limited by all the Redundant constraints.
///    ___________        ___________
///   /   /  |   /       /          /
///   \   \  |  /   ==>  \         /
///    \   \ | /          \       /
///     \___\|/            \_____/
///
///
template <typename Int>
bool cutCase(SmallVectorImpl<PresburgerBasicSet<Int>> &basicSetVector, unsigned i,
             unsigned j, const Info<Int> &infoA, const Info<Int> &infoB);

/// compute adjIneq pure Case and return whether it has worked.
///
/// The adjIneq pure case can be viewed as a single polytope originally (below:
/// on the right), that was cut into two parts by a strip with width 1.
/// This is computed by just using all the Redundant constraints.
///  ________ ____            ______________
/// |       //    |          |              |
/// |      //     |          |              |
/// |     //      |    ==>   |              |
/// |    //       |          |              |
/// |___//________|          |______________|
///
template <typename Int>
bool adjIneqPureCase(SmallVectorImpl<PresburgerBasicSet<Int>> &basicSetVector,
                     unsigned i, unsigned j, const Info<Int> &infoA,
                     const Info<Int> &infoB);

/// compute the non-pure adjIneq case and return whether it has worked.
/// Constraint t is the adjIneq.
///
/// In the non-pure adjIneq case, one of the polytopes is like an extension of
/// the other one. This can be computed, by inverting the adjIneq and checking
/// whether all constraints are stll valid for this new polytope.
///
/// In the example below, Polytope A is the big one on the left, Polytope B is
/// the small line at the right side. that gets included after coalescing.
///   ____          ____
///  |    \        |    \
///  |     \   ==> |     \
///  |_____|_      |______\
///
template <typename Int>
bool adjIneqCase(SmallVectorImpl<PresburgerBasicSet<Int>> &basicSetVector,
                 unsigned i, unsigned j, const Info<Int> &infoA, const Info<Int> &infoB);

/// compute the pure adjEqCase and return whether it has worked.
///
/// The pure case consists of two equalities, that are adjacent. Such equalities
/// can always be coalesced by finding the two constraints, that make them
/// become a trapezoid. This is done by wrapping the limiting constraints of the
/// equalities around each other to include the other one.
///                _
///    / /        / /
///   / /   ==>  / /
///  / /        /_/
///
template <typename Int>
bool adjEqCasePure(SmallVectorImpl<PresburgerBasicSet<Int>> &basicSetVector,
                   unsigned i, unsigned j, const Info<Int> &infoA,
                   const Info<Int> &infoB);

/// compute the adjEq Case for no Cut constraints.
///
/// The adjEq no cut case is simply an extension case, where the constraint,
/// that is adjacent to the equality, can be relaxed by 1 to include the other
/// polytope.
///   ______           _______
///  |      ||        |       |
///  |      ||        |       |
///  |      ||   ==>  |       |
///  |______||        |_______|
///
template <typename Int>
bool adjEqCaseNoCut(SmallVectorImpl<PresburgerBasicSet<Int>> &basicSetVector,
                    unsigned i, unsigned j, SmallVector<Int, 8> t,
                    Info<Int> &infoB);

/// compute the non-pure adjEqCase and return whether it has worked.
///
/// The non-pure case has cut constraints such that it is not a simple extension
/// like the no cut case. It is computed by wrapping those cut constraints and
/// checking, whether everything stilly holds.
///     ________           ________
///    |        |         |        \
///    |        | |  ==>  |         |
///    |        |         |        /
///    |        /         |       /
///    |_______/          |______/
///
template <typename Int>
bool adjEqCaseNonPure(SmallVectorImpl<PresburgerBasicSet<Int>> &basicSetVector,
                      unsigned i, unsigned j, const Info<Int> &infoA,
                      const Info<Int> &infoB);

template <typename Int>
void shift(SmallVectorImpl<Int> &t, int amount) {
  t.push_back(t.pop_back_val() + amount);
}

template <typename Int>
SmallVector<Int, 8> complement(ArrayRef<Int> t) {
  SmallVector<Int, 8> complement;
  for (unsigned k = 0; k < t.size() - 1; k++) {
    complement.push_back(-t[k]);
  }
  complement.push_back(-t.back() - 1);
  return complement;
}

/// returns all Equalities of a BasicSet as a SmallVector of ArrayRefs
template <typename Int>
void getBasicSetEqualities(const PresburgerBasicSet<Int> &bs,
                           SmallVectorImpl<ArrayRef<Int>> &eqs) {
  for (unsigned k = 0; k < bs.getNumEqualities(); k++) {
    eqs.push_back(bs.getEquality(k).getCoeffs());
  }
}

/// returns all Inequalities of a BasicSet as a SmallVector of ArrayRefs
template <typename Int>
void getBasicSetInequalities(const PresburgerBasicSet<Int> &bs,
                             SmallVectorImpl<ArrayRef<Int>> &ineqs) {
  for (unsigned k = 0; k < bs.getNumInequalities(); k++) {
    ineqs.push_back(bs.getInequality(k).getCoeffs());
  }
}

template <typename Int>
bool coalescePair(unsigned i, unsigned j,
                  SmallVector<PresburgerBasicSet<Int>, 4> &basicSetVector) {
  PresburgerBasicSet<Int> &bs1 = basicSetVector[i];
  PresburgerBasicSet<Int> &bs2 = basicSetVector[j];

  PresburgerBasicSet<Int>::alignDivs(bs1, bs2);

  Simplex simplex1(bs1);
  SmallVector<ArrayRef<Int>, 8> equalities1, inequalities1;
  getBasicSetEqualities(bs1, equalities1);
  getBasicSetInequalities(bs1, inequalities1);

  Simplex simplex2(bs2);
  SmallVector<ArrayRef<Int>, 8> equalities2, inequalities2;
  getBasicSetEqualities(bs2, equalities2);
  getBasicSetInequalities(bs2, inequalities2);

  Info<Int> info1, info2;
  if (!classify(simplex2, inequalities1, equalities1, info1))
    return false;
  if (!classify(simplex1, inequalities2, equalities2, info2))
    return false;
  // TODO: find better strategy than i--; break;
  // change indices when changing vector?
  if (!info1.redundant.empty() && info1.cut.empty() && !info1.adjIneq &&
      !info2.t) {
    // contained 2 in 1
    addCoalescedBasicSet(basicSetVector, i, j, basicSetVector[i]);
    return true;
  } else if (!info2.redundant.empty() && info2.cut.empty() && !info2.adjIneq &&
             !info1.t) {
    // contained 1 in 2
    addCoalescedBasicSet(basicSetVector, i, j, basicSetVector[j]);
    return true;
  } else if (!info1.redundant.empty() && !info1.cut.empty() && !info1.adjIneq &&
             !info2.t) {
    // cut or protrusion case 1
    if (cutCase(basicSetVector, i, j, info1, info2)) {
      return true;
    } else if (stickingOut(info1.cut, bs2) &&
               protrusionCase(basicSetVector, info1, info2, i, j)) {
      // protrusion
      return true;
    }
  } else if (!info2.redundant.empty() && !info2.cut.empty() && !info2.adjIneq &&
             !info1.t) {
    // cut or protrusion case 2
    if (cutCase(basicSetVector, j, i, info2, info1)) {
      return true;
    } else if (stickingOut(info2.cut, bs1) &&
               protrusionCase(basicSetVector, info2, info1, j, i)) {
      // protrusion case
      return true;
    }
  } else if (!info1.redundant.empty() && info1.adjIneq && info1.cut.empty() &&
             !info2.t && !info2.redundant.empty() && info2.adjIneq &&
             info2.cut.empty() && !info1.t) {
    // adjIneq, pure case
    if (adjIneqPureCase(basicSetVector, i, j, info1, info2)) {
      return true;
    }
  } else if (!info1.redundant.empty() && info1.adjIneq && info1.cut.empty() &&
             !info1.t && !info2.t) {
    // adjIneq complex case 1
    if (adjIneqCase(basicSetVector, i, j, info1, info2)) {
      return true;
    }
  } else if (!info2.redundant.empty() && info2.adjIneq && info2.cut.empty() &&
             !info2.t && !info1.t) {
    // adjIneq complex case 2
    if (adjIneqCase(basicSetVector, j, i, info2, info1)) {
      return true;
    }
  } else if (info1.t && info2.t) {
    // adjEq for two equalities
    if (adjEqCasePure(basicSetVector, i, j, info1, info2)) {
      return true;
    }
  } else if (info1.t && info2.cut.empty()) {
    // adjEq Case for one equality 1
    // compute the inequality, that is adjacent to an equality by computing
    // the complement of the inequality part of an equality
    SmallVector<Int, 8> adjEq = complement(info1.t.getValue());
    if (adjEqCaseNoCut(basicSetVector, i, j, adjEq, info2)) {
      // adjEq noCut case
      return true;
    } else if (info1.t &&
               adjEqCaseNonPure(basicSetVector, j, i, info2, info1)) {
      // adjEq case
      return true;
    }
  } else if (info2.t && info1.cut.empty()) {
    // adjEq Case for one equality 2
    // compute the inequality, that is adjacent to an equality by computing
    // the complement of the inequality part of an equality
    SmallVector<Int, 8> adjEq = complement(info2.t.getValue());
    if (adjEqCaseNoCut(basicSetVector, j, i, adjEq, info1)) {
      // adjEq noCut case
      return true;
    } else if (info2.t &&
               adjEqCaseNonPure(basicSetVector, i, j, info1, info2)) {
      // adjEq case
      return true;
    }
  }

  return false;
}

// Intersect the two basic sets and drop the subset if one is a subset of
/// another
template <typename Int>
bool coalesceDivs(unsigned i, unsigned j,
                  SmallVector<PresburgerBasicSet<Int>, 4> &basicSetVector) {
  PresburgerBasicSet<Int> bs1 = basicSetVector[i];
  PresburgerBasicSet<Int> bs2 = basicSetVector[j];

  PresburgerBasicSet<Int> intersect = bs1;
  intersect.intersect(bs2);
  intersect.simplify();

// TODO : Equal is very expensive, try to use something else
  if (PresburgerSet<Int>::equal(PresburgerSet(intersect), PresburgerSet(bs2))) {
    addCoalescedBasicSet(basicSetVector, i, j, basicSetVector[i]);
    return true;
  }

  if (PresburgerSet<Int>::equal(PresburgerSet(intersect), PresburgerSet(bs1))) {
    addCoalescedBasicSet(basicSetVector, i, j, basicSetVector[j]);
    return true;
  }

  return false;
}

template <typename Int>
PresburgerSet<Int> mlir::coalesce(PresburgerSet<Int> &set) {
  set.simplify();

  PresburgerSet<Int> newSet(set.getNumDims(), set.getNumSyms());
  SmallVector<PresburgerBasicSet<Int>, 4> basicSetVector = set.getBasicSets();

  for (unsigned i = 0; i < basicSetVector.size(); ++i) {
    for (unsigned j = i + 1; j < basicSetVector.size(); ++j) {

      PresburgerBasicSet<Int> bs1 = basicSetVector[i];
      PresburgerBasicSet<Int> bs2 = basicSetVector[j];

      bool coalesceStatus = coalescePair(i, j, basicSetVector);
      if (coalesceStatus) {
        i--;
        break;
      } else {
        // Revert changes
        basicSetVector[i] = bs1;
        basicSetVector[j] = bs2;
      }

      // If no existentials or divisions, there is no point in checking ahead
      if (bs1.getNumDivs() + bs1.getNumExists() == 0 &&
          bs2.getNumDivs() + bs2.getNumExists() == 0)
        continue;

      coalesceStatus = coalesceDivs(i, j, basicSetVector);
      if (coalesceStatus) {
        i--;
        break;
      } else {
        // Revert changes
        basicSetVector[i] = bs1;
        basicSetVector[j] = bs2;
      }
    }
  }

  for (PresburgerBasicSet<Int> &curr : basicSetVector)
    newSet.addBasicSet(curr);
  return newSet;
}

template <typename Int>
void addInequalities(PresburgerBasicSet<Int> &bs,
                     ArrayRef<ArrayRef<Int>> inequalities) {
  for (unsigned k = 0; k < inequalities.size(); k++) {
    bs.addInequality(inequalities[k]);
  }
}

template <typename Int>
void addEqualities(PresburgerBasicSet<Int> &bs,
                   ArrayRef<ArrayRef<Int>> equalities) {
  for (unsigned k = 0; k < equalities.size(); k++) {
    bs.addEquality(equalities[k]);
  }
}

template <typename Int>
bool adjIneqPureCase(SmallVectorImpl<PresburgerBasicSet<Int>> &basicSetVector,
                     unsigned i, unsigned j, const Info<Int> &infoA,
                     const Info<Int> &infoB) {
  PresburgerBasicSet<Int> newSet(
      basicSetVector[i].getNumDims(), basicSetVector[i].getNumParams(),
      basicSetVector[i].getNumDivs() + basicSetVector[i].getNumExists());
  addInequalities(newSet, infoA.redundant);
  addInequalities(newSet, infoB.redundant);
  addCoalescedBasicSet(basicSetVector, i, j, newSet);
  return true;
}

/// Currently erases both i and j and the pushes the new BasicSet to the back of
/// the vector.
/// TODO: probably change when changing looping strategy
template <typename Int>
void addCoalescedBasicSet(SmallVectorImpl<PresburgerBasicSet<Int>> &basicSetVector,
                          unsigned i, unsigned j,
                          const PresburgerBasicSet<Int> &bs) {
  PresburgerBasicSet newSet(bs.getNumDims(), bs.getNumParams(),
                          basicSetVector[i].getNumExists(),
                          basicSetVector[i].getDivisions());

  for (const EqualityConstraint<Int> &eq : bs.getEqualities())
    newSet.addEquality(eq.getCoeffs());

  for (const InequalityConstraint<Int> &ineq : bs.getInequalities())
    newSet.addInequality(ineq.getCoeffs());

  if (i < j) {
    basicSetVector.erase(basicSetVector.begin() + j);
    basicSetVector.erase(basicSetVector.begin() + i);
  } else {
    basicSetVector.erase(basicSetVector.begin() + i);
    basicSetVector.erase(basicSetVector.begin() + j);
  }
  
  // Simplify the newly created set
  newSet.simplify();
  basicSetVector.push_back(newSet);
}

template <typename Int>
bool protrusionCase(SmallVectorImpl<PresburgerBasicSet<Int>> &basicSetVector,
                    Info<Int> &infoA, const Info<Int> &infoB, unsigned i, unsigned j) {
  PresburgerBasicSet<Int> &a = basicSetVector[i];
  PresburgerBasicSet<Int> &b = basicSetVector[j];
  SmallVector<ArrayRef<Int>, 8> constraintsB, equalitiesB;
  getBasicSetEqualities(b, equalitiesB);
  getBasicSetInequalities(b, constraintsB);
  addEqualitiesAsInequalities(equalitiesB, constraintsB, infoA);

  // For every cut constraint t of a, bPrime is computed as b intersected with
  // t(x) + 1 = 0. For this, t is shifted by 1.
  SmallVector<SmallVector<Int, 8>, 8> wrapped;
  for (unsigned l = 0; l < infoA.cut.size(); l++) {
    auto t = llvm::to_vector<8>(infoA.cut[l]);
    PresburgerBasicSet<Int> bPrime = PresburgerBasicSet<Int>(
        b.getNumDims(), b.getNumParams(), b.getNumDivs() + b.getNumExists());

    addInequalities(bPrime, constraintsB);
    shift(t, 1);
    Simplex<Int> simp(bPrime);
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
      for (ArrayRef<Int> currCut : infoB.cut) {
        auto curr1 = llvm::to_vector<8>(currCut);
        Simplex<Int> simp2(bPrime);
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

  PresburgerBasicSet<Int> newSet(b.getNumDims(), b.getNumParams(),
                                 b.getNumDivs() + b.getNumExists());
  // If all the wrappings were succesfull, the two polytopes can be replaced by
  // a polytope with all of the redundant constraints and the wrapped
  // constraints.
  addInequalities(newSet, infoA.redundant);
  addInequalities(newSet, infoB.redundant);
  for (const SmallVector<Int, 8> &curr : wrapped) {
    newSet.addInequality(curr);
  }
  // Additionally for every remaining cut constraint t of a, t + 1 >= 0 is
  // added.
  for (ArrayRef<Int> currRef : infoA.cut) {
    auto curr = llvm::to_vector<8>(currRef);
    shift(curr, 1);
    newSet.addInequality(curr);
  }
  addCoalescedBasicSet(basicSetVector, i, j, newSet);
  return true;
}

template <typename Int>
bool stickingOut(ArrayRef<ArrayRef<Int>> cut,
                 const PresburgerBasicSet<Int> &bs) {
  Simplex<Int> simp(bs);
  // for every cut constraint t, compute the optimum in the direction of cut. If
  // the optimum is < -2, the polytopee doesn't stick too much out of the other
  // one.
  for (ArrayRef<Int> curr : cut) {
    auto result = simp.computeOptimum(Simplex<Int>::Direction::Down, curr);
    if (!result) {
      return false;
    }
    auto res = result.getValue();
    if (res <= Fraction<Int>(-2, 1)) {
      return false;
    }
  }
  return true;
}

template <typename Int>
bool mlir::sameConstraint(ArrayRef<Int> c1, ArrayRef<Int> c2) {
  Fraction<Int> ratio(0, 1);
  assert(c1.size() == c2.size() && "the constraints have different dimensions");
  // if c1 = a*c2, this iterates over the vector trying to find a as soon as
  // possible and then comparing with a.
  for (unsigned i = 0; i < c1.size(); i++) {
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

template <typename Int>
bool adjEqCaseNoCut(SmallVectorImpl<PresburgerBasicSet<Int>> &basicSetVector,
                    unsigned i, unsigned j, SmallVector<Int, 8> t,
                    Info<Int> &infoB) {
  PresburgerBasicSet<Int> &A = basicSetVector[j];
  PresburgerBasicSet<Int> &B = basicSetVector[i];
  // relax t by 1 and add all other constraints of A to newSet.
  // TODO: is looping only over the inequalities sufficient here?
  // Reasoning: if A had an equality, we would be in a different case.
  SmallVector<SmallVector<Int, 8>, 8> newSetInequalities;
  for (unsigned k = 0; k < A.getNumInequalities(); k++) {
    auto curr = llvm::to_vector<8>(A.getInequality(k).getCoeffs());
    if (!sameConstraint(t, curr)) {
      newSetInequalities.push_back(
          llvm::to_vector<8>(A.getInequality(k).getCoeffs()));
    }
  }
  shift(t, 1);
  newSetInequalities.push_back(t);
  PresburgerBasicSet<Int> newSet(A.getNumDims(), A.getNumParams(),
                                 A.getNumDivs() + A.getNumExists());
  for (const SmallVector<Int, 8> &curr : newSetInequalities) {
    newSet.addInequality(curr);
  }

  // for every constraint c of B, check if it is redundant for the newSet with
  // t added as an equality. This makes sure, that B actually is a simple
  // extension of A.
  Simplex<Int> simp(newSet);
  simp.addEquality(t);
  SmallVector<ArrayRef<Int>, 8> constraintsB, equalitiesB;
  for (unsigned k = 0; k < B.getNumEqualities(); k++) {
    equalitiesB.push_back(B.getEquality(k).getCoeffs());
  }
  for (unsigned k = 0; k < B.getNumInequalities(); k++) {
    constraintsB.push_back(B.getInequality(k).getCoeffs());
  }
  addEqualitiesAsInequalities(equalitiesB, constraintsB, infoB);
  for (ArrayRef<Int> curr : constraintsB) {
    if (simp.ineqType(curr) != Simplex<Int>::IneqType::Redundant) {
      return false;
    }
  }
  addCoalescedBasicSet(basicSetVector, i, j, newSet);
  return true;
}

template <typename Int>
bool adjEqCaseNonPure(SmallVectorImpl<PresburgerBasicSet<Int>> &basicSetVector,
                      unsigned i, unsigned j, const Info<Int> &infoA,
                      const Info<Int> &infoB) {
  PresburgerBasicSet<Int> &a = basicSetVector[i];
  PresburgerBasicSet<Int> &b = basicSetVector[j];
  SmallVector<SmallVector<Int, 8>, 8> wrapped;
  SmallVector<Int, 8> minusT;
  // the constraint of a adjacent to an equality, it the complement of the
  // constraint f b, that is part of an equality and adjacent to an inequality.
  SmallVector<Int, 8> t =
      complement(infoB.t.getValue());
  for (const Int n : t) {
    minusT.push_back(-n);
  }

  // TODO: can only cut be non-redundant?
  // The cut constraints of a are wrapped around -t to include B.
  for (ArrayRef<Int> currRef : infoA.cut) {
    // TODO: why does the pure case differ here in that it doesn't wrap t?
    auto curr = llvm::to_vector<8>(currRef);
    auto result = wrapping(b, minusT, curr);
    if (!result)
      return false;
    wrapped.push_back(result.getValue());
  }

  // Some of the wrapped constraints can now be non redudant.
  Simplex<Int> simp(b);
  for (const SmallVector<Int, 8> &curr : wrapped) {
    if (simp.ineqType(curr) != Simplex<Int>::IneqType::Redundant) {
      return false;
    }
  }

  shift(t, 1);
  shift(minusT, -1);
  // TODO: can only cut be non-redundant?
  // the cut constraints of b (except -t - 1) are wrapped around t + 1 to
  // include a.
  for (ArrayRef<Int> currRef : infoB.cut) {
    if (!sameConstraint(minusT, currRef)) {
      auto curr = llvm::to_vector<8>(currRef);
      auto result = wrapping(a, t, curr);
      if (!result)
        return false;
      wrapped.push_back(result.getValue());
    }
  }

  PresburgerBasicSet<Int> newSet(b.getNumDims(), b.getNumParams(),
                                 b.getNumDivs() + b.getNumExists());
  // The new polytope consists of all the wrapped constraints and all the
  // redundant constraints
  for (const SmallVector<Int, 8> &curr : wrapped) {
    newSet.addInequality(curr);
  }
  addInequalities(newSet, infoA.redundant);
  addInequalities(newSet, infoB.redundant);
  // additionally, t + 1 is added.
  newSet.addInequality(t);

  addCoalescedBasicSet(basicSetVector, i, j, newSet);
  return true;
}

template <typename Int>
bool adjEqCasePure(SmallVectorImpl<PresburgerBasicSet<Int>> &basicSetVector,
                   unsigned i, unsigned j, const Info<Int> &infoA,
                   const Info<Int> &infoB) {

  if (infoA.adjIneq && infoB.adjIneq)
    return false;

  PresburgerBasicSet<Int> &a = basicSetVector[i];
  PresburgerBasicSet<Int> &b = basicSetVector[j];
  SmallVector<SmallVector<Int, 8>, 8> wrapped;
  SmallVector<Int, 8> minusT;
  auto t = llvm::to_vector<8>(infoA.t.getValue());
  for (const Int n : t) {
    minusT.push_back(-n);
  }

  // TODO: can only cut be non-redundant?
  // The cut constraints of a (except t) are wrapped around -t to include B.
  for (ArrayRef<Int> currRef : infoA.cut) {
    if (!sameConstraint(t, currRef)) {
      auto curr = llvm::to_vector<8>(currRef);
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
  for (ArrayRef<Int> currRef : infoB.cut) {
    if (!sameConstraint(minusT, currRef)) {
      auto curr = llvm::to_vector<8>(currRef);
      auto result = wrapping(a, t, curr);
      if (!result)
        return false;
      wrapped.push_back(result.getValue());
    }
  }

  PresburgerBasicSet<Int> newSet(b.getNumDims(), b.getNumParams(),
                                 b.getNumDivs() + b.getNumExists());
  // The new polytope consists of all the wrapped constraints and all the
  // redundant constraints
  for (const SmallVector<Int, 8> &curr : wrapped) {
    newSet.addInequality(curr);
  }
  addInequalities(newSet, infoA.redundant);
  addInequalities(newSet, infoB.redundant);
  // Additionally, the constraint t + 1 is added to the new BasicSet.
  newSet.addInequality(t);
  addCoalescedBasicSet(basicSetVector, i, j, newSet);
  return true;
}

template <typename Int>
SmallVector<Int, 8> mlir::combineConstraint(ArrayRef<Int> c1,
                                                    ArrayRef<Int> c2,
                                                    Fraction<Int> &ratio) {
  assert(c1.size() == c2.size() && "dimensions must be equal");
  Int n = ratio.num;
  Int d = ratio.den;
  SmallVector<Int, 8> result;
  for (unsigned i = 0; i < c1.size(); i++) {
    result.push_back(-n * c1[i] + d * c2[i]);
  }
  return result;
}

template <typename Int>
Optional<SmallVector<Int, 8>>
mlir::wrapping(const PresburgerBasicSet<Int> &bs,
               SmallVectorImpl<Int> &valid,
               SmallVectorImpl<Int> &invalid) {
  assert(valid.size() == invalid.size() && "dimensions must be equal");
  unsigned n = bs.getNumTotalDims();
  Simplex<Int> simplex(n + 1);

  // for every constraint t(x) + c of bs, make it become t(x) + c*lambda
  for (unsigned k = 0; k < bs.getNumEqualities(); k++) {
    auto curr = llvm::to_vector<8>(bs.getEquality(k).getCoeffs());
    curr.push_back(0);
    simplex.addEquality(curr);
  }
  for (unsigned k = 0; k < bs.getNumInequalities(); k++) {
    auto curr = llvm::to_vector<8>(bs.getInequality(k).getCoeffs());
    curr.push_back(0);
    simplex.addInequality(curr);
  }

  // add lambda >= 0
  // TODO: why does it work with n here? Isn't this constraint actually n+2
  // long?
  SmallVector<Int, 8> lambda(n, 0);
  lambda.push_back(1);
  lambda.push_back(0);
  simplex.addInequality(lambda);

  // make the valid constraint be equal to 1 and add it as an equality
  valid.push_back(-1);
  simplex.addEquality(valid);

  // transform the invalid constraint into lambda space
  invalid.push_back(0);
  Optional<Fraction<Int>> result =
      simplex.computeOptimum(Simplex<Int>::Direction::Down, invalid);
  if (!result) {
    return {};
  }

  // retransform valid and invalid into normal space before combining them.
  valid.pop_back();
  invalid.pop_back();
  return combineConstraint(valid, invalid, result.getValue());
}

template <typename Int>
bool classify(Simplex<Int> &simp, ArrayRef<ArrayRef<Int>> inequalities,
              ArrayRef<ArrayRef<Int>> equalities, Info<Int> &info) {
  if (!classifyIneq(simp, inequalities, info))
    return false;
  SmallVector<ArrayRef<Int>, 8> eqAsIneq;
  addEqualitiesAsInequalities(equalities, eqAsIneq, info);
  for (ArrayRef<Int> currentConstraint : eqAsIneq) {
    typename Simplex<Int>::IneqType ty = simp.ineqType(currentConstraint);
    switch (ty) {
    case Simplex<Int>::IneqType::Redundant:
      info.redundant.push_back(currentConstraint);
      break;
    case Simplex<Int>::IneqType::Cut:
      info.cut.push_back(currentConstraint);
      break;
    case Simplex<Int>::IneqType::AdjIneq:
      if (info.adjIneq)
        // if two adjacent constraints are found, we can surely not coalesce
        // this town is too small for two adjIneq
        return false;
      info.adjIneq = currentConstraint;
      info.t = currentConstraint;
      break;
    case Simplex<Int>::IneqType::AdjEq:
      // TODO: possibly needs to change if simplex can handle adjacent to
      // equality
      info.t = currentConstraint;
      break;
    case Simplex<Int>::IneqType::Separate:
      // coalescing always failes when a separate constraint is encountered.
      return false;
    }
  }
  return true;
}

template <typename Int>
bool classifyIneq(Simplex<Int> &simp, ArrayRef<ArrayRef<Int>> constraints,
                  Info<Int> &info) {
  for (ArrayRef<Int> currentConstraint : constraints) {
    typename Simplex<Int>::IneqType ty = simp.ineqType(currentConstraint);
    switch (ty) {
    case Simplex<Int>::IneqType::Redundant:
      info.redundant.push_back(currentConstraint);
      break;
    case Simplex<Int>::IneqType::Cut:
      info.cut.push_back(currentConstraint);
      break;
    case Simplex<Int>::IneqType::AdjIneq:
      if (info.adjIneq)
        // if two adjacent constraints are found, we can surely not coalesce
        // this town is too small for two adjIneq
        return false;
      info.adjIneq = currentConstraint;
      break;
    case Simplex<Int>::IneqType::AdjEq:
      // TODO: possibly needs to change if simplex can handle adjacent to
      // equality
      break;
    case Simplex<Int>::IneqType::Separate:
      // coalescing always failes when a separate constraint is encountered.
      return false;
    }
  }
  return true;
}

template <typename Int>
bool adjIneqCase(SmallVectorImpl<PresburgerBasicSet<Int>> &basicSetVector,
                 unsigned i, unsigned j, const Info<Int> &infoA, const Info<Int> &infoB) {
  ArrayRef<Int> t = infoA.adjIneq.getValue();
  PresburgerBasicSet<Int> bs(
      basicSetVector[i].getNumDims(), basicSetVector[i].getNumParams(),
      basicSetVector[i].getNumDivs() + basicSetVector[i].getNumExists());
  addInequalities(bs, infoA.redundant);
  addInequalities(bs, infoA.cut);
  addInequalities(bs, infoB.redundant);

  // If all constraints of a are added but the adjacent one and all the
  // Redundant ones from b, are all cut constraints of b now Redundant?
  // If so, all Redundant constraints of a and b together define the new
  // polytope
  Simplex<Int> comp(bs);
  comp.addInequality(complement(t));
  for (ArrayRef<Int> curr : infoB.cut) {
    if (comp.ineqType(curr) != Simplex<Int>::IneqType::Redundant) {
      return false;
    }
  }
  if (infoB.adjIneq) {
    if (comp.ineqType(infoB.adjIneq.getValue()) !=
        Simplex<Int>::IneqType::Redundant) {
      return false;
    }
  }
  PresburgerBasicSet<Int> newSet(
      basicSetVector[i].getNumDims(), basicSetVector[i].getNumParams(),
      basicSetVector[i].getNumDivs() + basicSetVector[i].getNumExists());
  addInequalities(newSet, infoA.redundant);
  addInequalities(newSet, infoB.redundant);
  addCoalescedBasicSet(basicSetVector, i, j, newSet);
  return true;
}

template <typename Int>
bool cutCase(SmallVectorImpl<PresburgerBasicSet<Int>> &basicSetVector, unsigned i,
             unsigned j, const Info<Int> &infoA, const Info<Int> &infoB) {
  // if all facets are contained, the redundant constraints of both a and b
  // define the new polytope
  for (ArrayRef<Int> curr : infoA.cut) {
    if (!containedFacet(curr, basicSetVector[i], infoB.cut)) {
      return false;
    }
  }
  PresburgerBasicSet<Int> newSet(
      basicSetVector[i].getNumDims(), basicSetVector[i].getNumParams(),
      basicSetVector[i].getNumDivs() + basicSetVector[i].getNumExists());
  addInequalities(newSet, infoA.redundant);
  addInequalities(newSet, infoB.redundant);
  addCoalescedBasicSet(basicSetVector, i, j, newSet);
  return true;
}

template <typename Int>
bool mlir::containedFacet(ArrayRef<Int> ineq,
                          const PresburgerBasicSet<Int> &bs,
                          ArrayRef<ArrayRef<Int>> cut) {
  Simplex<Int> simp(bs);
  simp.addEquality(ineq);
  for (ArrayRef<Int> curr : cut) {
    if (simp.ineqType(curr) != Simplex<Int>::IneqType::Redundant) {
      return false;
    }
  }
  return true;
}

template <typename Int>
void addEqualitiesAsInequalities(ArrayRef<ArrayRef<Int>> eq,
                                 SmallVectorImpl<ArrayRef<Int>> &target,
                                 Info<Int> &info) {
  for (ArrayRef<Int> curr : eq) {
    target.push_back(curr);
    // TODO: fix this memory leak
    SmallVector<Int, 8> inverted;
    for (const Int n : curr) {
      inverted.push_back(-n);
    }
    info.hack.push_back(inverted);
    ArrayRef<Int> invertedRef(info.hack.back());
    target.push_back(invertedRef);
  }
}

template <typename Int>
void mlir::dump(ArrayRef<Int> cons) {
  llvm::errs() << cons[cons.size() - 1] << " + ";
  for (unsigned i = 1; i < cons.size(); i++) {
    llvm::errs() << cons[i - 1] << "x" << i - 1;
    if (i == cons.size() - 1) {
      break;
    }
    llvm::errs() << " + ";
  }
  llvm::errs() << " >= 0\n";
}

#endif // MLIR_ANALYSIS_PRESBURGER_COALESCE_IMPL_H
