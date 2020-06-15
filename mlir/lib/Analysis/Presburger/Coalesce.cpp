#include "mlir/Analysis/Presburger/Coalesce.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Presburger/Set.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include <iostream>

using namespace mlir;
// struct for classified constraints
struct Info {
  SmallVector<SmallVector<int64_t, 8>, 8> redundant;
  SmallVector<SmallVector<int64_t, 8>, 8> cut;
  Optional<SmallVector<int64_t, 8>> adj_ineq;
  Optional<SmallVector<int64_t, 8>> t;
};

void complement(ArrayRef<int64_t> t, SmallVectorImpl<int64_t> &complement);

void shift(SmallVectorImpl<int64_t> &t, int amount);

void dumpInfo(Info &info);

// adds all Constraints to bs
void addEqualities(FlatAffineConstraints &bs,
                   const SmallVector<SmallVector<int64_t, 8>, 8> &equalities);

void addInequalities(
    FlatAffineConstraints &bs,
    const SmallVector<SmallVector<int64_t, 8>, 8> &inequalities);

// only gets called by classify
// classify of all constraints
//
// returns true if it has not encountered a separate constraints
bool classify_ineq(Simplex &simp,
                   const SmallVector<SmallVector<int64_t, 8>, 8> &constraints,
                   Info &info);

// same thing as classify_ineq, but also return if there is an equality
// constraint adjacent to a the other polytope
// returns true if it has not encountered a separate constraints
bool classify(Simplex &simp,
              const SmallVector<SmallVector<int64_t, 8>, 8> &inequalities,
              const SmallVector<SmallVector<int64_t, 8>, 8> &equalities,
              Info &info);

// compute the protrusionCase and return whether it has worked
bool protrusionCase(SmallVectorImpl<FlatAffineConstraints> &basicSetVector,
                    Info &info_a, Info &info_b, unsigned i, unsigned j);

// compute, whether a constraint of cut sticks out of bs by more than 2
bool stickingOut(SmallVector<SmallVector<int64_t, 8>, 8> cut,
                 FlatAffineConstraints &bs);

// add a FlatAffineConstraints and removes the sets at i and j
void addCoalescedBasicSet(
    SmallVectorImpl<FlatAffineConstraints> &basicSetVector, unsigned i,
    unsigned j, FlatAffineConstraints &bs);

// compute the cut case and return whether it has worked.
bool cutCase(SmallVector<FlatAffineConstraints, 4> &basicSetVector, unsigned i,
             unsigned j, Info &info_a, Info &info_b);

// compute adj_ineq pure Case and return whether it has worked
bool adjIneqPureCase(SmallVector<FlatAffineConstraints, 4> &basicSetVector,
                     unsigned i, unsigned j, Info &info_a, Info &info_b);

// compute the non-pure adj_ineq case and return whether it has worked.
// Constraint t is the adj_ineq
bool adjIneqCase(SmallVector<FlatAffineConstraints, 4> &basicSetVector,
                 unsigned i, unsigned j, Info &info_a, Info &info_b);

// compute the adj_eqCase and return whether it has worked
bool adjEqCase(SmallVectorImpl<FlatAffineConstraints> &basicSetVector,
               unsigned i, unsigned j, Info &info_a, Info &info_b, bool pure);

// compute the adj_eq Case for no CUT constraints
bool adjEqCaseNoCut(SmallVectorImpl<FlatAffineConstraints> &basicSetVector,
                    unsigned i, unsigned j, SmallVector<int64_t, 8> t);

void shift(SmallVectorImpl<int64_t> &t, int amount) {
  t.push_back(t.pop_back_val() + 1);
}

void complement(ArrayRef<int64_t> t, SmallVectorImpl<int64_t> &complement) {
  for (size_t k = 0; k < t.size() - 1; k++) {
    complement.push_back(-t[k]);
  }
  complement.push_back(t.back() + 1);
}

static SmallVector<int64_t, 8> arrayRefToSmallVector(ArrayRef<int64_t> ref) {
  SmallVector<int64_t, 8> res;
  for (const int64_t &curr : ref)
    res.push_back(curr);
  return res;
}

void getBasicSetEqualities(FlatAffineConstraints &bs,
                           SmallVector<SmallVector<int64_t, 8>, 8> &eqs) {
  for (unsigned k = 0; k < bs.getNumEqualities(); k++) {
    eqs.push_back(arrayRefToSmallVector(bs.getEquality(k)));
  }
}

void getBasicSetInequalities(FlatAffineConstraints &bs,
                             SmallVector<SmallVector<int64_t, 8>, 8> &ineqs) {
  for (unsigned k = 0; k < bs.getNumInequalities(); k++) {
    ineqs.push_back(arrayRefToSmallVector(bs.getInequality(k)));
  }
}

PresburgerSet mlir::coalesce(PresburgerSet &set) {
  PresburgerSet new_set(set.getNumDims(), set.getNumSyms());
  SmallVector<FlatAffineConstraints, 4> basicSetVector =
      set.getFlatAffineConstraints();
  for (size_t i = 0; i < basicSetVector.size(); i++) {
    for (size_t j = i + 1; j < basicSetVector.size(); j++) {
      FlatAffineConstraints bs1 = basicSetVector[i];
      Simplex simplex_1(bs1);
      SmallVector<SmallVector<int64_t, 8>, 8> equalities1, inequalities1;
      getBasicSetEqualities(bs1, equalities1);
      getBasicSetInequalities(bs1, inequalities1);

      FlatAffineConstraints bs2 = basicSetVector[j];
      Simplex simplex_2(bs2);
      SmallVector<SmallVector<int64_t, 8>, 8> equalities2, inequalities2;
      getBasicSetEqualities(bs2, equalities2);
      getBasicSetInequalities(bs2, inequalities2);
      Info info_1, info_2;
      if (!classify(simplex_2, inequalities1, equalities1, info_1))
        continue;
      if (!classify(simplex_1, inequalities2, equalities2, info_2))
        continue;
      if (!info_1.redundant.empty() && info_1.cut.empty() && !info_1.adj_ineq &&
          !info_2.t) { // contained 2 in 1
        basicSetVector.erase(basicSetVector.begin() + j);
        if (j < i) {
          i--;
        }
        j--;
      } else if (!info_2.redundant.empty() && info_2.cut.empty() &&
                 !info_2.adj_ineq && !info_1.t) { // contained 1 in 2
        basicSetVector.erase(basicSetVector.begin() + i);
        i--;
        break;
      } else if (!info_1.redundant.empty() && !info_1.cut.empty() &&
                 !info_1.adj_ineq && !info_2.t) { // cut or protrusion
        if (cutCase(basicSetVector, i, j, info_1, info_2)) {
          i--;
          break;
        } else if (stickingOut(info_1.cut, bs2) &&
                   protrusionCase(basicSetVector, info_1, info_2, i,
                                  j)) { // protrusion
          i--;
          break;
        }
      } else if (!info_2.redundant.empty() && !info_2.cut.empty() &&
                 !info_2.adj_ineq && !info_1.t) { // cut or protrusion
        if (cutCase(basicSetVector, j, i, info_2, info_1)) {
          i--;
          break;
        } else if (stickingOut(info_2.cut, bs1) &&
                   protrusionCase(basicSetVector, info_2, info_1, j,
                                  i)) { // protrusion
          i--;
          break;
        }
      } else if (!info_1.redundant.empty() && info_1.adj_ineq &&
                 info_1.cut.empty() && !info_2.t && !info_2.redundant.empty() &&
                 info_2.adj_ineq && info_2.cut.empty() &&
                 !info_1.t) { // adj_ineq, pure case
        if (adjIneqPureCase(basicSetVector, i, j, info_1, info_2)) {
          i--;
          break;
        }
      } else if (!info_1.redundant.empty() && info_1.adj_ineq &&
                 info_1.cut.empty() && !info_1.t &&
                 !info_2.t) { // adj_ineq complex case 1
        if (adjIneqCase(basicSetVector, i, j, info_1, info_2)) {
          i--;
          break;
        }
      } else if (!info_2.redundant.empty() && info_2.adj_ineq &&
                 info_2.cut.empty() && !info_2.t &&
                 !info_1.t) { // adj_ineq complex case 2
        if (adjIneqCase(basicSetVector, j, i, info_2, info_1)) {
          i--;
          break;
        }
      } else if (info_1.t && info_2.t) { // adj_eq for two equalities
        if (adjEqCase(basicSetVector, i, j, info_1, info_2, true)) {
          i--;
          break;
        }
      } else if (info_1.t &&
                 info_2.cut.empty()) { // adj_eq Case for one equality
        SmallVector<int64_t, 8> adjEq;
        complement(info_1.t.getValue(), adjEq);
        if (adjEqCaseNoCut(basicSetVector, i, j, adjEq)) { // adj_eq noCut cas
          i--;
          break;
        } else if (info_1.t && adjEqCase(basicSetVector, i, j, info_1, info_2,
                                         false)) { // adjEq case
          i--;
          break;
        }
      } else if (info_2.t &&
                 info_1.cut.empty()) { // adj_eq Case for one equality
        SmallVector<int64_t, 8> adjEq;
        complement(info_2.t.getValue(), adjEq);
        if (adjEqCaseNoCut(basicSetVector, j, i,
                           adjEq)) { // adj_eq noCut cas
          i--;
          break;
        } else if (info_2.t && adjEqCase(basicSetVector, j, i, info_2, info_1,
                                         false)) { // adjEq case
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

void addInequalities(
    FlatAffineConstraints &bs,
    const SmallVector<SmallVector<int64_t, 8>, 8> &inequalities) {
  for (size_t k = 0; k < inequalities.size(); k++) {
    bs.addInequality(inequalities[k]);
  }
}

void addEqualities(FlatAffineConstraints &bs,
                   const SmallVector<SmallVector<int64_t, 8>, 8> &equalities) {
  for (size_t k = 0; k < equalities.size(); k++) {
    bs.addEquality(equalities[k]);
  }
}

bool adjIneqPureCase(SmallVector<FlatAffineConstraints, 4> &basicSetVector,
                     unsigned i, unsigned j, Info &info_a, Info &info_b) {
  FlatAffineConstraints newSet(basicSetVector[i].getNumIds(),
                               basicSetVector[i].getNumSymbolIds());
  addInequalities(newSet, info_a.redundant);
  addInequalities(newSet, info_a.cut);
  addInequalities(newSet, info_b.redundant);
  addInequalities(newSet, info_b.cut);
  addCoalescedBasicSet(basicSetVector, i, j, newSet);
  return true;
}

void addCoalescedBasicSet(
    SmallVectorImpl<FlatAffineConstraints> &basicSetVector, unsigned i,
    unsigned j, FlatAffineConstraints &bs) {
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
                    Info &info_a, Info &info_b, unsigned i, unsigned j) {
  FlatAffineConstraints a = basicSetVector[i];
  FlatAffineConstraints b = basicSetVector[j];
  SmallVector<SmallVector<int64_t, 8>, 8> inequalities_b, equalities_b;
  getBasicSetEqualities(b, equalities_b);
  getBasicSetInequalities(b, inequalities_b);
  addAsIneq(equalities_b, inequalities_b);
  SmallVector<SmallVector<int64_t, 8>, 8> wrapped;
  for (size_t l = 0; l < info_a.cut.size(); l++) {
    SmallVector<int64_t, 8> t = info_a.cut[l];
    FlatAffineConstraints bPrime = FlatAffineConstraints(b.getNumDimIds());
    addInequalities(bPrime, inequalities_b);
    shift(t, 1);
    Simplex simp(bPrime);
    bPrime.addEquality(t);
    if (simp.isEmpty()) {
      shift(t, -1);
      info_a.redundant.push_back(t);
      info_a.cut.erase(info_a.cut.begin() + l);
    } else {
      for (size_t k = 0; k < info_b.cut.size(); k++) {
        SmallVector<int64_t, 8> curr1 = info_b.cut[k];
        Simplex simp2(bPrime);
        simp2.addEquality(curr1);
        if (!simp2.isEmpty()) { // This can be not sufficient!
          auto result = wrapping(a, t, curr1);
          if (!result) {
            return false;
          }
          wrapped.push_back(result.getValue());
        }
      }
    }
  }
  FlatAffineConstraints new_set(b.getNumDimIds());
  addInequalities(new_set, info_a.redundant);
  addInequalities(new_set, wrapped);
  for (size_t k = 0; k < info_a.cut.size(); k++) {
    SmallVector<int64_t, 8> curr = info_a.cut[k];
    int64_t cons = curr.pop_back_val();
    curr.push_back(cons + 1);
    new_set.addInequality(curr);
  }
  addCoalescedBasicSet(basicSetVector, i, j, new_set);
  return true;
}

bool stickingOut(SmallVector<SmallVector<int64_t, 8>, 8> cut,
                 FlatAffineConstraints &bs) {
  Simplex simp(bs);
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
  for (size_t i = 0; i < c1.size(); i++) {
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
  FlatAffineConstraints A = basicSetVector[j];
  FlatAffineConstraints B = basicSetVector[i];
  SmallVector<SmallVector<int64_t, 8>, 8> new_set_inequalities;
  for (size_t k = 0; k < A.getNumInequalities(); k++) {
    SmallVector<int64_t, 8> curr = arrayRefToSmallVector(A.getInequality(k));
    if (!sameConstraint(t, curr)) {
      new_set_inequalities.push_back(arrayRefToSmallVector(A.getInequality(k)));
    }
  }
  shift(t, 1);
  new_set_inequalities.push_back(t);

  FlatAffineConstraints new_set(A.getNumDimIds());
  for (size_t k = 0; k < A.getNumEqualities(); k++) {
    new_set.addEquality(A.getEquality(k));
  }
  addInequalities(new_set, new_set_inequalities);
  Simplex simp(new_set);
  simp.addEquality(t);
  SmallVector<SmallVector<int64_t, 8>, 8> inequalities_b;
  SmallVector<SmallVector<int64_t, 8>, 8> equalities_b;
  for (size_t k = 0; k < B.getNumEqualities(); k++) {
    equalities_b.push_back(arrayRefToSmallVector(B.getEquality(k)));
  }
  for (size_t k = 0; k < B.getNumInequalities(); k++) {
    inequalities_b.push_back(arrayRefToSmallVector(B.getInequality(k)));
  }
  addAsIneq(equalities_b, inequalities_b);
  for (size_t k = 0; k < inequalities_b.size(); k++) {
    if (simp.ineqType(inequalities_b[k]) != Simplex::IneqType::REDUNDANT) {
      return false;
    }
  }
  addCoalescedBasicSet(basicSetVector, i, j, new_set);
  return true;
}

bool adjEqCase(SmallVectorImpl<FlatAffineConstraints> &basicSetVector,
               unsigned i, unsigned j, Info &info_a, Info &info_b, bool pure) {
  FlatAffineConstraints a = basicSetVector[i];
  FlatAffineConstraints b = basicSetVector[j];
  SmallVector<SmallVector<int64_t, 8>, 8> wrapped;
  SmallVector<int64_t, 8> minusT;
  SmallVector<int64_t, 8> t = info_a.t.getValue();
  for (size_t k = 0; k < t.size(); k++) {
    minusT.push_back(-t[k]);
  }
  for (size_t k = 0; k < info_a.cut.size();
       k++) { // TODO: can only cut be non_redundant?
    if (!sameConstraint(t, info_a.cut[k])) {
      auto curr = wrapping(b, minusT, info_a.cut[k]);
      if (curr) {
        wrapped.push_back(curr.getValue());
      } else {
        return false;
      }
    }
  }
  if (!pure) {
    Simplex simp(b);
    for (size_t k = 0; k < wrapped.size(); k++) {
      if (simp.ineqType(wrapped[k]) != Simplex::IneqType::REDUNDANT) {
        return false;
      }
    }
  }
  shift(t, 1);
  shift(minusT, -1);
  for (size_t k = 0; k < info_b.cut.size();
       k++) { // TODO: can only cut be non_redundant?
    if (!sameConstraint(minusT, info_b.cut[k])) {
      auto curr = wrapping(a, t, info_b.cut[k]);
      if (curr) {
        wrapped.push_back(curr.getValue());
      } else {
        return false;
      }
    }
  }
  FlatAffineConstraints new_set(b.getNumIds(), b.getNumSymbolIds());
  if (pure) {
    new_set.addInequality(t);
  } else {
    shift(t, -2);
    SmallVector<int64_t, 8> tComplement;
    complement(t, tComplement);
    new_set.addInequality(tComplement);
  }
  addInequalities(new_set, info_a.redundant);
  addInequalities(new_set, info_b.redundant);
  addInequalities(new_set, wrapped);
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
mlir::wrapping(FlatAffineConstraints &bs, SmallVectorImpl<int64_t> &valid,
               SmallVectorImpl<int64_t> &invalid) {
  size_t n = bs.getNumDimIds();
  Simplex simplex(n + 1);
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

  SmallVector<int64_t, 8> lambda;
  for (size_t k = 0; k < n; k++) {
    lambda.push_back(0);
  }
  lambda.push_back(1);
  lambda.push_back(0); // Is this needed?
  simplex.addInequality(lambda);

  valid.push_back(-1);
  simplex.addEquality(valid);

  invalid.push_back(0);
  Optional<Fraction> result =
      simplex.computeOptimum(Simplex::Direction::Down, invalid);
  if (!result) {
    return {};
  }
  valid.pop_back();
  invalid.pop_back();
  return combineConstraint(valid, invalid, result.getValue());
}

bool classify(Simplex &simp,
              const SmallVector<SmallVector<int64_t, 8>, 8> &inequalities,
              const SmallVector<SmallVector<int64_t, 8>, 8> &equalities,
              Info &info) {
  Optional<SmallVector<int64_t, 8>> dummy = {};
  if (!classify_ineq(simp, inequalities, info))
    return false;
  SmallVector<SmallVector<int64_t, 8>, 8> eqAsIneq;
  addAsIneq(equalities, eqAsIneq);
  for (SmallVector<int64_t, 8> current_constraint : eqAsIneq) {
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
        return false; // this town is too small for two adj_ineq
      info.adj_ineq = current_constraint;
      info.t = current_constraint;
      break;
    case Simplex::IneqType::ADJ_EQ:
      info.t = current_constraint;
      break;
    case Simplex::IneqType::SEPARATE:
      return false; // coalescing failed
    }
  }
  return true;
}

bool classify_ineq(Simplex &simp,
                   const SmallVector<SmallVector<int64_t, 8>, 8> &constraints,
                   Info &info) {
  for (SmallVector<int64_t, 8> current_constraint : constraints) {
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
        return false; // this town is too small for two adj_ineq
      info.adj_ineq = current_constraint;
      break;
    case Simplex::IneqType::ADJ_EQ:
      break;
    case Simplex::IneqType::SEPARATE:
      return false; // coalescing failed
    }
  }
  return true;
}

bool adjIneqCase(SmallVector<FlatAffineConstraints, 4> &basicSetVector,
                 unsigned i, unsigned j, Info &info_a, Info &info_b) {
  ArrayRef<int64_t> t = info_a.adj_ineq.getValue();
  FlatAffineConstraints bs(basicSetVector[i].getNumDimIds(),
                           basicSetVector[i].getNumSymbolIds());
  addInequalities(bs, info_a.redundant);
  addInequalities(bs, info_a.cut);
  addInequalities(bs, info_b.redundant);
  Simplex complement(bs);
  SmallVector<int64_t, 8> tComplement;
  for (size_t k = 0; k < t.size() - 1; k++) {
    tComplement.push_back(-t[k]);
  }
  tComplement.push_back(t[t.size() - 1] - 1);
  complement.addInequality(tComplement);
  for (size_t k = 0; k < info_b.cut.size(); k++) {
    if (complement.ineqType(info_b.cut[k]) != Simplex::IneqType::REDUNDANT) {
      return false;
    }
  }
  if (info_b.adj_ineq) {
    if (complement.ineqType(info_b.adj_ineq.getValue()) !=
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
             unsigned j, Info &info_a, Info &info_b) {
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

bool mlir::containedFacet(ArrayRef<int64_t> ineq, FlatAffineConstraints &bs,
                          SmallVector<SmallVector<int64_t, 8>, 8> &cut) {
  Simplex simp(bs);
  simp.addEquality(ineq);
  for (size_t i = 0; i < cut.size(); i++) {
    if (simp.ineqType(cut[i]) != Simplex::IneqType::REDUNDANT) {
      return false;
    }
  }
  return true;
}

void mlir::addAsIneq(ArrayRef<SmallVector<int64_t, 8>> eq,
                     SmallVectorImpl<SmallVector<int64_t, 8>> &target) {
  for (size_t i = 0; i < eq.size(); i++) {
    SmallVector<int64_t, 8> curr = eq[i];
    target.push_back(curr);
    SmallVector<int64_t, 8> complement;
    for (size_t j = 0; j < curr.size(); j++) {
      complement.push_back(-curr[j]);
    }
    target.push_back(complement);
  }
}

void dumpInfo(Info &info) {
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

void mlir::dump(SmallVectorImpl<int64_t> &cons) {
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
