#include "mlir/Analysis/Presburger/Coalesce.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Dialect/Presburger/Set.h"
#include <iostream>

using namespace mlir;
using namespace mlir::presburger;
// struct for classified constraints
struct Info {
  SmallVector<SmallVector<int64_t, 8>, 8> redundant;
  SmallVector<SmallVector<int64_t, 8>, 8> cut;
  Optional<SmallVector<int64_t, 8>> adj_ineq;
  Optional<SmallVector<int64_t, 8>> t;
};

void dumpInfo(Info *info);

// adds all Constraints to bs
void addEqualities(FlatAffineConstraints &bs,
                   SmallVector<SmallVector<int64_t, 8>, 8> equalities);

void addInequalities(FlatAffineConstraints &bs,
                     SmallVector<SmallVector<int64_t, 8>, 8> inequalities);

// classify of all constraints
// returns true if it has not encountered a separate constraints
bool classify_ineq(Simplex &simp,
                   SmallVector<SmallVector<int64_t, 8>, 8> &constraints,
                   Info *info);

// same thing as classify_ineq, but also return if there is an equality
// constraint adjacent to a the other polytope
// returns true if it has not encountered a separate constraints
bool classify(Simplex &simp,
              SmallVector<SmallVector<int64_t, 8>, 8> inequalities,
              SmallVector<SmallVector<int64_t, 8>, 8> equalities, Info *info);

// compute the protrusionCase and return whether it has worked
bool protrusionCase(SmallVectorImpl<FlatAffineConstraints> &basicSetVector,
                    Info *info_a, Info *info_b, FlatAffineConstraints &a,
                    FlatAffineConstraints &b, int i, int j);

// compute, whether a constraint of cut sticks out of bs by more than 2
bool stickingOut(SmallVector<SmallVector<int64_t, 8>, 8> cut,
                 FlatAffineConstraints &bs);

// add a FlatAffineConstraints and removes the sets at i and j
void addNewBasicSet(SmallVectorImpl<FlatAffineConstraints> &basicSetVector,
                    int i, int j, FlatAffineConstraints &bs);

// compute the cut case and return whether it has worked.
bool cutCase(SmallVector<FlatAffineConstraints, 4> &basicSetVector, int i,
             int j, Info *info_a, Info *info_b);

// compute adj_ineq pure Case and return whether it has worked
bool adjIneqPureCase(SmallVector<FlatAffineConstraints, 4> &basicSetVector,
                     int i, int j, Info *info_a, Info *info_b);

// compute the non-pure adj_ineq case and return whether it has worked.
// Constraint t is the adj_ineq
bool adjIneqCase(SmallVector<FlatAffineConstraints, 4> &basicSetVector, int i,
                 int j, Info *info_a, Info *info_b);

// compute the adj_eqCase and return whether it has worked
bool adjEqCase(SmallVectorImpl<FlatAffineConstraints> &basicSetVector, int i,
               int j, Info *info_a, Info *info_b, FlatAffineConstraints a,
               FlatAffineConstraints b, bool pure);

// compute the adj_eq Case for no CUT constraints
bool adjEqCaseNoCut(SmallVectorImpl<FlatAffineConstraints> &basicSetVector,
                    int i, int j, SmallVector<int64_t, 8> t);

static SmallVector<int64_t, 8> arrayRefToSmallVector(ArrayRef<int64_t> ref) {
  SmallVector<int64_t, 8> res;
  for (const int64_t &curr : ref)
    res.push_back(curr);
  return res;
}

void getBasicSetEqualities(FlatAffineConstraints &bs,
                           SmallVector<SmallVector<int64_t, 8>, 8> &eqs) {
  for (size_t k = 0; k < bs.getNumEqualities(); k++) {
    eqs.push_back(arrayRefToSmallVector(bs.getEquality(k)));
  }
}

void getBasicSetInequalities(FlatAffineConstraints &bs,
                             SmallVector<SmallVector<int64_t, 8>, 8> &ineqs) {
  for (size_t k = 0; k < bs.getNumInequalities(); k++) {
    ineqs.push_back(arrayRefToSmallVector(bs.getInequality(k)));
  }
}

PresburgerSet mlir::coalesce(PresburgerSet &set) {
  PresburgerSet new_set(set.getNumDims());
  SmallVector<FlatAffineConstraints, 4> basicSetVector =
      set.getFlatAffineConstraints();
  for (size_t i = 0; i < basicSetVector.size(); i++) {
    for (size_t j = 0; j < basicSetVector.size(); j++) {
      if (i == j)
        continue;
      FlatAffineConstraints bs1 = basicSetVector[i];
      Simplex simplex_1(bs1);
      std::cout << i << std::endl;
      std::cout << j << std::endl;
      SmallVector<SmallVector<int64_t, 8>, 8> equalities1, inequalities1;
      getBasicSetEqualities(bs1, equalities1);
      getBasicSetInequalities(bs1, inequalities1);

      FlatAffineConstraints bs2 = basicSetVector[j];
      Simplex simplex_2(bs2);
      SmallVector<SmallVector<int64_t, 8>, 8> equalities2, inequalities2;
      getBasicSetEqualities(bs2, equalities2);
      getBasicSetInequalities(bs2, inequalities2);
      Info *info_1 = new Info();
      Info *info_2 = new Info();
      if (!classify(simplex_2, inequalities1, equalities1, info_1))
        continue;
      if (!classify(simplex_1, inequalities2, equalities2, info_2))
        continue;
      dumpInfo(info_1);
      dumpInfo(info_2);
      if (!info_1->redundant.empty() && info_1->cut.empty() &&
          !info_1->adj_ineq && !info_2->t) { // contained 2 in 1
        std::cout << "1" << std::endl;
        basicSetVector.erase(basicSetVector.begin() + j);
        if (j < i) {
          i--;
        }
        j--;
      } else if (!info_2->redundant.empty() && info_2->cut.empty() &&
                 !info_2->adj_ineq && !info_1->t) { // contained 1 in 2
        std::cout << "2" << std::endl;
        basicSetVector.erase(basicSetVector.begin() + i);
        i--;
        break;
      } else if (!info_1->redundant.empty() && !info_1->cut.empty() &&
                 !info_1->adj_ineq && !info_2->t) { // cut or protrusion
        std::cout << "3" << std::endl;
        if (cutCase(basicSetVector, i, j, info_1, info_2)) {
          i--;
          break;
        } else if (stickingOut(info_1->cut, bs2) &&
                   protrusionCase(basicSetVector, info_1, info_2, bs1, bs2, i,
                                  j)) { // protrusion
          i--;
          break;
        }
      } else if (!info_2->redundant.empty() && !info_2->cut.empty() &&
                 !info_2->adj_ineq && !info_1->t) { // cut or protrusion
        std::cout << "4" << std::endl;
        if (cutCase(basicSetVector, j, i, info_2, info_1)) {
          i--;
          break;
        } else if (stickingOut(info_2->cut, bs1) &&
                   protrusionCase(basicSetVector, info_2, info_1, bs2, bs1, j,
                                  i)) { // protrusion
          i--;
          break;
        }
      } else if (!info_1->redundant.empty() && info_1->adj_ineq &&
                 info_1->cut.empty() && !info_2->t &&
                 !info_2->redundant.empty() && info_2->adj_ineq &&
                 info_2->cut.empty() && !info_1->t) { // adj_ineq, pure case
        std::cout << "5" << std::endl;
        if (adjIneqPureCase(basicSetVector, i, j, info_1, info_2)) {
          i--;
          break;
        }
      } else if (!info_1->redundant.empty() && info_1->adj_ineq &&
                 info_1->cut.empty() && !info_1->t &&
                 !info_2->t) { // adj_ineq complex case 1
        std::cout << "6" << std::endl;
        if (adjIneqCase(basicSetVector, i, j, info_1, info_2)) {
          i--;
          break;
        }
      } else if (!info_2->redundant.empty() && info_2->adj_ineq &&
                 info_2->cut.empty() && !info_2->t &&
                 !info_1->t) { // adj_ineq complex case 2
        std::cout << "7" << std::endl;
        if (adjIneqCase(basicSetVector, j, i, info_2, info_1)) {
          i--;
          break;
        }
      } else if (info_1->t && info_2->t) { // adj_eq for two equalities
        std::cout << "8" << std::endl;
        if (adjEqCase(basicSetVector, i, j, info_1, info_2, bs1, bs2, true)) {
          i--;
          break;
        }
      } else if (info_1->t &&
                 info_2->cut.empty()) { // adj_eq Case for one equality
        std::cout << "9" << std::endl;
        SmallVector<int64_t, 8> adjEq, t;
        t = info_1->t.getValue();
        for (size_t k = 0; k < t.size() - 1; k++) {
          adjEq.push_back(-t[k]);
        }
        adjEq.push_back(t[t.size() - 1] + 1);
        if (adjEqCaseNoCut(basicSetVector, i, j, adjEq)) { // adj_eq noCut cas
          i--;
          break;
        } else if (info_1->t && adjEqCase(basicSetVector, i, j, info_1, info_2,
                                          bs1, bs2, false)) { // adjEq case
          i--;
          break;
        }
      } else if (info_2->t &&
                 info_1->cut.empty()) { // adj_eq Case for one equality
        std::cout << "10" << std::endl;
        SmallVector<int64_t, 8> adjEq, t;
        t = info_2->t.getValue();
        for (size_t k = 0; k < t.size() - 1; k++) {
          adjEq.push_back(-t[k]);
        }
        adjEq.push_back(t[t.size() - 1] + 1);
        if (adjEqCaseNoCut(basicSetVector, j, i,
                           adjEq)) { // adj_eq noCut cas
          i--;
          break;
        } else if (info_2->t && adjEqCase(basicSetVector, j, i, info_2, info_1,
                                          bs2, bs1, false)) { // adjEq case
          i--;
          break;
        }
      }
      delete info_1;
      delete info_2;
    }
  }
  for (size_t i = 0; i < basicSetVector.size(); i++) {
    new_set.addFlatAffineConstraints(basicSetVector[i]);
  }
  return new_set;
}

void addInequalities(FlatAffineConstraints &bs,
                     SmallVector<SmallVector<int64_t, 8>, 8> inequalities) {
  for (size_t k = 0; k < inequalities.size(); k++) {
    bs.addInequality(inequalities[k]);
  }
}

void addEqualities(FlatAffineConstraints &bs,
                   SmallVector<SmallVector<int64_t, 8>, 8> equalities) {
  for (size_t k = 0; k < equalities.size(); k++) {
    bs.addEquality(equalities[k]);
  }
}

bool adjIneqPureCase(SmallVector<FlatAffineConstraints, 4> &basicSetVector,
                     int i, int j, Info *info_a, Info *info_b) {
  size_t n = basicSetVector[0].getNumDimIds();
  FlatAffineConstraints newSet(n);
  addInequalities(newSet, info_a->redundant);
  addInequalities(newSet, info_a->cut);
  addInequalities(newSet, info_b->redundant);
  addInequalities(newSet, info_b->cut);
  addNewBasicSet(basicSetVector, i, j, newSet);
  return true;
}

void addNewBasicSet(SmallVectorImpl<FlatAffineConstraints> &basicSetVector,
                    int i, int j, FlatAffineConstraints &bs) {
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
                    Info *info_a, Info *info_b, FlatAffineConstraints &a,
                    FlatAffineConstraints &b, int i, int j) {
  SmallVector<SmallVector<int64_t, 8>, 8> inequalities_b, equalities_b;
  getBasicSetEqualities(b, equalities_b);
  getBasicSetInequalities(b, inequalities_b);
  addAsIneq(equalities_b, inequalities_b);
  SmallVector<SmallVector<int64_t, 8>, 8> wrapped;
  for (size_t i = 0; i < info_a->cut.size(); i++) {
    SmallVector<int64_t, 8> curr = info_a->cut[i];
    FlatAffineConstraints bPrime = FlatAffineConstraints(b.getNumDimIds());
    addInequalities(bPrime, inequalities_b);
    SmallVector<int64_t, 8> new_cons;
    for (size_t k = 0; k < curr.size() - 1; k++) {
      new_cons.push_back(curr[k]);
    }
    new_cons.push_back(curr[curr.size() - 1] + 1);
    Simplex simp(bPrime);
    bPrime.addEquality(new_cons);
    if (simp.isEmpty()) {
      info_a->redundant.push_back(curr);
      info_a->cut.erase(info_a->cut.begin() + i);
    } else {
      int64_t cons = curr.pop_back_val();
      curr.push_back(cons + 1);
      for (size_t k = 0; k < info_b->cut.size(); k++) {
        SmallVector<int64_t, 8> curr1 = info_b->cut[k];
        Simplex simp2(bPrime);
        simp2.addEquality(curr1);
        if (!simp2.isEmpty()) { // This can be not sufficient!
          auto result = wrapping(a, curr, curr1);
          if (!result) {
            return false;
          }
          wrapped.push_back(result.getValue());
        }
      }
    }
  }
  FlatAffineConstraints new_set(b.getNumDimIds());
  addInequalities(new_set, info_a->redundant);
  addInequalities(new_set, wrapped);
  for (size_t k = 0; k < info_a->cut.size(); k++) {
    SmallVector<int64_t, 8> curr = info_a->cut[k];
    int64_t cons = curr.pop_back_val();
    curr.push_back(cons + 1);
    new_set.addInequality(curr);
  }
  addNewBasicSet(basicSetVector, i, j, new_set);
  return true;
}

bool stickingOut(SmallVector<SmallVector<int64_t, 8>, 8> cut,
                 FlatAffineConstraints &bs) {
  Simplex simp(bs);
  for (size_t k = 0; k < cut.size(); k++) {
    SmallVector<int64_t, 8> curr = cut[k];
    auto result = simp.computeOptimum(Simplex::Direction::DOWN, curr);
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

bool mlir::sameConstraint(SmallVectorImpl<int64_t> &c1,
                          SmallVectorImpl<int64_t> &c2) {
  int64_t ratio = 0;
  bool worked = (c1.size() == c1.size());
  for (size_t i = 0; i < c1.size(); i++) {
    if (c2[i] != 0 && ratio == 0) {
      ratio = c1[i] / c2[i];
      worked = worked && (c1[i] == ratio * c2[i]);
    } else if (c2[i] != 0 && ratio != 0) {
      worked = worked && (c1[i] == ratio * c2[i]);
    } else {
      worked = worked && (c1[i] == c2[i]);
    }
  }
  return worked;
}

bool adjEqCaseNoCut(SmallVectorImpl<FlatAffineConstraints> &basicSetVector,
                    int i, int j, SmallVector<int64_t, 8> t) {
  FlatAffineConstraints A = basicSetVector[j];
  FlatAffineConstraints B = basicSetVector[i];
  std::cout << "hello" << std::endl;
  SmallVector<SmallVector<int64_t, 8>, 8> new_set_inequalities;
  for (size_t k = 0; k < A.getNumInequalities(); k++) {
    SmallVector<int64_t, 8> curr = arrayRefToSmallVector(A.getInequality(k));
    if (!sameConstraint(t, curr)) {
      new_set_inequalities.push_back(arrayRefToSmallVector(A.getInequality(k)));
    }
  }
  int64_t cons = t.pop_back_val();
  t.push_back(cons + 1);
  new_set_inequalities.push_back(t);

  FlatAffineConstraints new_set(A.getNumDimIds());
  for (size_t k = 0; k < A.getNumEqualities(); k++) {
    new_set.addEquality(arrayRefToSmallVector(A.getEquality(k)));
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
  addNewBasicSet(basicSetVector, i, j, new_set);
  return true;
}

bool adjEqCase(SmallVectorImpl<FlatAffineConstraints> &basicSetVector, int i,
               int j, Info *info_a, Info *info_b, FlatAffineConstraints a,
               FlatAffineConstraints b, bool pure) {
  SmallVector<SmallVector<int64_t, 8>, 8> wrapped;
  SmallVector<int64_t, 8> minusT;
  SmallVector<int64_t, 8> t = info_a->t.getValue();
  dumpInfo(info_a);
  dumpInfo(info_b);
  for (size_t k = 0; k < t.size(); k++) {
    minusT.push_back(-t[k]);
  }
  for (size_t k = 0; k < info_a->cut.size();
       k++) { // TODO: can only cut be non_redundant?
    if (!sameConstraint(t, info_a->cut[k])) {
      auto curr = wrapping(b, minusT, info_a->cut[k]);
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
  int64_t cons = t.pop_back_val();
  t.push_back(cons + 1);
  cons = minusT.pop_back_val();
  minusT.push_back(cons - 1);
  for (size_t k = 0; k < info_b->cut.size();
       k++) { // TODO: can only cut be non_redundant?
    if (!sameConstraint(minusT, info_b->cut[k])) {
      auto curr = wrapping(a, t, info_b->cut[k]);
      if (curr) {
        wrapped.push_back(curr.getValue());
      } else {
        return false;
      }
    }
  }
  size_t n = b.getNumDimIds();
  FlatAffineConstraints new_set(n);
  if (pure) {
    new_set.addInequality(t);
  } else {
    cons = t.pop_back_val();
    t.push_back(cons - 2);
    SmallVector<int64_t, 8> tComplement;
    for (size_t k = 0; k < t.size() - 1; k++) {
      tComplement.push_back(-t[k]);
    }
    tComplement.push_back(-t[t.size() - 1] - 1);
    new_set.addInequality(tComplement);
  }
  addInequalities(new_set, info_a->redundant);
  addInequalities(new_set, info_b->redundant);
  addInequalities(new_set, wrapped);
  addNewBasicSet(basicSetVector, i, j, new_set);
  return true;
}

SmallVector<int64_t, 8> mlir::combineConstraint(SmallVectorImpl<int64_t> &c1,
                                                SmallVectorImpl<int64_t> &c2,
                                                Fraction<int64_t> ratio) {
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
  Optional<Fraction<int64_t>> result =
      simplex.computeOptimum(Simplex::Direction::DOWN, invalid);
  if (!result) {
    return {};
  }
  valid.pop_back();
  invalid.pop_back();
  return combineConstraint(valid, invalid, result.getValue());
}

bool classify(Simplex &simp,
              SmallVector<SmallVector<int64_t, 8>, 8> inequalities,
              SmallVector<SmallVector<int64_t, 8>, 8> equalities, Info *info) {
  Optional<SmallVector<int64_t, 8>> dummy = {};
  if (!classify_ineq(simp, inequalities, info))
    return false;
  SmallVector<SmallVector<int64_t, 8>, 8> eqAsIneq;
  addAsIneq(equalities, eqAsIneq);
  for (SmallVector<int64_t, 8> current_constraint : eqAsIneq) {
    Simplex::IneqType ty = simp.ineqType(current_constraint);
    switch (ty) {
    case Simplex::IneqType::REDUNDANT:
      info->redundant.push_back(current_constraint);
      break;
    case Simplex::IneqType::CUT:
      info->cut.push_back(current_constraint);
      break;
    case Simplex::IneqType::ADJ_INEQ:
      if (info->adj_ineq)
        return false; // this town is too small for tow adj_ineq
      info->adj_ineq = current_constraint;
      info->t = current_constraint;
      break;
    case Simplex::IneqType::ADJ_EQ:
      info->t = current_constraint;
      break;
    case Simplex::IneqType::SEPARATE:
      return false; // coalescing failed
    }
  }
  return true;
}

bool classify_ineq(Simplex &simp,
                   SmallVector<SmallVector<int64_t, 8>, 8> &constraints,
                   Info *info) {
  for (SmallVector<int64_t, 8> current_constraint : constraints) {
    Simplex::IneqType ty = simp.ineqType(current_constraint);
    switch (ty) {
    case Simplex::IneqType::REDUNDANT:
      info->redundant.push_back(current_constraint);
      break;
    case Simplex::IneqType::CUT:
      info->cut.push_back(current_constraint);
      break;
    case Simplex::IneqType::ADJ_INEQ:
      if (info->adj_ineq)
        return false; // this town is too small for tow adj_ineq
      info->adj_ineq = current_constraint;
      break;
    case Simplex::IneqType::ADJ_EQ:
      break;
    case Simplex::IneqType::SEPARATE:
      return false; // coalescing failed
    }
  }
  return true;
}

bool adjIneqCase(SmallVector<FlatAffineConstraints, 4> &basicSetVector, int i,
                 int j, Info *info_a, Info *info_b) {
  SmallVector<int64_t, 8> t = info_a->adj_ineq.getValue();
  size_t n = basicSetVector[i].getNumDimIds();
  FlatAffineConstraints bs(n);
  addInequalities(bs, info_a->redundant);
  addInequalities(bs, info_a->cut);
  addInequalities(bs, info_b->redundant);
  Simplex complement(bs);
  SmallVector<int64_t, 8> tComplement;
  for (size_t k = 0; k < t.size() - 1; k++) {
    tComplement.push_back(-t[k]);
  }
  tComplement.push_back(t[t.size() - 1] - 1);
  complement.addInequality(tComplement);
  for (size_t k = 0; k < info_b->cut.size(); k++) {
    if (complement.ineqType(info_b->cut[k]) != Simplex::IneqType::REDUNDANT) {
      return false;
    }
  }
  if (info_b->adj_ineq) {
    if (complement.ineqType(info_b->adj_ineq.getValue()) !=
        Simplex::IneqType::REDUNDANT) {
      return false;
    }
  }
  FlatAffineConstraints newSet(n);
  addInequalities(newSet, info_a->redundant);
  addInequalities(newSet, info_b->redundant);
  addNewBasicSet(basicSetVector, i, j, newSet);
  return true;
}

bool cutCase(SmallVector<FlatAffineConstraints, 4> &basicSetVector, int i,
             int j, Info *info_a, Info *info_b) {
  for (size_t k = 0; k < info_a->cut.size(); k++) {
    if (!containedFacet(info_a->cut[k], basicSetVector[i], info_b->cut)) {
      return false;
    }
  }
  size_t n = basicSetVector[i].getNumDimIds();
  FlatAffineConstraints new_set(n);
  addInequalities(new_set, info_a->redundant);
  addInequalities(new_set, info_b->redundant);
  addNewBasicSet(basicSetVector, i, j, new_set);
  return true;
}

bool mlir::containedFacet(SmallVectorImpl<int64_t> &ineq,
                          FlatAffineConstraints &bs,
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

void mlir::addAsIneq(SmallVector<SmallVector<int64_t, 8>, 8> &eq,
                     SmallVector<SmallVector<int64_t, 8>, 8> &target) {
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

void dumpInfo(Info *info) {
  std::cout << "red:" << std::endl;
  for (size_t k = 0; k < info->redundant.size(); k++) {
    dump(info->redundant[k]);
  }
  std::cout << "cut:" << std::endl;
  for (size_t k = 0; k < info->cut.size(); k++) {
    dump(info->cut[k]);
  }
  if (info->adj_ineq) {
    std::cout << "adj_ineq:" << std::endl;
    dump(info->adj_ineq.getValue());
  }
  if (info->t) {
    std::cout << "t:" << std::endl;
    dump(info->t.getValue());
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
