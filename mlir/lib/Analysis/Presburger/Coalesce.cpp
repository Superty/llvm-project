#include "mlir/Analysis/Presburger/Coalesce.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Dialect/Presburger/Set.h"
#include <iostream>

using namespace mlir;
using namespace mlir::presburger;
// struct for classified constraints
struct Classification {
  SmallVector<SmallVector<int64_t, 8>, 8> redundant;
  SmallVector<SmallVector<int64_t, 8>, 8> cut;
  SmallVector<SmallVector<int64_t, 8>, 8> adj_ineq;
  SmallVector<SmallVector<int64_t, 8>, 8> adj_eq;
  SmallVector<SmallVector<int64_t, 8>, 8> separate;
  SmallVector<SmallVector<int64_t, 8>, 8> non_cut;
  SmallVector<SmallVector<int64_t, 8>, 8> non_adj;
  SmallVector<SmallVector<int64_t, 8>, 8> non_redundant;
};

// struct for all infos needed from classification
struct Info {
  Optional<SmallVector<int64_t, 8>> t;
  Classification *classification;
  ~Info() { delete classification; }
};

//adds all Constraints to bs
void addEqualities(FlatAffineConstraints &bs, SmallVector<SmallVector<int64_t, 8>, 8> equalities);

void addInequalities(FlatAffineConstraints &bs, SmallVector<SmallVector<int64_t, 8>, 8> inequalities);

// classify of all constraints into redundant, cut, adj_ineq, adj_eq, separate,
// non_cut, non_adj, where non_cut and non_adj are everything but cut or
// adj_ineq respectively
void classify_ineq(Simplex &simp, SmallVector<SmallVector<int64_t, 8>, 8> &constraints,
                   Classification *classi);

// same thing as classify_ineq, but also return if there is an equality
// constraint adjacent to a the other polytope
void classify(Simplex &simp, SmallVector<SmallVector<int64_t, 8>, 8> inequalities,
              SmallVector<SmallVector<int64_t, 8>, 8> equalities, Info *info);

// compute the protrusionCase and return whether it has worked
bool protrusionCase(SmallVectorImpl<FlatAffineConstraints> &basicSetVector,
                    SmallVector<SmallVector<int64_t, 8>, 8> cut_a,
                    SmallVector<SmallVector<int64_t, 8>, 8> non_redundant_b,
                    SmallVector<SmallVector<int64_t, 8>, 8> redundant, FlatAffineConstraints &a, FlatAffineConstraints &b,
                    int i, int j);

// compute, whether a constraint of cut sticks out of bs by more than 2
bool stickingOut(SmallVector<SmallVector<int64_t, 8>, 8> cut, FlatAffineConstraints &bs);


// add a FlatAffineConstraints and removes the sets at i and j
void addNewBasicSet(SmallVectorImpl<FlatAffineConstraints> &basicSetVector, int i, int j,
                    FlatAffineConstraints &bs);

// compute the cut case and return whether it has worked.
bool cutCase(SmallVector<FlatAffineConstraints, 4> &basicSetVector, int i, int j,
             SmallVector<SmallVector<int64_t, 8>, 8> &cut_1, SmallVector<SmallVector<int64_t, 8>, 8> &cut_2,
             SmallVector<SmallVector<int64_t, 8>, 8> &non_cut_1,
             SmallVector<SmallVector<int64_t, 8>, 8> &non_cut_2);

// compute adj_ineq pure Case and return whether it has worked
bool adjIneqPureCase(SmallVector<FlatAffineConstraints, 4> &basicSetVector, int i, int j,
                     SmallVector<SmallVector<int64_t, 8>, 8> non_adj_1,
                     SmallVector<SmallVector<int64_t, 8>, 8> non_adj_2);

// compute the non-pure adj_ineq case and return whether it has worked.
// Constraint t is the adj_ineq
bool adjIneqCase(SmallVector<FlatAffineConstraints, 4> &basicSetVector, int i, int j,
                 SmallVector<SmallVector<int64_t, 8>, 8> non_adj_1,
                 SmallVector<SmallVector<int64_t, 8>, 8> redundant_1,
                 SmallVector<SmallVector<int64_t, 8>, 8> redundant_2, SmallVector<int64_t, 8> t,
                 SmallVector<SmallVector<int64_t, 8>, 8> ineq, SmallVector<SmallVector<int64_t, 8>, 8> eq);

// compute the adj_eqCase and return whether it has worked
bool adjEqCase(SmallVectorImpl<FlatAffineConstraints> &basicSetVector, int i, int j,
               SmallVector<SmallVector<int64_t, 8>, 8> redundant_1,
               SmallVector<SmallVector<int64_t, 8>, 8> redundant_2, FlatAffineConstraints a, FlatAffineConstraints b,
               SmallVector<int64_t, 8> t, SmallVector<SmallVector<int64_t, 8>, 8> cut,
               SmallVector<SmallVector<int64_t, 8>, 8> non_redundant_1,
               SmallVector<SmallVector<int64_t, 8>, 8> non_redundant_2, bool pure);

// compute the adj_eq Case for no CUT constraints
bool adjEqCaseNoCut(SmallVectorImpl<FlatAffineConstraints> &basicSetVector, int i, int j,
                    SmallVector<int64_t, 8> t);


static SmallVector<int64_t, 8> arrayRefToSmallVector(ArrayRef<int64_t> ref) {
  SmallVector<int64_t, 8> res;
  for(const int64_t &curr : ref) 
    res.push_back(curr);
  return res;
}

PresburgerSet mlir::coalesce(PresburgerSet &set) {
  PresburgerSet new_set(set.getNumDims());
  SmallVector<FlatAffineConstraints, 4> basicSetVector = set.getFlatAffineConstraints();
  for (size_t i = 0; i < basicSetVector.size(); i++) {
    for (size_t j = 0; j < basicSetVector.size(); j++) {
      if (i == j)
        continue;
      FlatAffineConstraints bs1 = basicSetVector[i];
      Simplex simplex_1(bs1);
      std::cout << i << std::endl;
      std::cout << j << std::endl;
      SmallVector<SmallVector<int64_t, 8>, 8> equalities1;
      SmallVector<SmallVector<int64_t, 8>, 8> inequalities1;
      for (size_t k = 0; k < bs1.getNumEqualities(); k++) {
        equalities1.push_back(arrayRefToSmallVector(bs1.getEquality(k)));
      }
      for (size_t k = 0; k < bs1.getNumInequalities(); k++) {
        inequalities1.push_back(arrayRefToSmallVector(bs1.getInequality(k)));
      }
      FlatAffineConstraints bs2 = basicSetVector[j];
      Simplex simplex_2(bs2);
      SmallVector<SmallVector<int64_t, 8>, 8> equalities2;
      SmallVector<SmallVector<int64_t, 8>, 8> inequalities2; 
      for (size_t k = 0; k < bs2.getNumEqualities(); k++) {
        equalities2.push_back(arrayRefToSmallVector(bs2.getEquality(k)));
      }
      for (size_t k = 0; k < bs2.getNumInequalities(); k++) {
        inequalities2.push_back(arrayRefToSmallVector(bs2.getInequality(k)));
      }
      Info *info_1 = new Info();
      Info *info_2 = new Info();
      classify(simplex_2, inequalities1, equalities1, info_1);
      classify(simplex_1, inequalities2, equalities2, info_2);
      
      std::cout << "red:" << std::endl;
      for(size_t k = 0; k < info_1->classification->redundant.size(); k++) {
        dump(info_1->classification->redundant[k]);
      }
      std::cout << "cut:" << std::endl;
      for(size_t k = 0; k < info_1->classification->cut.size(); k++) {
        dump(info_1->classification->cut[k]);
      }
      std::cout << "adj_ineq:" << std::endl;
      for(size_t k = 0; k < info_1->classification->adj_ineq.size(); k++) {
        dump(info_1->classification->adj_ineq[k]);
      }
      std::cout << "adj_eq:" << std::endl;
      for(size_t k = 0; k < info_1->classification->adj_eq.size(); k++) {
        dump(info_1->classification->adj_eq[k]);
      }
      std::cout << "separate:" << std::endl;
      for(size_t k = 0; k < info_1->classification->separate.size(); k++) {
        dump(info_1->classification->separate[k]);
      }
      if (info_1->t) {
        std::cout << "t:" << std::endl;
        dump(info_1->t.getValue());
      }
      if (!info_1->classification->redundant.empty() &&
          info_1->classification->cut.empty() &&
          info_1->classification->adj_ineq.empty() &&
          info_1->classification->adj_eq.empty() &&
          info_1->classification->separate.empty()) { // contained 2 in 1
        basicSetVector.erase(basicSetVector.begin() + j);
        if (j < i) {
          i--;
        }
        j--;
      } else if (!info_2->classification->redundant.empty() &&
                 info_2->classification->cut.empty() &&
                 info_2->classification->adj_ineq.empty() &&
                 info_2->classification->adj_eq.empty() &&
                 info_2->classification->separate.empty()) { // contained 1 in 2
        basicSetVector.erase(basicSetVector.begin() + i);
        i--;
        break;
      } else if (!info_1->classification->redundant.empty() &&
                 !info_1->classification->cut.empty() &&
                 info_1->classification->adj_ineq.empty() &&
                 info_1->classification->adj_eq.empty() &&
                 info_1->classification->separate
                     .empty()) { // cut or protrusion
        if (cutCase(basicSetVector, i, j, info_1->classification->cut,
                    info_2->classification->cut,
                    info_1->classification->non_cut,
                    info_2->classification->non_cut)) {
          i--;
          break;
        } else if (stickingOut(info_1->classification->cut, bs2) &&
                   protrusionCase(basicSetVector, info_1->classification->cut,
                                  info_2->classification->non_redundant,
                                  info_1->classification->redundant, bs1,
                                  bs2, i, j)) { // protrusion
          i--;
          break;
        }
      } else if (!info_2->classification->redundant.empty() &&
                 !info_2->classification->cut.empty() &&
                 info_2->classification->adj_ineq.empty() &&
                 info_2->classification->adj_eq.empty() &&
                 info_2->classification->separate
                     .empty()) { // cut or protrusion
        if (cutCase(basicSetVector, j, i, info_2->classification->cut,
                    info_1->classification->cut,
                    info_2->classification->non_cut,
                    info_1->classification->non_cut)) {
          i--;
          break;
        } else if (stickingOut(info_2->classification->cut, bs1) &&
                   protrusionCase(basicSetVector, info_2->classification->cut,
                                  info_1->classification->non_redundant,
                                  info_2->classification->redundant, bs2,
                                  bs1, j,
                                  i)) { // protrusion
          i--;
          break;
        }
      } else if (!info_1->classification->redundant.empty() &&
                 info_1->classification->adj_ineq.size() == 1 &&
                 info_1->classification->cut.empty() &&
                 info_1->classification->adj_eq.empty() &&
                 info_1->classification->separate.empty() &&
                 !info_2->classification->redundant.empty() &&
                 info_2->classification->adj_ineq.size() == 1 &&
                 info_2->classification->cut.empty() &&
                 info_2->classification->adj_eq.empty() &&
                 info_2->classification->separate
                     .empty()) { // adj_ineq, pure case
        if (adjIneqPureCase(basicSetVector, i, j,
                            info_1->classification->non_adj,
                            info_2->classification->non_adj)) {
          i--;
          break;
        }
      } else if (!info_1->classification->redundant.empty() &&
                 info_1->classification->adj_ineq.size() == 1 &&
                 info_1->classification->cut.empty() &&
                 info_1->classification->adj_eq.empty() &&
                 info_1->classification->separate
                     .empty() && !info_2->t) { // adj_ineq complex case 1
        SmallVector<int64_t, 8> adj_ineq = info_1->classification->adj_ineq[0];
        if (adjIneqCase(basicSetVector, i, j, info_1->classification->non_adj,
                        info_1->classification->redundant,
                        info_2->classification->redundant, adj_ineq,
                        inequalities2, equalities2)) {
          i--;
          break;
        }
      } else if (!info_2->classification->redundant.empty() &&
                 info_2->classification->adj_ineq.size() == 1 &&
                 info_2->classification->cut.empty() &&
                 info_2->classification->adj_eq.empty() &&
                 info_2->classification->separate
                     .empty() && !info_1->t) { // adj_ineq complex case 2
        SmallVector<int64_t, 8> adj_ineq = info_2->classification->adj_ineq[0];
        if (adjIneqCase(basicSetVector, j, i, info_2->classification->non_adj,
                        info_2->classification->redundant,
                        info_1->classification->redundant, adj_ineq,
                        inequalities1, equalities1)) {
          i--;
          break;
        }
      } else if (info_1->t && info_2->t) { // adj_eq for two equalities
          std::cout << "1" << std::endl;
          if (adjEqCase(basicSetVector, i, j, info_1->classification->redundant,
                      info_2->classification->redundant, bs1, bs2,
                      info_1->t.getValue(), info_1->classification->cut,
                      info_1->classification->non_redundant,
                      info_2->classification->non_redundant, true)) {
          i--;
          break;
        }
      } else if (info_1->t && 
                 info_2->classification->cut.empty() &&
                 info_1->classification->separate
                     .empty()) { // adj_eq Case for one equality
          SmallVector<int64_t, 8> adjEq, t;
          t = info_1->t.getValue();
          for(size_t k = 0; k < t.size()-1; k++) {
            adjEq.push_back(-t[k]);
          }
          adjEq.push_back(t[t.size()-1]+1);
          if (adjEqCaseNoCut(
                basicSetVector, i, j,
                adjEq)) { // adj_eq noCut cas
          i--;
          break;
        } else if (info_1->t &&
                   adjEqCase(basicSetVector, i, j,
                             info_1->classification->redundant,
                             info_2->classification->redundant, bs1,
                             bs2, info_1->t.getValue(),
                             info_1->classification->cut,
                             info_1->classification->non_redundant,
                             info_2->classification->non_redundant,
                             false)) { // adjEq case
          i--;
          break;
        }
      } else if (info_2->t &&
                 info_1->classification->cut.empty() &&
                 info_2->classification->separate
                     .empty()) { // adj_eq Case for one equality
          SmallVector<int64_t, 8> adjEq, t;
          t = info_2->t.getValue();
          for(size_t k = 0; k < t.size()-1; k++) {
            adjEq.push_back(-t[k]);
          }
          adjEq.push_back(t[t.size()-1]+1);
        if (adjEqCaseNoCut(
                basicSetVector, j, i,
                adjEq)) { // adj_eq noCut cas
          i--;
          break;
        } else if (info_2->t &&
                   adjEqCase(basicSetVector, j, i, info_2->classification->redundant,
                      info_1->classification->redundant, bs2, bs1,
                      info_2->t.getValue(), info_2->classification->cut,
                      info_2->classification->non_redundant,
                      info_1->classification->non_redundant,
                      false)) { // adjEq case
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


void addInequalities(FlatAffineConstraints &bs, SmallVector<SmallVector<int64_t, 8>, 8> inequalities) {
  for (size_t k = 0; k < inequalities.size(); k++) {
    bs.addInequality(inequalities[k]);
  }
}

void addEqualities(FlatAffineConstraints &bs, SmallVector<SmallVector<int64_t, 8>, 8> equalities) {
  for (size_t k = 0; k < equalities.size(); k++) {
    bs.addEquality(equalities[k]);
  }
}

bool adjIneqPureCase(SmallVector<FlatAffineConstraints, 4> &basicSetVector, int i, int j,
                     SmallVector<SmallVector<int64_t, 8>, 8> non_adj_1,
                     SmallVector<SmallVector<int64_t, 8>, 8> non_adj_2) {
  size_t n = basicSetVector[0].getNumDimIds();
  FlatAffineConstraints newSet(n);
  addInequalities(newSet, non_adj_1);
  addInequalities(newSet, non_adj_2);
  addNewBasicSet(basicSetVector, i, j, newSet);
  return true;
}

void addNewBasicSet(SmallVectorImpl<FlatAffineConstraints> &basicSetVector, int i, int j,
                    FlatAffineConstraints &bs) {
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
                    SmallVector<SmallVector<int64_t, 8>, 8> cut_a,
                    SmallVector<SmallVector<int64_t, 8>, 8> non_redundant_b,
                    SmallVector<SmallVector<int64_t, 8>, 8> redundant, FlatAffineConstraints &a, FlatAffineConstraints &b,
                    int i, int j) {
  SmallVector<SmallVector<int64_t, 8>, 8> inequalities_b;
  SmallVector<SmallVector<int64_t, 8>, 8> equalities_b;
  for (size_t k = 0; k < b.getNumEqualities(); k++) {
    equalities_b.push_back(arrayRefToSmallVector(b.getEquality(k)));
  }
  for (size_t k = 0; k < b.getNumInequalities(); k++) {
    inequalities_b.push_back(arrayRefToSmallVector(b.getInequality(k)));
  }
  addAsIneq(equalities_b, inequalities_b);
  SmallVector<SmallVector<int64_t, 8>, 8> wrapped;
  for (size_t i = 0; i < cut_a.size(); i++) {
    SmallVector<int64_t, 8> curr = cut_a[i];
    FlatAffineConstraints bPrime = FlatAffineConstraints(b.getNumDimIds());
    addInequalities(bPrime, inequalities_b);
    SmallVector<int64_t, 8> new_cons;
    for(size_t k = 0; k < curr.size()-1 ; k++) {
      new_cons.push_back(curr[k]);
    }
    new_cons.push_back(curr[curr.size()-1] + 1);    
    Simplex simp(bPrime);
    bPrime.addEquality(new_cons);
    if (simp.isEmpty()) {
      redundant.push_back(curr);
      cut_a.erase(cut_a.begin() + i);
    } else {
      int64_t cons = curr.pop_back_val();
      curr.push_back(cons+1);
      for (size_t k = 0; k < non_redundant_b.size(); k++) {
        SmallVector<int64_t, 8> curr1 = non_redundant_b[k];
        Simplex simp2(bPrime);
        simp2.addEquality(curr1);
        if (!simp2.isEmpty()) {//This can be not sufficient!
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
  addInequalities(new_set, redundant);
  addInequalities(new_set, wrapped);
  for (size_t k = 0; k < cut_a.size(); k++) {
    SmallVector<int64_t, 8> curr = cut_a[k];
    int64_t cons = curr.pop_back_val();
    curr.push_back(cons+1);
    new_set.addInequality(curr);
  }
  addNewBasicSet(basicSetVector, i, j, new_set);
  return true;
}

bool stickingOut(SmallVector<SmallVector<int64_t, 8>, 8> cut, FlatAffineConstraints &bs) {
  Simplex simp(bs);
  for (size_t k = 0; k < cut.size(); k++) {
    SmallVector<int64_t, 8> curr = cut[k];
    auto result =
        simp.computeOptimum(Simplex::Direction::DOWN, curr);
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

bool mlir::sameConstraint(SmallVector<int64_t, 8> c1, SmallVector<int64_t, 8> c2) {
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

bool adjEqCaseNoCut(SmallVectorImpl<FlatAffineConstraints> &basicSetVector, int i, int j,
                    SmallVector<int64_t, 8> t) {
  FlatAffineConstraints A = basicSetVector[j];
  FlatAffineConstraints B = basicSetVector[i];
  SmallVector<SmallVector<int64_t, 8>, 8> new_set_inequalities;
  for (size_t k = 0; k < A.getNumInequalities(); k++) {
    if (!sameConstraint(t, arrayRefToSmallVector(A.getInequality(k)))) {
      new_set_inequalities.push_back(arrayRefToSmallVector(A.getInequality(k)));
    }
  }
  int64_t cons = t.pop_back_val();
  t.push_back(cons+1);
  new_set_inequalities.push_back(t);

  FlatAffineConstraints new_set(A.getNumDimIds());
  for(size_t k = 0; k < A.getNumEqualities(); k++) {
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
    if (simp.ineqType(inequalities_b[k]) !=
        Simplex::IneqType::REDUNDANT) {
      return false;
    }
  }
  addNewBasicSet(basicSetVector, i, j, new_set);
  return true;
}

bool adjEqCase(SmallVectorImpl<FlatAffineConstraints> &basicSetVector, int i, int j,
               SmallVector<SmallVector<int64_t, 8>, 8> redundant_1,
               SmallVector<SmallVector<int64_t, 8>, 8> redundant_2, FlatAffineConstraints a, FlatAffineConstraints b,
               SmallVector<int64_t, 8> t, SmallVector<SmallVector<int64_t, 8>, 8> cut,
               SmallVector<SmallVector<int64_t, 8>, 8> non_redundant_1,
               SmallVector<SmallVector<int64_t, 8>, 8> non_redundant_2, bool pure) {
  SmallVector<SmallVector<int64_t, 8>, 8> wrapped;
  SmallVector<int64_t, 8> minusT;
  dump(t);
  for (size_t k = 0; k < t.size(); k++) {
    minusT.push_back(-t[k]);
  }
  dump(minusT);
  for (size_t k = 0; k < non_redundant_1.size(); k++) {
    if (!sameConstraint(t, non_redundant_1[k])) {
      auto curr = wrapping(b, minusT, non_redundant_1[k]);
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
      if (simp.ineqType(wrapped[k]) !=
          Simplex::IneqType::REDUNDANT) {
        return false;
      }
    }
  }
  int64_t cons = t.pop_back_val();
  t.push_back(cons+1);
  cons = minusT.pop_back_val();
  minusT.push_back(cons-1);
  dump(t);
  dump(minusT);
  for (size_t k = 0; k < non_redundant_2.size(); k++) {
    if (!sameConstraint(minusT, non_redundant_2[k])) {
      auto curr = wrapping(a, t, non_redundant_2[k]);
      if (curr) {
        wrapped.push_back(curr.getValue());
      } else {
        return false;
      }
    }
  }
  size_t n = b.getNumDimIds();
  FlatAffineConstraints new_set(n);
  if(pure) {
    new_set.addInequality(t);
  } else {
    cons = t.pop_back_val();
    t.push_back(cons-2);
    SmallVector<int64_t, 8> tComplement;
    for(size_t k = 0; k < t.size()-1; k++) {
      tComplement.push_back(-t[k]);
    }
    tComplement.push_back(-t[t.size()-1]-1);
    dump(tComplement);
    new_set.addInequality(tComplement);
  }
  addInequalities(new_set, redundant_1);
  addInequalities(new_set, redundant_2);
  addInequalities(new_set, wrapped);
  addNewBasicSet(basicSetVector, i, j, new_set);
  return true;
}

SmallVector<int64_t, 8> mlir::combineConstraint(SmallVector<int64_t, 8> c1, SmallVector<int64_t, 8> c2, Fraction<int64_t> ratio) {
  int64_t n = ratio.num;
  int64_t d = ratio.den;
  SmallVector<int64_t, 8> result;
  for (size_t i = 0; i < c1.size(); i++) {
    result.push_back(-n * c1[i] + d * c2[i]);
  }
  return result;
}

Optional<SmallVector<int64_t, 8>> mlir::wrapping(FlatAffineConstraints bs, SmallVector<int64_t, 8> valid,
                                   SmallVector<int64_t, 8> invalid) {
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
  lambda.push_back(0); //Is this needed?
  simplex.addInequality(lambda);

  valid.push_back(-1);
  simplex.addEquality(valid);

  invalid.push_back(0);  
  Optional<Fraction<int64_t>> result = simplex.computeOptimum(
      Simplex::Direction::DOWN, invalid);
  if (!result) {
    return {};
  }
  valid.pop_back();
  invalid.pop_back();
  return combineConstraint(valid, invalid, result.getValue());
}

void classify(Simplex &simp, SmallVector<SmallVector<int64_t, 8>, 8> inequalities,
              SmallVector<SmallVector<int64_t, 8>, 8> equalities, Info *info) {
  Optional<SmallVector<int64_t, 8>> dummy = {};
  Classification *ineqs = new Classification();
  classify_ineq(simp, inequalities, ineqs);
  info->classification = ineqs;
  SmallVector<SmallVector<int64_t, 8>, 8> eqAsIneq;
  addAsIneq(equalities, eqAsIneq);
  for (SmallVector<int64_t, 8> current_constraint : eqAsIneq) {
    Simplex::IneqType ty = simp.ineqType(current_constraint);
    switch (ty) {
    case Simplex::IneqType::REDUNDANT:
      ineqs->redundant.push_back(current_constraint);
      ineqs->non_cut.push_back(current_constraint);
      ineqs->non_adj.push_back(current_constraint);
      break;
    case Simplex::IneqType::CUT:
      ineqs->cut.push_back(current_constraint);
      ineqs->non_adj.push_back(current_constraint);
      ineqs->non_redundant.push_back(current_constraint);
      break;
    case Simplex::IneqType::ADJ_INEQ:
      ineqs->adj_ineq.push_back(current_constraint);
      ineqs->non_cut.push_back(current_constraint);
      ineqs->non_redundant.push_back(current_constraint);
      info->t = current_constraint;
      break;
    case Simplex::IneqType::ADJ_EQ:
      ineqs->adj_eq.push_back(current_constraint);
      ineqs->non_cut.push_back(current_constraint);
      ineqs->non_adj.push_back(current_constraint);
      info->t = current_constraint;
      break;
    case Simplex::IneqType::SEPARATE:
      ineqs->separate.push_back(current_constraint);
      ineqs->non_cut.push_back(current_constraint);
      ineqs->non_adj.push_back(current_constraint);
      ineqs->non_redundant.push_back(current_constraint);
      break;
    }
  }
}

/* This function returns the classifcation of all constraints divided into
 * redundant, cut, adj_ineq, adj_eq, separate, non_cut, non_adj, where non_cut
 * and non_adj are everything but cut or adj_ineq respectively
 */
void classify_ineq(Simplex &simp, SmallVector<SmallVector<int64_t, 8>, 8> &constraints,
                   Classification *classi) {
  for (SmallVector<int64_t, 8> current_constraint : constraints) {
    Simplex::IneqType ty = simp.ineqType(current_constraint);
    switch (ty) {
    case Simplex::IneqType::REDUNDANT:
      classi->redundant.push_back(current_constraint);
      classi->non_cut.push_back(current_constraint);
      classi->non_adj.push_back(current_constraint);
      break;
    case Simplex::IneqType::CUT:
      classi->cut.push_back(current_constraint);
      classi->non_adj.push_back(current_constraint);
      classi->non_redundant.push_back(current_constraint);
      break;
    case Simplex::IneqType::ADJ_INEQ:
      classi->adj_ineq.push_back(current_constraint);
      classi->non_cut.push_back(current_constraint);
      classi->non_redundant.push_back(current_constraint);
      break;
    case Simplex::IneqType::ADJ_EQ:
      classi->adj_eq.push_back(current_constraint);
      classi->non_cut.push_back(current_constraint);
      classi->non_adj.push_back(current_constraint);
      break;
    case Simplex::IneqType::SEPARATE:
      classi->separate.push_back(current_constraint);
      classi->non_cut.push_back(current_constraint);
      classi->non_adj.push_back(current_constraint);
      classi->non_redundant.push_back(current_constraint);
      break;
    }
  }
}

bool adjIneqCase(SmallVector<FlatAffineConstraints, 4> &basicSetVector, int i, int j,
                 SmallVector<SmallVector<int64_t, 8>, 8> non_adj_1,
                 SmallVector<SmallVector<int64_t, 8>, 8> redundant_1,
                 SmallVector<SmallVector<int64_t, 8>, 8> redundant_2, SmallVector<int64_t, 8> t,
                 SmallVector<SmallVector<int64_t, 8>, 8> ineq, SmallVector<SmallVector<int64_t, 8>, 8> eq) {
  size_t n = basicSetVector[i].getNumDimIds();
  FlatAffineConstraints bs(n);
  addInequalities(bs, non_adj_1);
  addInequalities(bs, redundant_2);
  Simplex complement(bs);
  Simplex original(bs);
  SmallVector<int64_t, 8> tComplement;
  for(size_t k = 0; k < t.size()-1; k++) {
    tComplement.push_back(-t[k]);
  }  
  tComplement.push_back(t[t.size()-1]-1);
  complement.addInequality(tComplement);
  for (size_t k = 0; k < ineq.size(); k++) {
    if (complement.ineqType(ineq[k]) !=
            Simplex::IneqType::REDUNDANT &&
        original.ineqType(ineq[k]) !=
            Simplex::IneqType::REDUNDANT) {
      return false;
    }
  }
  for (size_t k = 0; k < eq.size(); k++) {
    if (complement.ineqType(eq[k]) !=
            Simplex::IneqType::REDUNDANT &&
        original.ineqType(eq[k]) !=
            Simplex::IneqType::REDUNDANT) {
      return false;
    }
  }
  FlatAffineConstraints newSet(n);
  addInequalities(newSet, redundant_1);
  addInequalities(newSet, redundant_2);
  addNewBasicSet(basicSetVector, i, j, newSet);
  return true;
}

bool cutCase(SmallVector<FlatAffineConstraints, 4> &basicSetVector, int i, int j,
             SmallVector<SmallVector<int64_t, 8>, 8> &cut_1, SmallVector<SmallVector<int64_t, 8>, 8> &cut_2,
             SmallVector<SmallVector<int64_t, 8>, 8> &non_cut_1,
             SmallVector<SmallVector<int64_t, 8>, 8> &non_cut_2) {
  for (size_t k = 0; k < cut_1.size(); k++) {
    if (!containedFacet(cut_1[k], basicSetVector[i], cut_2)) {
      return false;
    }
  }
  size_t n = basicSetVector[i].getNumDimIds();
  FlatAffineConstraints new_set(n);
  addInequalities(new_set, non_cut_1);
  addInequalities(new_set, non_cut_2);
  addNewBasicSet(basicSetVector, i, j, new_set);
  return true;
}

bool mlir::containedFacet(SmallVector<int64_t, 8> &ineq, FlatAffineConstraints &bs,
                    SmallVector<SmallVector<int64_t, 8>, 8> &cut) {
  Simplex simp(bs);
  simp.addEquality(ineq);
  for (size_t i = 0; i < cut.size(); i++) {
    if (simp.ineqType(cut[i]) !=
        Simplex::IneqType::REDUNDANT) {
      return false;
    }
  }
  return true;
}


void mlir::addAsIneq(SmallVector<SmallVector<int64_t, 8>, 8> &eq, SmallVector<SmallVector<int64_t, 8>, 8> &target) {
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

void mlir::dump(SmallVector<int64_t, 8> &cons) {
  std::cout << cons[cons.size()-1] << " + ";
  for(size_t i = 1; i < cons.size(); i++) {
    std::cout << cons[i-1] << "x" << i-1;
    if(i == cons.size() -1) {
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

