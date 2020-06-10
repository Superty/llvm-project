#include "mlir/Analysis/Presburger/Coalesce.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Dialect/Presburger/Set.h"

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

/*// compute the protrusionCase and return whether it has worked
bool protrusionCase(std::vector<BasicSet> &basic_set_vector,
                    std::vector<Constraint> cut_a,
                    SmallVector<SmallVector<int64_t, 8>, 8> non_redundant_b,
                    SmallVector<SmallVector<int64_t, 8>, 8> redundant, BasicSet &a, BasicSet &b,
                    int i, int j);

// compute, whether a constraint of cut sticks out of bs by more than 2
bool stickingOut(SmallVector<SmallVector<int64_t, 8>, 8> cut, BasicSet &bs);

// transform constraints into the lambda-space needed for wrapping
SmallVector<int64_t, 8> transform(SmallVector<int64_t, 8> c);

// add a BasicSet and removes the sets at i and j
void addNewBasicSet(std::vector<BasicSet> &basic_set_vector, int i, int j,
                    BasicSet &bs);

// compute the cut case and return whether it has worked.
bool cutCase(std::vector<BasicSet> &basic_set_vector, int i, int j,
             SmallVector<SmallVector<int64_t, 8>, 8> &cut_1, SmallVector<SmallVector<int64_t, 8>, 8> &cut_2,
             SmallVector<SmallVector<int64_t, 8>, 8> &non_cut_1,
             SmallVector<SmallVector<int64_t, 8>, 8> &non_cut_2);

// compute adj_ineq pure Case and return whether it has worked
bool adjIneqPureCase(std::vector<BasicSet> &basic_set_vector, int i, int j,
                     SmallVector<SmallVector<int64_t, 8>, 8> non_adj_1,
                     SmallVector<SmallVector<int64_t, 8>, 8> non_adj_2);

// compute the non-pure adj_ineq case and return whether it has worked.
// Constraint t is the adj_ineq
bool adjIneqCase(std::vector<BasicSet> &basic_set_vector, int i, int j,
                 SmallVector<SmallVector<int64_t, 8>, 8> non_adj_1,
                 SmallVector<SmallVector<int64_t, 8>, 8> redundant_1,
                 SmallVector<SmallVector<int64_t, 8>, 8> redundant_2, SmallVector<int64_t, 8> t,
                 SmallVector<SmallVector<int64_t, 8>, 8> ineq, SmallVector<SmallVector<int64_t, 8>, 8> eq);

// compute the adj_eqCase and return whether it has worked
bool adjEqCase(std::vector<BasicSet> &basic_set_vector, int i, int j,
               SmallVector<SmallVector<int64_t, 8>, 8> redundant_1,
               SmallVector<SmallVector<int64_t, 8>, 8> redundant_2, BasicSet a, BasicSet b,
               SmallVector<int64_t, 8> t, SmallVector<SmallVector<int64_t, 8>, 8> cut,
               SmallVector<SmallVector<int64_t, 8>, 8> non_redundant_1,
               SmallVector<SmallVector<int64_t, 8>, 8> non_redundant_2, bool pure);

// compute the adj_eq Case for no CUT constraints
bool adjEqCaseNoCut(std::vector<BasicSet> &basic_set_vector, int i, int j,
                    SmallVector<int64_t, 8> t);
*/

static SmallVector<int64_t, 8> arrayRefToSmallVector(ArrayRef<int64_t> ref) {
  SmallVector<int64_t, 8> res;
  for(const int64_t &curr : ref) 
    res.push_back(curr);
  return res;
}

PresburgerSet mlir::coalesce(PresburgerSet &set) {
  PresburgerSet new_set(set.getNumDims());
  SmallVector<FlatAffineConstraints, 4> basic_set_vector = set.getFlatAffineConstraints();
  for (size_t i = 0; i < basic_set_vector.size(); i++) {
    for (size_t j = 0; j < basic_set_vector.size(); j++) {
      if (i == j)
        continue;
      FlatAffineConstraints bs1 = basic_set_vector[i];
      Simplex simplex_1(bs1);
      SmallVector<SmallVector<int64_t, 8>, 8> equalities1;
      SmallVector<SmallVector<int64_t, 8>, 8> inequalities1;
      for (size_t k = 0; k < bs1.getNumEqualities(); k++) {
        equalities1.push_back(arrayRefToSmallVector(bs1.getEquality(k)));
      }
      for (size_t k = 0; k < bs1.getNumInequalities(); k++) {
        inequalities1.push_back(arrayRefToSmallVector(bs1.getInequality(k)));
      }
      FlatAffineConstraints bs2 = basic_set_vector[j];
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

      if (!info_1->classification->redundant.empty() &&
          info_1->classification->cut.empty() &&
          info_1->classification->adj_ineq.empty() &&
          info_1->classification->adj_eq.empty() &&
          info_1->classification->separate.empty()) { // contained 2 in 1
        basic_set_vector.erase(basic_set_vector.begin() + j);
        if (j < i) {
          i--;
        }
        j--;
      } else if (!info_2->classification->redundant.empty() &&
                 info_2->classification->cut.empty() &&
                 info_2->classification->adj_ineq.empty() &&
                 info_2->classification->adj_eq.empty() &&
                 info_2->classification->separate.empty()) { // contained 1 in 2
        basic_set_vector.erase(basic_set_vector.begin() + i);
        i--;
        break;
      } /*else if (!info_1->classification->redundant.empty() &&
                 !info_1->classification->cut.empty() &&
                 info_1->classification->adj_ineq.empty() &&
                 info_1->classification->adj_eq.empty() &&
                 info_1->classification->separate
                     .empty()) { // cut or protrusion
        if (cutCase(basic_set_vector, i, j, info_1->classification->cut,
                    info_2->classification->cut,
                    info_1->classification->non_cut,
                    info_2->classification->non_cut)) {
          i--;
          break;
        } else if (stickingOut(info_1->classification->cut, basic_set2) &&
                   protrusionCase(basic_set_vector, info_1->classification->cut,
                                  info_2->classification->non_redundant,
                                  info_1->classification->redundant, basic_set1,
                                  basic_set2, i, j)) { // protrusion
          i--;
          break;
        }
      } else if (!info_2->classification->redundant.empty() &&
                 !info_2->classification->cut.empty() &&
                 info_2->classification->adj_ineq.empty() &&
                 info_2->classification->adj_eq.empty() &&
                 info_2->classification->separate
                     .empty()) { // cut or protrusion
        if (cutCase(basic_set_vector, j, i, info_2->classification->cut,
                    info_1->classification->cut,
                    info_2->classification->non_cut,
                    info_1->classification->non_cut)) {
          i--;
          break;
        } else if (stickingOut(info_2->classification->cut, basic_set1) &&
                   protrusionCase(basic_set_vector, info_2->classification->cut,
                                  info_1->classification->non_redundant,
                                  info_2->classification->redundant, basic_set2,
                                  basic_set1, j,
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
        if (adjIneqPureCase(basic_set_vector, i, j,
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
                     .empty()) { // adj_ineq complex case 1
        SmallVector<int64_t, 8> adj_ineq = info_1->classification->adj_ineq.at(0);
        if (adjIneqCase(basic_set_vector, i, j, info_1->classification->non_adj,
                        info_1->classification->redundant,
                        info_2->classification->redundant, adj_ineq,
                        inequalities_2, equalities_2)) {
          i--;
          break;
        }
      } else if (!info_2->classification->redundant.empty() &&
                 info_2->classification->adj_ineq.size() == 1 &&
                 info_2->classification->cut.empty() &&
                 info_2->classification->adj_eq.empty() &&
                 info_2->classification->separate
                     .empty()) { // adj_ineq complex case 2
        SmallVector<int64_t, 8> adj_ineq = info_2->classification->adj_ineq.at(0);
        if (adjIneqCase(basic_set_vector, j, i, info_2->classification->non_adj,
                        info_2->classification->redundant,
                        info_1->classification->redundant, adj_ineq,
                        inequalities_1, equalities_1)) {
          i--;
          break;
        }
      } else if (info_1->t && info_2->t) { // adj_eq for two equalities
        if (adjEqCase(basic_set_vector, i, j, info_1->classification->redundant,
                      info_2->classification->redundant, basic_set1, basic_set2,
                      info_1->t.value(), info_1->classification->cut,
                      info_1->classification->non_redundant,
                      info_2->classification->non_redundant, true)) {
          i--;
          break;
        }
      } else if (!info_2->classification->adj_eq.empty() &&
                 info_2->classification->cut.empty() &&
                 info_1->classification->separate
                     .empty()) { // adj_eq Case for one equality
        if (adjEqCaseNoCut(
                basic_set_vector, i, j,
                info_2->classification->adj_eq.at(0))) { // adj_eq noCut cas
          i--;
          break;
        } else if (info_1->t &&
                   adjEqCase(basic_set_vector, i, j,
                             info_1->classification->redundant,
                             info_2->classification->redundant, basic_set1,
                             basic_set2, info_1->t.value(),
                             info_1->classification->cut,
                             info_1->classification->non_redundant,
                             info_2->classification->non_redundant,
                             false)) { // adjEq case
          i--;
          break;
        }
      } else if (!info_1->classification->adj_eq.empty() &&
                 info_1->classification->cut.empty() &&
                 info_2->classification->separate
                     .empty()) { // adj_eq Case for one equality
        if (adjEqCaseNoCut(
                basic_set_vector, j, i,
                info_1->classification->adj_eq.at(0))) { // adj_eq noCut cas
          i--;
          break;
        }
        if (adjEqCase(basic_set_vector, j, i, info_2->classification->redundant,
                      info_1->classification->redundant, basic_set2, basic_set1,
                      info_2->t.value(), info_2->classification->cut,
                      info_2->classification->non_redundant,
                      info_1->classification->non_redundant,
                      false)) { // adjEq case
          i--;
          break;
        }
      }*/
      delete info_1;
      delete info_2;
    }
  }
  for (size_t i = 0; i < basic_set_vector.size(); i++) {
    new_set.addFlatAffineConstraints(basic_set_vector[i]);
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
/*
bool adjIneqPureCase(std::vector<BasicSet> &basic_set_vector, int i, int j,
                     SmallVector<SmallVector<int64_t, 8>, 8> non_adj_1,
                     SmallVector<SmallVector<int64_t, 8>, 8> non_adj_2) {
  size_t n = basic_set_vector.at(0).getNumDimensions();
  BasicSet new_set(n);
  addConstraints(new_set, non_adj_1);
  addConstraints(new_set, non_adj_2);
  addNewBasicSet(basic_set_vector, i, j, new_set);
  return true;
}

void addNewBasicSet(std::vector<BasicSet> &basic_set_vector, int i, int j,
                    BasicSet &bs) {
  if (i < j) {
    removeElement(j, basic_set_vector);
    removeElement(i, basic_set_vector);
  } else {
    removeElement(i, basic_set_vector);
    removeElement(j, basic_set_vector);
  }
  basic_set_vector.push_back(bs);
}

bool protrusionCase(std::vector<BasicSet> &basic_set_vector,
                    SmallVector<SmallVector<int64_t, 8>, 8> cut_a,
                    SmallVector<SmallVector<int64_t, 8>, 8> non_redundant_b,
                    SmallVector<SmallVector<int64_t, 8>, 8> redundant, BasicSet &a, BasicSet &b,
                    int i, int j) {
  SmallVector<SmallVector<int64_t, 8>, 8> b_eq = b.getEqualities();
  SmallVector<SmallVector<int64_t, 8>, 8> b_ineq = b.getInequalities();
  addAsIneq(b_eq, b_ineq);
  SmallVector<SmallVector<int64_t, 8>, 8> wrapped;
  for (size_t i = 0; i < cut_a.size(); i++) {
    SmallVector<int64_t, 8> curr = cut_a.at(i);
    BasicSet bPrime = BasicSet(b.getNumDimensions());
    addConstraints(bPrime, b_ineq);
    SmallVector<int64_t, 8> new_cons(curr.constant() + 1, curr.coefficientVector(),
                        Constraint::Kind::Equality);
    Simplex<> simp(bPrime);
    bPrime.addConstraint(new_cons);
    if (simp.isEmpty()) {
      redundant.push_back(curr);
      removeElement(i, cut_a);
    } else {
      curr.shift(1);
      for (size_t k = 0; k < non_redundant_b.size(); k++) {
        SmallVector<int64_t, 8> curr1 = non_redundant_b.at(k);
        Simplex simp2(bPrime);
        simp2.addEq(curr1.constant(), curr1.getIndexedCoefficients());
        if (!simp2.isEmpty()) {//This can be not sufficient!
          auto result = wrapping(a, curr, curr1);
          if (!result) {
            return false;
          }
          wrapped.push_back(result.value());
        }
      }
    }
  }
  BasicSet new_set(b.getNumDimensions());
  addConstraints(new_set, redundant);
  addConstraints(new_set, wrapped);
  for (size_t k = 0; k < cut_a.size(); k++) {
    SmallVector<int64_t, 8> curr = cut_a.at(k);
    SmallVector<int64_t, 8> new_cons(curr.constant() + 1, curr.coefficientVector(),
                        Constraint::Kind::Inequality);
    new_set.addConstraint(new_cons);
  }
  addNewBasicSet(basic_set_vector, i, j, new_set);
  return true;
}

bool stickingOut(SmallVector<SmallVector<int64_t, 8>, 8> cut, BasicSet &bs) {
  Simplex<> simp(bs);
  for (size_t k = 0; k < cut.size(); k++) {
    SmallVector<int64_t, 8> curr = cut.at(k);
    auto result =
        simp.computeOptimum(Simplex<>::Direction::DOWN, curr.constant(),
                            curr.getIndexedCoefficients());
    if (!result) {
      return false;
    }
    if (result <= -2) {
      return false;
    }
  }
  return true;
}

bool sameConstraint(SmallVector<int64_t, 8> c1, SmallVector<int64_t, 8> c2) {
  INT ratio = 0;
  bool worked = true;
  if (c2.constant() != 0) {
    ratio = c1.constant() / c2.constant();
    worked = worked && (c1.constant() == ratio * c2.constant());
  } else {
    worked = worked && (c1.constant() == c2.constant());
  }
  std::vector<INT> coeffs1, coeffs2;
  coeffs1 = c1.coefficientVector();
  coeffs2 = c2.coefficientVector();
  worked = worked && (coeffs1.size() == coeffs2.size());
  for (size_t i = 0; i < coeffs1.size(); i++) {
    if (coeffs2.at(i) != 0 && ratio == 0) {
      ratio = coeffs1.at(i) / coeffs2.at(i);
      worked = worked && (coeffs1.at(i) == ratio * coeffs2.at(i));
    } else if (coeffs2.at(i) != 0 && ratio != 0) {
      worked = worked && (coeffs1.at(i) == ratio * coeffs2.at(i));
    } else {
      worked = worked && (coeffs1.at(i) == coeffs2.at(i));
    }
  }
  return worked;
}

bool adjEqCaseNoCut(std::vector<BasicSet> &basic_set_vector, int i, int j,
                    SmallVector<int64_t, 8> t) {
  BasicSet A = basic_set_vector.at(j);
  BasicSet B = basic_set_vector.at(i);
  SmallVector<SmallVector<int64_t, 8>, 8> new_set_inequalities;
  SmallVector<SmallVector<int64_t, 8>, 8> inequalities_a = A.getInequalities();
  SmallVector<SmallVector<int64_t, 8>, 8> new_set_equalities = A.getEqualities();
  for (size_t k = 0; k < inequalities_a.size(); k++) {
    if (!sameConstraint(t, inequalities_a.at(k))) {
      new_set_inequalities.push_back(inequalities_a.at(k));
    }
  }
  t.shift(1);
  new_set_inequalities.push_back(t);

  BasicSet new_set(A.getNumDimensions());
  addConstraints(new_set, new_set_equalities);
  addConstraints(new_set, new_set_inequalities);
  Simplex<> simp(new_set);
  simp.addEq(t.constant(), t.getIndexedCoefficients());
  SmallVector<SmallVector<int64_t, 8>, 8> inequalities_b = B.getInequalities();
  SmallVector<SmallVector<int64_t, 8>, 8> equalities_b = B.getEqualities();
  addAsIneq(equalities_b, inequalities_b);
  for (size_t k = 0; k < inequalities_b.size(); k++) {
    if (simp.ineqType(inequalities_b.at(k).constant(),
                      inequalities_b.at(k).getIndexedCoefficients()) !=
        Simplex::IneqType::REDUNDANT) {
      return false;
    }
  }
  addNewBasicSet(basic_set_vector, i, j, new_set);
  return true;
}

bool adjEqCase(std::vector<BasicSet> &basic_set_vector, int i, int j,
               SmallVector<SmallVector<int64_t, 8>, 8> redundant_1,
               SmallVector<SmallVector<int64_t, 8>, 8> redundant_2, BasicSet a, BasicSet b,
               SmallVector<int64_t, 8> t, SmallVector<SmallVector<int64_t, 8>, 8> cut,
               SmallVector<SmallVector<int64_t, 8>, 8> non_redundant_1,
               SmallVector<SmallVector<int64_t, 8>, 8> non_redundant_2, bool pure) {
  SmallVector<SmallVector<int64_t, 8>, 8> wrapped;
  std::vector<INT> coeffs = t.coefficientVector();
  for (size_t k = 0; k < coeffs.size(); k++) {
    coeffs.at(k) = -coeffs.at(k);
  }
  SmallVector<int64_t, 8> minusT(-t.constant(), coeffs, Constraint::Kind::Inequality);
  for (size_t k = 0; k < non_redundant_1.size(); k++) {
    if (!sameConstraint(t, non_redundant_1.at(k))) {
      auto curr = wrapping(b, minusT, non_redundant_1.at(k));
      if (curr) {
        wrapped.push_back(curr.value());
      } else {
        return false;
      }
    }
  }
  if (!pure) {
    Simplex<> simp(b);
    for (size_t k = 0; k < wrapped.size(); k++) {
      if (simp.ineqType(wrapped.at(k).constant(),
                        wrapped.at(k).getIndexedCoefficients()) !=
          Simplex::IneqType::REDUNDANT) {
        return false;
      }
    }
  }
  t.shift(1);
  minusT.shift(-1);
  for (size_t k = 0; k < non_redundant_2.size(); k++) {
    if (!sameConstraint(minusT, non_redundant_2.at(k))) {
      auto curr = wrapping(a, t, non_redundant_2.at(k));
      if (curr) {
        wrapped.push_back(curr.value());
      } else {
        return false;
      }
    }
  }
  size_t n = b.getNumDimensions();
  BasicSet new_set(n);
  if(pure) {
    new_set.addConstraint(t);
  } else {
    t.shift(-2);
    new_set.addConstraint(Constraint::complement(t));
  }
  addConstraints(new_set, redundant_1);
  addConstraints(new_set, redundant_2);
  addConstraints(new_set, wrapped);
  addNewBasicSet(basic_set_vector, i, j, new_set);
  return true;
}

SmallVector<int64_t, 8> combineConstraint(SmallVector<int64_t, 8> c1, SmallVector<int64_t, 8> c2, SafeRational ratio) {
  INT n = ratio.num;
  INT d = ratio.den;
  INT constant = -n * c1.constant() + d * c2.constant();
  std::vector<INT> coeffs1, coeffs2, newCoeffs;
  coeffs1 = c1.coefficientVector();
  coeffs2 = c2.coefficientVector();
  for (size_t i = 0; i < coeffs1.size(); i++) {
    newCoeffs.push_back(-n * coeffs1.at(i) + d * coeffs2.at(i));
  }
  return SmallVector<int64_t, 8>(constant, newCoeffs, Constraint::Kind::Inequality);
}

SmallVector<int64_t, 8> transform(SmallVector<int64_t, 8> c) {
  std::vector<INT> coeffs = c.coefficientVector();
  coeffs.push_back(c.constant());
  SmallVector<int64_t, 8> new_cons(0, coeffs, c.getKind());
  return new_cons;
}

Optional<SmallVector<int64_t, 8>> wrapping(BasicSet bs, SmallVector<int64_t, 8> valid,
                                   SmallVector<int64_t, 8> invalid) {
  size_t n = bs.getNumDimensions();
  Simplex<> simp(n + 1);
  SmallVector<ArrayRef<int64>, 64> eqs = bs.getEqualities();
  SmallVector<SmallVector<int64_t, 8>, 8> ineqs = bs.getInequalities();

  for (size_t k = 0; k < bs.getNumEqualities(); k++) {
    Constraint new_cons = transform(eqs.at(k));
    simp.addConstraint(new_cons);
  }
  for (size_t k = 0; k < bs.getNumInequalities(); k++) {
    SmallVector<int64_t, 8> new_cons = transform(ineqs.at(k));
    simp.addConstraint(new_cons);
  }

  std::vector<INT> lambda;
  lambda.reserve(n + 1);
  for (size_t k = 0; k < n; k++) {
    lambda.push_back(0);
  }
  lambda.push_back(1);
  SmallVector<int64_t, 8> lambda_cons(0, lambda, Constraint::Kind::Inequality);
  simp.addConstraint(lambda_cons);

  SmallVector<int64_t, 8> transformed_valid = transform(valid);
  SmallVector<int64_t, 8> x1(-1, transformed_valid.coefficientVector(),
                Constraint::Kind::Equality);
  simp.addConstraint(x1);

  SmallVector<int64_t, 8> transformed_invalid = transform(invalid);
  Optional<SafeRational> result = simp.computeOptimum(
      Simplex<>::Direction::DOWN, transformed_invalid.constant(),
      transformed_invalid.getIndexedCoefficients());
  if (!result) {
    return {};
  }
  return combineConstraint(valid, invalid, result.value());
}
*/
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

/*bool adjIneqCase(std::vector<BasicSet> &basic_set_vector, int i, int j,
                 SmallVector<SmallVector<int64_t, 8>, 8> non_adj_1,
                 SmallVector<SmallVector<int64_t, 8>, 8> redundant_1,
                 SmallVector<SmallVector<int64_t, 8>, 8> redundant_2, SmallVector<int64_t, 8> t,
                 SmallVector<SmallVector<int64_t, 8>, 8> ineq, SmallVector<SmallVector<int64_t, 8>, 8> eq) {
  size_t n = basic_set_vector.at(i).getNumDimensions();
  BasicSet bs(n);
  addConstraints(bs, non_adj_1);
  addConstraints(bs, redundant_2);
  bs.addConstraint(Constraint::complement(t));
  Simplex complement(bs);
  Simplex original(bs);
  for (size_t k = 0; k < ineq.size(); k++) {
    if (complement.ineqType(ineq.at(k).constant(),
                            ineq.at(k).getIndexedCoefficients()) !=
            Simplex::IneqType::REDUNDANT &&
        original.ineqType(ineq.at(k).constant(),
                          ineq.at(k).getIndexedCoefficients()) !=
            Simplex::IneqType::REDUNDANT) {
      return false;
    }
  }
  for (size_t k = 0; k < eq.size(); k++) {
    if (complement.ineqType(eq.at(k).constant(),
                            eq.at(k).getIndexedCoefficients()) !=
            Simplex::IneqType::REDUNDANT &&
        original.ineqType(eq.at(k).constant(),
                          eq.at(k).getIndexedCoefficients()) !=
            Simplex::IneqType::REDUNDANT) {
      return false;
    }
  }
  BasicSet new_set(n);
  addConstraints(new_set, redundant_1);
  addConstraints(new_set, redundant_2);
  addNewBasicSet(basic_set_vector, i, j, new_set);
  return true;
}

bool cutCase(std::vector<BasicSet> &basic_set_vector, int i, int j,
             SmallVector<SmallVector<int64_t, 8>, 8> &cut_1, SmallVector<SmallVector<int64_t, 8>, 8> &cut_2,
             SmallVector<SmallVector<int64_t, 8>, 8> &non_cut_1,
             SmallVector<SmallVector<int64_t, 8>, 8> &non_cut_2) {
  for (size_t k = 0; k < cut_1.size(); k++) {
    if (!containedFacet(cut_1.at(k), basic_set_vector.at(i), cut_2)) {
      return false;
    }
  }
  size_t n = basic_set_vector.at(i).getNumDimensions();
  BasicSet new_set(n);
  addConstraints(new_set, non_cut_1);
  addConstraints(new_set, non_cut_2);
  addNewBasicSet(basic_set_vector, i, j, new_set);
  return true;
}

bool containedFacet(SmallVector<int64_t, 8> &ineq, BasicSet &bs,
                    SmallVector<SmallVector<int64_t, 8>, 8> &cut) {
  Simplex simp(bs);
  simp.addEq(ineq.constant(), ineq.getIndexedCoefficients());
  for (size_t i = 0; i < cut.size(); i++) {
    if (simp.ineqType(cut.at(i).constant(),
                      cut.at(i).getIndexedCoefficients()) !=
        Simplex::IneqType::REDUNDANT) {
      return false;
    }
  }
  return true;
}
*/
void mlir::addAsIneq(SmallVector<SmallVector<int64_t, 8>, 8> eq, SmallVector<SmallVector<int64_t, 8>, 8> &target) {
  for (size_t i = 0; i < eq.size(); i++) {
    SmallVector<int64_t, 8> curr = eq[i];
    target.push_back(curr);
    SmallVector<int64_t, 8> complement;
    for (size_t k = 0; k < curr.size(); k++) {
      complement.push_back(curr[k]);
    }
    target.push_back(complement);
  }
}

/*This function removes the BasicSet at position pos
 */
/*template <typename T> void removeElement(int pos, std::vector<T> &vec) {
  vec.erase(vec.begin() + pos);
}*/

