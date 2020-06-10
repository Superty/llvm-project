#include "mlir/Analysis/Presburger/Coalesce.h"
#include "mlir/Analysis/Presburger/Simplex.h"

namespace mlir {

// struct for classified constraints
struct Classification {
  SmallVector<ArrayRef<int64>, 64> redundant;
  SmallVector<ArrayRef<int64>, 64> cut;
  SmallVector<ArrayRef<int64>, 64> adj_ineq;
  SmallVector<ArrayRef<int64>, 64> adj_eq;
  SmallVector<ArrayRef<int64>, 64> separate;
  SmallVector<ArrayRef<int64>, 64> non_cut;
  SmallVector<ArrayRef<int64>, 64> non_adj;
  SmallVector<ArrayRef<int64>, 64> non_redundant;
};

// struct for all infos needed from classification
struct Info {
  std::optional<ArrayRef<int64_t>> t;
  Classification *classification;
  ~Info() { delete classification; }
};

//adds all Constraints to bs
void addConstraints(BasicSet &bs, SmallVector<ArrayRef<int64>, 64> constraints);

// classify of all constraints into redundant, cut, adj_ineq, adj_eq, separate,
// non_cut, non_adj, where non_cut and non_adj are everything but cut or
// adj_ineq respectively
void classify_ineq(Simplex<> &simp, SmallVector<ArrayRef<int64>, 64> &constraints,
                   Classification *classi);

// same thing as classify_ineq, but also return if there is an equality
// constraint adjacent to a the other polytope
void classify(Simplex<> &simp, SmallVector<ArrayRef<int64>, 64> inequalities,
              SmallVector<ArrayRef<int64>, 64> equalities, Info *info);

/*// compute the protrusionCase and return whether it has worked
bool protrusionCase(std::vector<BasicSet> &basic_set_vector,
                    std::vector<Constraint> cut_a,
                    SmallVector<ArrayRef<int64>, 64> non_redundant_b,
                    SmallVector<ArrayRef<int64>, 64> redundant, BasicSet &a, BasicSet &b,
                    int i, int j);

// compute, whether a constraint of cut sticks out of bs by more than 2
bool stickingOut(SmallVector<ArrayRef<int64>, 64> cut, BasicSet &bs);

// transform constraints into the lambda-space needed for wrapping
ArrayRef<int64_t> transform(ArrayRef<int64_t> c);

// add a BasicSet and removes the sets at i and j
void addNewBasicSet(std::vector<BasicSet> &basic_set_vector, int i, int j,
                    BasicSet &bs);

// compute the cut case and return whether it has worked.
bool cutCase(std::vector<BasicSet> &basic_set_vector, int i, int j,
             SmallVector<ArrayRef<int64>, 64> &cut_1, SmallVector<ArrayRef<int64>, 64> &cut_2,
             SmallVector<ArrayRef<int64>, 64> &non_cut_1,
             SmallVector<ArrayRef<int64>, 64> &non_cut_2);

// compute adj_ineq pure Case and return whether it has worked
bool adjIneqPureCase(std::vector<BasicSet> &basic_set_vector, int i, int j,
                     SmallVector<ArrayRef<int64>, 64> non_adj_1,
                     SmallVector<ArrayRef<int64>, 64> non_adj_2);

// compute the non-pure adj_ineq case and return whether it has worked.
// Constraint t is the adj_ineq
bool adjIneqCase(std::vector<BasicSet> &basic_set_vector, int i, int j,
                 SmallVector<ArrayRef<int64>, 64> non_adj_1,
                 SmallVector<ArrayRef<int64>, 64> redundant_1,
                 SmallVector<ArrayRef<int64>, 64> redundant_2, ArrayRef<int64_t> t,
                 SmallVector<ArrayRef<int64>, 64> ineq, SmallVector<ArrayRef<int64>, 64> eq);

// compute the adj_eqCase and return whether it has worked
bool adjEqCase(std::vector<BasicSet> &basic_set_vector, int i, int j,
               SmallVector<ArrayRef<int64>, 64> redundant_1,
               SmallVector<ArrayRef<int64>, 64> redundant_2, BasicSet a, BasicSet b,
               ArrayRef<int64_t> t, SmallVector<ArrayRef<int64>, 64> cut,
               SmallVector<ArrayRef<int64>, 64> non_redundant_1,
               SmallVector<ArrayRef<int64>, 64> non_redundant_2, bool pure);

// compute the adj_eq Case for no CUT constraints
bool adjEqCaseNoCut(std::vector<BasicSet> &basic_set_vector, int i, int j,
                    ArrayRef<int64_t> t);
*/

Set coalesce(Set &set) {
  Set new_set(set.getNumDimensions());
  std::vector<BasicSet> basic_set_vector = set.getBasicSets();
  for (size_t i = 0; i < basic_set_vector.size(); i++) {
    for (size_t j = 0; j < basic_set_vector.size(); j++) {
      if (i == j)
        continue;
      BasicSet basic_set1 = basic_set_vector.at(i);
      Simplex<> simplex_1(basic_set1);
      SmallVector<ArrayRef<int64>, 64> equalities_1 = basic_set1.getEqualities();
      SmallVector<ArrayRef<int64>, 64> inequalities_1 = basic_set1.getInequalities();
      BasicSet basic_set2 = basic_set_vector.at(j);
      Simplex<> simplex_2(basic_set2);
      SmallVector<ArrayRef<int64>, 64> equalities_2 = basic_set2.getEqualities();
      SmallVector<ArrayRef<int64>, 64> inequalities_2 = basic_set2.getInequalities();
      Info *info_1 = new Info();
      Info *info_2 = new Info();
      classify(simplex_2, inequalities_1, equalities_1, info_1);
      classify(simplex_1, inequalities_2, equalities_2, info_2);

      if (!info_1->classification->redundant.empty() &&
          info_1->classification->cut.empty() &&
          info_1->classification->adj_ineq.empty() &&
          info_1->classification->adj_eq.empty() &&
          info_1->classification->separate.empty()) { // contained 2 in 1
        removeElement(j, basic_set_vector);
        if (j < i) {
          i--;
        }
        j--;
      } else if (!info_2->classification->redundant.empty() &&
                 info_2->classification->cut.empty() &&
                 info_2->classification->adj_ineq.empty() &&
                 info_2->classification->adj_eq.empty() &&
                 info_2->classification->separate.empty()) { // contained 1 in 2
        removeElement(i, basic_set_vector);
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
        ArrayRef<int64_t> adj_ineq = info_1->classification->adj_ineq.at(0);
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
        ArrayRef<int64_t> adj_ineq = info_2->classification->adj_ineq.at(0);
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
    new_set.addBasicSet(basic_set_vector.at(i));
  }
  return new_set;
}


void addConstraints(BasicSet &bs, SmallVector<ArrayRef<int64>, 64> constraints) {
  for (size_t k = 0; k < constraints.size(); k++) {
    bs.addConstraint(constraints.at(k));
  }
}
/*
bool adjIneqPureCase(std::vector<BasicSet> &basic_set_vector, int i, int j,
                     SmallVector<ArrayRef<int64>, 64> non_adj_1,
                     SmallVector<ArrayRef<int64>, 64> non_adj_2) {
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
                    SmallVector<ArrayRef<int64>, 64> cut_a,
                    SmallVector<ArrayRef<int64>, 64> non_redundant_b,
                    SmallVector<ArrayRef<int64>, 64> redundant, BasicSet &a, BasicSet &b,
                    int i, int j) {
  SmallVector<ArrayRef<int64>, 64> b_eq = b.getEqualities();
  SmallVector<ArrayRef<int64>, 64> b_ineq = b.getInequalities();
  addAsIneq(b_eq, b_ineq);
  SmallVector<ArrayRef<int64>, 64> wrapped;
  for (size_t i = 0; i < cut_a.size(); i++) {
    ArrayRef<int64_t> curr = cut_a.at(i);
    BasicSet bPrime = BasicSet(b.getNumDimensions());
    addConstraints(bPrime, b_ineq);
    ArrayRef<int64_t> new_cons(curr.constant() + 1, curr.coefficientVector(),
                        Constraint::Kind::Equality);
    Simplex<> simp(bPrime);
    bPrime.addConstraint(new_cons);
    if (simp.isEmpty()) {
      redundant.push_back(curr);
      removeElement(i, cut_a);
    } else {
      curr.shift(1);
      for (size_t k = 0; k < non_redundant_b.size(); k++) {
        ArrayRef<int64_t> curr1 = non_redundant_b.at(k);
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
    ArrayRef<int64_t> curr = cut_a.at(k);
    ArrayRef<int64_t> new_cons(curr.constant() + 1, curr.coefficientVector(),
                        Constraint::Kind::Inequality);
    new_set.addConstraint(new_cons);
  }
  addNewBasicSet(basic_set_vector, i, j, new_set);
  return true;
}

bool stickingOut(SmallVector<ArrayRef<int64>, 64> cut, BasicSet &bs) {
  Simplex<> simp(bs);
  for (size_t k = 0; k < cut.size(); k++) {
    ArrayRef<int64_t> curr = cut.at(k);
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

bool sameConstraint(ArrayRef<int64_t> c1, ArrayRef<int64_t> c2) {
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
                    ArrayRef<int64_t> t) {
  BasicSet A = basic_set_vector.at(j);
  BasicSet B = basic_set_vector.at(i);
  SmallVector<ArrayRef<int64>, 64> new_set_inequalities;
  SmallVector<ArrayRef<int64>, 64> inequalities_a = A.getInequalities();
  SmallVector<ArrayRef<int64>, 64> new_set_equalities = A.getEqualities();
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
  SmallVector<ArrayRef<int64>, 64> inequalities_b = B.getInequalities();
  SmallVector<ArrayRef<int64>, 64> equalities_b = B.getEqualities();
  addAsIneq(equalities_b, inequalities_b);
  for (size_t k = 0; k < inequalities_b.size(); k++) {
    if (simp.ineqType(inequalities_b.at(k).constant(),
                      inequalities_b.at(k).getIndexedCoefficients()) !=
        IneqType::REDUNDANT) {
      return false;
    }
  }
  addNewBasicSet(basic_set_vector, i, j, new_set);
  return true;
}

bool adjEqCase(std::vector<BasicSet> &basic_set_vector, int i, int j,
               SmallVector<ArrayRef<int64>, 64> redundant_1,
               SmallVector<ArrayRef<int64>, 64> redundant_2, BasicSet a, BasicSet b,
               ArrayRef<int64_t> t, SmallVector<ArrayRef<int64>, 64> cut,
               SmallVector<ArrayRef<int64>, 64> non_redundant_1,
               SmallVector<ArrayRef<int64>, 64> non_redundant_2, bool pure) {
  SmallVector<ArrayRef<int64>, 64> wrapped;
  std::vector<INT> coeffs = t.coefficientVector();
  for (size_t k = 0; k < coeffs.size(); k++) {
    coeffs.at(k) = -coeffs.at(k);
  }
  ArrayRef<int64_t> minusT(-t.constant(), coeffs, Constraint::Kind::Inequality);
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
          IneqType::REDUNDANT) {
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

ArrayRef<int64_t> combineConstraint(ArrayRef<int64_t> c1, ArrayRef<int64_t> c2, SafeRational ratio) {
  INT n = ratio.num;
  INT d = ratio.den;
  INT constant = -n * c1.constant() + d * c2.constant();
  std::vector<INT> coeffs1, coeffs2, newCoeffs;
  coeffs1 = c1.coefficientVector();
  coeffs2 = c2.coefficientVector();
  for (size_t i = 0; i < coeffs1.size(); i++) {
    newCoeffs.push_back(-n * coeffs1.at(i) + d * coeffs2.at(i));
  }
  return ArrayRef<int64_t>(constant, newCoeffs, Constraint::Kind::Inequality);
}

ArrayRef<int64_t> transform(ArrayRef<int64_t> c) {
  std::vector<INT> coeffs = c.coefficientVector();
  coeffs.push_back(c.constant());
  ArrayRef<int64_t> new_cons(0, coeffs, c.getKind());
  return new_cons;
}

std::optional<ArrayRef<int64_t>> wrapping(BasicSet bs, ArrayRef<int64_t> valid,
                                   ArrayRef<int64_t> invalid) {
  size_t n = bs.getNumDimensions();
  Simplex<> simp(n + 1);
  SmallVector<ArrayRef<int64>, 64> eqs = bs.getEqualities();
  SmallVector<ArrayRef<int64>, 64> ineqs = bs.getInequalities();

  for (size_t k = 0; k < bs.getNumEqualities(); k++) {
    Constraint new_cons = transform(eqs.at(k));
    simp.addConstraint(new_cons);
  }
  for (size_t k = 0; k < bs.getNumInequalities(); k++) {
    ArrayRef<int64_t> new_cons = transform(ineqs.at(k));
    simp.addConstraint(new_cons);
  }

  std::vector<INT> lambda;
  lambda.reserve(n + 1);
  for (size_t k = 0; k < n; k++) {
    lambda.push_back(0);
  }
  lambda.push_back(1);
  ArrayRef<int64_t> lambda_cons(0, lambda, Constraint::Kind::Inequality);
  simp.addConstraint(lambda_cons);

  ArrayRef<int64_t> transformed_valid = transform(valid);
  ArrayRef<int64_t> x1(-1, transformed_valid.coefficientVector(),
                Constraint::Kind::Equality);
  simp.addConstraint(x1);

  ArrayRef<int64_t> transformed_invalid = transform(invalid);
  std::optional<SafeRational> result = simp.computeOptimum(
      Simplex<>::Direction::DOWN, transformed_invalid.constant(),
      transformed_invalid.getIndexedCoefficients());
  if (!result) {
    return {};
  }
  return combineConstraint(valid, invalid, result.value());
}
*/
void classify(Simplex &simp, SmallVector<ArrayRef<int64>, 64> inequalities,
              SmallVector<ArrayRef<int64>, 64> equalities, Info *info) {
  std::optional<ArrayRef<int64_t>> dummy = {};
  Classification *ineqs = new Classification();
  classify_ineq(simp, inequalities, ineqs);
  info->classification = ineqs;
  SmallVector<ArrayRef<int64>, 64> eqAsIneq;
  addAsIneq(equalities, eqAsIneq);
  for (ArrayRef<int64_t> current_constraint : eqAsIneq) {
    IneqType ty = simp.ineqType(current_constraint.constant(),
                                current_constraint.getIndexedCoefficients());
    switch (ty) {
    case IneqType::REDUNDANT:
      ineqs->redundant.push_back(current_constraint);
      ineqs->non_cut.push_back(current_constraint);
      ineqs->non_adj.push_back(current_constraint);
      break;
    case IneqType::CUT:
      ineqs->cut.push_back(current_constraint);
      ineqs->non_adj.push_back(current_constraint);
      ineqs->non_redundant.push_back(current_constraint);
      break;
    case IneqType::ADJ_INEQ:
      ineqs->adj_ineq.push_back(current_constraint);
      ineqs->non_cut.push_back(current_constraint);
      ineqs->non_redundant.push_back(current_constraint);
      info->t = current_constraint;
      break;
    case IneqType::ADJ_EQ:
      ineqs->adj_eq.push_back(current_constraint);
      ineqs->non_cut.push_back(current_constraint);
      ineqs->non_adj.push_back(current_constraint);
      info->t = current_constraint;
      break;
    case IneqType::SEPARATE:
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
void classify_ineq(Simplex &simp, SmallVector<ArrayRef<int64_t>, 64> &constraints,
                   Classification *classi) {
  for (ArrayRef<int64_t> current_constraint : constraints) {
    IneqType ty = simp.ineqType(current_constraint);
    switch (ty) {
    case IneqType::REDUNDANT:
      classi->redundant.push_back(current_constraint);
      classi->non_cut.push_back(current_constraint);
      classi->non_adj.push_back(current_constraint);
      break;
    case IneqType::CUT:
      classi->cut.push_back(current_constraint);
      classi->non_adj.push_back(current_constraint);
      classi->non_redundant.push_back(current_constraint);
      break;
    case IneqType::ADJ_INEQ:
      classi->adj_ineq.push_back(current_constraint);
      classi->non_cut.push_back(current_constraint);
      classi->non_redundant.push_back(current_constraint);
      break;
    case IneqType::ADJ_EQ:
      classi->adj_eq.push_back(current_constraint);
      classi->non_cut.push_back(current_constraint);
      classi->non_adj.push_back(current_constraint);
      break;
    case IneqType::SEPARATE:
      classi->separate.push_back(current_constraint);
      classi->non_cut.push_back(current_constraint);
      classi->non_adj.push_back(current_constraint);
      classi->non_redundant.push_back(current_constraint);
      break;
    }
  }
}

/*bool adjIneqCase(std::vector<BasicSet> &basic_set_vector, int i, int j,
                 SmallVector<ArrayRef<int64>, 64> non_adj_1,
                 SmallVector<ArrayRef<int64>, 64> redundant_1,
                 SmallVector<ArrayRef<int64>, 64> redundant_2, ArrayRef<int64_t> t,
                 SmallVector<ArrayRef<int64>, 64> ineq, SmallVector<ArrayRef<int64>, 64> eq) {
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
            IneqType::REDUNDANT &&
        original.ineqType(ineq.at(k).constant(),
                          ineq.at(k).getIndexedCoefficients()) !=
            IneqType::REDUNDANT) {
      return false;
    }
  }
  for (size_t k = 0; k < eq.size(); k++) {
    if (complement.ineqType(eq.at(k).constant(),
                            eq.at(k).getIndexedCoefficients()) !=
            IneqType::REDUNDANT &&
        original.ineqType(eq.at(k).constant(),
                          eq.at(k).getIndexedCoefficients()) !=
            IneqType::REDUNDANT) {
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
             SmallVector<ArrayRef<int64>, 64> &cut_1, SmallVector<ArrayRef<int64>, 64> &cut_2,
             SmallVector<ArrayRef<int64>, 64> &non_cut_1,
             SmallVector<ArrayRef<int64>, 64> &non_cut_2) {
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

bool containedFacet(ArrayRef<int64_t> &ineq, BasicSet &bs,
                    SmallVector<ArrayRef<int64>, 64> &cut) {
  Simplex simp(bs);
  simp.addEq(ineq.constant(), ineq.getIndexedCoefficients());
  for (size_t i = 0; i < cut.size(); i++) {
    if (simp.ineqType(cut.at(i).constant(),
                      cut.at(i).getIndexedCoefficients()) !=
        IneqType::REDUNDANT) {
      return false;
    }
  }
  return true;
}
*/
void addAsIneq(SmallVector<ArrayRef<int64>, 64> eq, SmallVector<ArrayRef<int64>, 64> &target) {
  for (size_t i = 0; i < eq.size(); i++) {
    ArrayRef<int64_t> curr =
        Constraint::inequalityFromEquality(eq.at(i), false, false);
    ArrayRef<int64_t> complement =
        Constraint::inequalityFromEquality(eq.at(i), true, false);
    target.push_back(curr);
    target.push_back(complement);
  }
}

/*This function removes the BasicSet at position pos
 */
/*template <typename T> void removeElement(int pos, std::vector<T> &vec) {
  vec.erase(vec.begin() + pos);
}*/
} // namespace mlir

