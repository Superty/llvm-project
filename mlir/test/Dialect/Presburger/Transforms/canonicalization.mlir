// RUN: presburgerc -canonicalize %s | FileCheck %s

// CHECK-LABEL: func @simple_union
func @simple_union() -> !presburger.set<1,0> {
  %set1 = presburger.set #presburger<"(x)[] : (x >= 0)">
  %set2 = presburger.set #presburger<"(x)[] : (x - 1 >= 0)">

  // CHECK-NEXT: %[[S:.*]] = presburger.set #presburger<"(d0)[] : (d0 >= 0 or d0 - 1 >= 0)">
  // CHECK-NEXT: return %[[S]]
  %uset = presburger.union %set1, %set2 : !presburger.set<1,0>
  return %uset : !presburger.set<1,0>
}

// -----

// CHECK-LABEL: func @union_multi_dim
func @union_multi_dim() -> !presburger.set<2,0> {
  %set1 = presburger.set #presburger<"(x, y)[] : (x >= 0 and -x + 10 >= 0)">
  %set2 = presburger.set #presburger<"(x, z)[] : (x - 1 >= 0 or z <= 42)">

  // CHECK-NEXT: %[[S:.*]] = presburger.set #presburger<"(d0, d1)[] : (d0 >= 0 and -d0 + 10 >= 0 or d0 - 1 >= 0 or -d1 + 42 >= 0)">
  // CHECK-NEXT: return %[[S]]
  %uset = presburger.union %set1, %set2 : !presburger.set<2,0>
  return %uset : !presburger.set<2,0>
}

// -----

// CHECK-LABEL: func @union_with_sym
func @union_with_sym() -> !presburger.set<1,1> {
  %set1 = presburger.set #presburger<"(x)[N] : (x >= 0 and -x + N >= 0)">
  %set2 = presburger.set #presburger<"(y)[M] : (y - 1 >= 0 or M + y <= 42)">

  // CHECK-NEXT: %[[S:.*]] = presburger.set #presburger<"(d0)[s0] : (d0 >= 0 and -d0 + s0 >= 0 or d0 - 1 >= 0 or -d0 - s0 + 42 >= 0)">
  // CHECK-NEXT: return %[[S]]
  %uset = presburger.union %set1, %set2 : !presburger.set<1,1>
  return %uset : !presburger.set<1,1>
}

// -----

// CHECK-LABEL: func @simple_intersect
func @simple_intersect() -> !presburger.set<1,0> {
  %set1 = presburger.set #presburger<"(x)[] : (x >= 0)">
  %set2 = presburger.set #presburger<"(x)[] : (x - 1 >= 0)">

  // CHECK-NEXT: %[[S:.*]] = presburger.set #presburger<"(d0)[] : (d0 >= 0 and d0 - 1 >= 0)">
  // CHECK-NEXT: return %[[S]]
  %uset = presburger.intersect %set1, %set2 : !presburger.set<1,0>
  return %uset : !presburger.set<1,0>
}

// -----

// CHECK-LABEL: func @intersect_multi_dim
func @intersect_multi_dim() -> !presburger.set<2,0> {
  %set1 = presburger.set #presburger<"(x, y)[] : (x >= 0 and -x + 10 >= 0)">
  %set2 = presburger.set #presburger<"(x, z)[] : (x - 1 >= 0 or z <= 42)">

  // CHECK-NEXT: %[[S:.*]] = presburger.set #presburger<"(d0, d1)[] : (d0 >= 0 and -d0 + 10 >= 0 and d0 - 1 >= 0 or d0 >= 0 and -d0 + 10 >= 0 and -d1 + 42 >= 0)">
  // CHECK-NEXT: return %[[S]]
  %uset = presburger.intersect %set1, %set2 : !presburger.set<2,0>
  return %uset : !presburger.set<2,0>
}

// -----

// CHECK-LABEL: func @combined
func @combined() -> !presburger.set<1,0> {
  %set1 = presburger.set #presburger<"(x)[] : (x >= 0)">
  %set2 = presburger.set #presburger<"(x)[] : (x - 1 >= 0)">
  %set3 = presburger.set #presburger<"(y)[] : (-y + 42 >= 0)">

  // CHECK-NEXT: %[[S:.*]] = presburger.set #presburger<"(d0)[] : (d0 >= 0 and d0 - 1 >= 0 or -d0 + 42 >= 0)">
  // CHECK-NEXT: return %[[S]]
  %iset = presburger.intersect %set1, %set2 : !presburger.set<1,0>
  %uset = presburger.union %iset, %set3 : !presburger.set<1,0>
  return %uset : !presburger.set<1,0>
}

// -----
