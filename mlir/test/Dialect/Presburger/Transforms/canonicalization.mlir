// RUN: mlir-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: func @simple_union
func @simple_union() -> !presburger.set<1,0> {
  %set1 = presburger.set #presburger<"set(x) : (x >= 0)">
  %set2 = presburger.set #presburger<"set(x) : (x - 1 >= 0)">

  // CHECK-NEXT: %[[S:.*]] = presburger.set #presburger<"set(d0) : (d0 >= 0)">
  // CHECK-NEXT: return %[[S]]
  %uset = presburger.union %set1, %set2 : !presburger.set<1,0>
  return %uset : !presburger.set<1,0>
}

// -----

// CHECK-LABEL: func @union_multi_dim
func @union_multi_dim() -> !presburger.set<2,0> {
  %set1 = presburger.set #presburger<"set(x, y) : (x >= 0 and -x + 10 >= 0)">
  %set2 = presburger.set #presburger<"set(x, z) : (x - 1 >= 0 or z <= 42)">

  // CHECK-NEXT: %[[S:.*]] = presburger.set #presburger<"set(d0, d1) : (d0 >= 0 and -d0 + 10 >= 0 or d0 - 1 >= 0 or -d1 + 42 >= 0)">
  // CHECK-NEXT: return %[[S]]
  %uset = presburger.union %set1, %set2 : !presburger.set<2,0>
  return %uset : !presburger.set<2,0>
}

// -----

// CHECK-LABEL: func @simple_intersect
func @simple_intersect() -> !presburger.set<1,0> {
  %set1 = presburger.set #presburger<"set(x) : (x >= 0)">
  %set2 = presburger.set #presburger<"set(x) : (x - 1 >= 0)">

  // CHECK-NEXT: %[[S:.*]] = presburger.set #presburger<"set(d0) : (d0 - 1 >= 0)">
  // CHECK-NEXT: return %[[S]]
  %uset = presburger.intersect %set1, %set2 : !presburger.set<1,0>
  return %uset : !presburger.set<1,0>
}

// -----

// CHECK-LABEL: func @intersect_multi_dim
func @intersect_multi_dim() -> !presburger.set<2,0> {
  %set1 = presburger.set #presburger<"set(x, y) : (x >= 0 and -x + 10 >= 0)">
  %set2 = presburger.set #presburger<"set(x, z) : (x - 1 >= 0 or z <= 42)">

  // CHECK-NEXT: %[[S:.*]] = presburger.set #presburger<"set(d0, d1) : (d0 >= 0 and -d0 + 10 >= 0 and d0 - 1 >= 0 or d0 >= 0 and -d0 + 10 >= 0 and -d1 + 42 >= 0)">
  // CHECK-NEXT: return %[[S]]
  %uset = presburger.intersect %set1, %set2 : !presburger.set<2,0>
  return %uset : !presburger.set<2,0>
}

// -----

// CHECK-LABEL: func @combined
func @combined() -> !presburger.set<1,0> {
  %set1 = presburger.set #presburger<"set(x) : (x >= 0)">
  %set2 = presburger.set #presburger<"set(x) : (x - 1 >= 0)">
  %set3 = presburger.set #presburger<"set(y) : (-y + 42 >= 0)">

  // CHECK-NEXT: %[[S:.*]] = presburger.set #presburger<"set(d0) : ()">
  // CHECK-NEXT: return %[[S]]
  %iset = presburger.intersect %set1, %set2 : !presburger.set<1,0>
  %uset = presburger.union %iset, %set3 : !presburger.set<1,0>
  return %uset : !presburger.set<1,0>
}

// -----
// Subtract 


// CHECK-LABEL: func @simple_subtract
func @simple_subtract() -> !presburger.set<1,0> {
  %set1 = presburger.set #presburger<"set(x) : (x + 1 >= 0)">
  %set2 = presburger.set #presburger<"set(x) : (x >= 0)">

  // CHECK-NEXT: %[[S:.*]] = presburger.set #presburger<"set(d0) : (d0 + 1 >= 0 and -d0 - 1 >= 0)">
  // CHECK-NEXT: return %[[S]]
  %uset = presburger.subtract %set1, %set2 : !presburger.set<1,0>
  return %uset : !presburger.set<1,0>
}

// -----

// CHECK-LABEL: func @subtract_multi_dim
func @subtract_multi_dim() -> !presburger.set<3,0> {
  %set1 = presburger.set #presburger<"set(x,y,z) : (x + y + z >= 0 and x + y - 10 >= 0)">
  %set2 = presburger.set #presburger<"set(x,y,z) : (x >= 0)">

  // CHECK-NEXT: %[[S:.*]] = presburger.set #presburger<"set(d0, d1, d2) : (d0 + d1 + d2 >= 0 and d0 + d1 - 10 >= 0 and -d0 - 1 >= 0)">
  // CHECK-NEXT: return %[[S]]
  %uset = presburger.subtract %set1, %set2 : !presburger.set<3,0>
  return %uset : !presburger.set<3,0>
}

// -----

// CHECK-LABEL: func @subtract_simple_equality
func @subtract_simple_equality() -> !presburger.set<1,0> {
  %set1 = presburger.set #presburger<"set(x) : (x + 1 >= 0)">
  %set2 = presburger.set #presburger<"set(x) : (x = 0)">

  // CHECK-NEXT: %[[S:.*]] = presburger.set #presburger<"set(d0) : (d0 + 1 >= 0 and -d0 - 1 >= 0 or d0 + 1 >= 0 and d0 - 1 >= 0)">
  // CHECK-NEXT: return %[[S]]
  %uset = presburger.subtract %set1, %set2 : !presburger.set<1,0>
  return %uset : !presburger.set<1,0>
}

// -----

// CHECK-LABEL: func @subtract_multi_dim_equality
func @subtract_multi_dim_equality() -> !presburger.set<2,0> {
  %set1 = presburger.set #presburger<"set(x, y) : (x + 1 >= 0)">
  %set2 = presburger.set #presburger<"set(x, y) : (x = y)">

  // CHECK-NEXT: %[[S:.*]] = presburger.set #presburger<"set(d0, d1) : (d0 + 1 >= 0 and -d0 + d1 - 1 >= 0 or d0 + 1 >= 0 and d0 - d1 - 1 >= 0)">
  // CHECK-NEXT: return %[[S]]
  %uset = presburger.subtract %set1, %set2 : !presburger.set<2,0>
  return %uset : !presburger.set<2,0>
}

// -----

// CHECK-LABEL: func @subtract_non_convex_eqs
func @subtract_non_convex_eqs() -> !presburger.set<2,0> {
  %set1 = presburger.set #presburger<"set(x, y) : (x + 1 >= 0)">
  %set2 = presburger.set #presburger<"set(x, y) : (x = y or x + y = 0)">

  // CHECK-NEXT: %[[S:.*]] = presburger.set #presburger<"set(d0, d1) : (d0 + 1 >= 0 and -d0 + d1 - 1 >= 0 and -d0 - d1 - 1 >= 0 or d0 + 1 >= 0 and -d0 + d1 - 1 >= 0 and d0 + d1 - 1 >= 0 or d0 + 1 >= 0 and d0 - d1 - 1 >= 0 and -d0 - d1 - 1 >= 0 or d0 + 1 >= 0 and d0 - d1 - 1 >= 0 and d0 + d1 - 1 >= 0)">
  // CHECK-NEXT: return %[[S]]
  %uset = presburger.subtract %set1, %set2 : !presburger.set<2,0>
  return %uset : !presburger.set<2,0>
}

// ----

// CHECK-LABEL: func @subtract_multiple_ineqs
func @subtract_multiple_ineqs() -> !presburger.set<2,0> {
  %set1 = presburger.set #presburger<"set(x, y) : (x + 1 >= 0)">
  %set2 = presburger.set #presburger<"set(x, y) : (x >= y and x <= 0)">

  // TODO: these constraints will be simplified as soon as simplification is available
  // CHECK-NEXT: %[[S:.*]] = presburger.set #presburger<"set(d0, d1) : (d0 + 1 >= 0 and -d0 + d1 - 1 >= 0 or d0 + 1 >= 0 and d0 - d1 >= 0 and d0 - 1 >= 0)">
  // CHECK-NEXT: return %[[S]]
  %uset = presburger.subtract %set1, %set2 : !presburger.set<2,0>
  return %uset : !presburger.set<2,0>
}

// ----

// CHECK-LABEL: func @subtract_non_convex_ineqs
func @subtract_non_convex_ineqs() -> !presburger.set<2,0> {
  %set1 = presburger.set #presburger<"set(x, y) : (x + 1 >= 0)">
  %set2 = presburger.set #presburger<"set(x, y) : (x >= y and x <= 0 or x >= 10)">

  // CHECK-NEXT: %[[S:.*]] = presburger.set #presburger<"set(d0, d1) : (d0 + 1 >= 0 and -d0 + d1 - 1 >= 0 and -d0 + 9 >= 0 or d0 + 1 >= 0 and d0 - d1 >= 0 and d0 - 1 >= 0 and -d0 + 9 >= 0)">
  // CHECK-NEXT: return %[[S]]
  %uset = presburger.subtract %set1, %set2 : !presburger.set<2,0>
  return %uset : !presburger.set<2,0>
}

// ----

// CHECK-LABEL: func @subtract_non_convex_ineqs2
func @subtract_non_convex_ineqs2() -> !presburger.set<2,0> {
  %set1 = presburger.set #presburger<"set(x, y) : (x + 1 >= 0 and y >= -5)">
  %set2 = presburger.set #presburger<"set(x, y) : (x >= y and x <= 0 or x >= 10)">

  // CHECK-NEXT: %[[S:.*]] = presburger.set #presburger<"set(d0, d1) : (d0 + 1 >= 0 and d1 + 5 >= 0 and -d0 + d1 - 1 >= 0 and -d0 + 9 >= 0 or d0 + 1 >= 0 and d1 + 5 >= 0 and d0 - d1 >= 0 and d0 - 1 >= 0 and -d0 + 9 >= 0)">
  // CHECK-NEXT: return %[[S]]
  %uset = presburger.subtract %set1, %set2 : !presburger.set<2,0>
  return %uset : !presburger.set<2,0>
}

// ----

// CHECK-LABEL: func @subtract_non_convex_ineqs3
func @subtract_non_convex_ineqs3() -> !presburger.set<2,0> {
  %set1 = presburger.set #presburger<"set(x, y) : (x + 1 >= 0 or y >= -5)">
  %set2 = presburger.set #presburger<"set(x, y) : (x >= y and x <= 0 or x >= 10)">

  // CHECK-NEXT: %[[S:.*]] = presburger.set #presburger<"set(d0, d1) : (d0 + 1 >= 0 and -d0 + d1 - 1 >= 0 and -d0 + 9 >= 0 or d0 + 1 >= 0 and d0 - d1 >= 0 and d0 - 1 >= 0 and -d0 + 9 >= 0 or d1 + 5 >= 0 and -d0 + d1 - 1 >= 0 and -d0 + 9 >= 0 or d1 + 5 >= 0 and d0 - d1 >= 0 and d0 - 1 >= 0 and -d0 + 9 >= 0)">
  // CHECK-NEXT: return %[[S]]
  %uset = presburger.subtract %set1, %set2 : !presburger.set<2,0>
  return %uset : !presburger.set<2,0>
}

// ---- 

// CHECK-LABEL: func @complement_simple
func @complement_simple() -> !presburger.set<1,0> {
  // CHECK-NEXT: %[[S:.*]] = presburger.set #presburger<"set(d0) : (-d0 - 1 >= 0)">
  // CHECK-NEXT: return %[[S]]
  %set = presburger.set #presburger<"set(x) : (x >= 0)">

  %uset = presburger.complement %set : !presburger.set<1,0>
  return %uset : !presburger.set<1,0>
}

// ----

// CHECK-LABEL: func @complement_empty
func @complement_empty() -> !presburger.set<1,0> {
  // TODO if we add (1 = 2) it breaks the next test, as this is not marked empty 
  // CHECK-NEXT: %[[S:.*]] = presburger.set #presburger<"set(d0) : ()">
  // CHECK-NEXT: return %[[S]]
  %set = presburger.set #presburger<"set(x) : (1 = 0)">

  %uset = presburger.complement %set : !presburger.set<1,0>
  return %uset : !presburger.set<1,0>
}

// ----

// CHECK-LABEL: func @complement_universe
func @complement_universe() -> !presburger.set<1,0> {
  // TODO check how empty is supposed to look
  // CHECK-NEXT: %[[S:.*]] = presburger.set #presburger<"set(d0) : (1 = 0)">
  // CHECK-NEXT: return %[[S]]
  %set = presburger.set #presburger<"set(x) : ()">

  %uset = presburger.complement %set : !presburger.set<1,0>
  return %uset : !presburger.set<1,0>
}

// ----

// CHECK-LABEL: func @complement_multi_dim
func @complement_multi_dim() -> !presburger.set<2,0> {
  // CHECK-NEXT: %[[S:.*]] = presburger.set #presburger<"set(d0, d1) : (-d0 - 1 >= 0 and d1 + 9 >= 0 or d0 >= 0 and -d1 - 1 >= 0 and d1 + 9 >= 0)">
  // CHECK-NEXT: return %[[S]]
  %set = presburger.set #presburger<"set(x,y) : (x >= 0 and y >= 0 or y <= -10)">

  %uset = presburger.complement %set : !presburger.set<2,0>
  return %uset : !presburger.set<2,0>
}

// equality
// ----

// CHECK-LABEL: func @equal_simple_pos
func @equal_simple_pos() -> i1 {
  // CHECK-NEXT: %[[S:.*]] = constant true
  // CHECK-NEXT: return %[[S]]
  %set1 = presburger.set #presburger<"set(x) : (x >= 0)">
  %set2 = presburger.set #presburger<"set(x) : (x >= 0)">

  %r = presburger.equal %set1, %set2 : !presburger.set<1,0>, !presburger.set<1,0>
  return %r : i1
}

// ----

// CHECK-LABEL: func @equal_simple_neg
func @equal_simple_neg() -> i1 {
  // CHECK-NEXT: %[[S:.*]] = constant false
  // CHECK-NEXT: return %[[S]]
  %set1 = presburger.set #presburger<"set(x) : (x >= 0)">
  %set2 = presburger.set #presburger<"set(x) : (x <= 0)">

  %r = presburger.equal %set1, %set2 : !presburger.set<1,0>, !presburger.set<1,0>
  return %r : i1
}

// ----

// CHECK-LABEL: func @equal_multidim_pos
func @equal_multidim_pos() -> i1 {
  // CHECK-NEXT: %[[S:.*]] = constant true
  // CHECK-NEXT: return %[[S]]
  %set1 = presburger.set #presburger<"set(x,y) : (x >= 0 and x + y = 0 or x = 4 and y = 2)">
  %set2 = presburger.set #presburger<"set(x,y) : (y <= 0 and x + y = 0 or x = 4 and y = 2)">

  %r = presburger.equal %set1, %set2 : !presburger.set<2,0>, !presburger.set<2,0>
  return %r : i1
}

// ----

// CHECK-LABEL: func @equal_multidim_neg
func @equal_multidim_neg() -> i1 {
  // CHECK-NEXT: %[[S:.*]] = constant false
  // CHECK-NEXT: return %[[S]]
  %set1 = presburger.set #presburger<"set(x,y) : (x >= 0 and x + y = 0 or x = 4 and y = 2)">
  %set2 = presburger.set #presburger<"set(x,y) : (y <= 1 and x + y = 0 or x = 4 and y = 2)">

  %r = presburger.equal %set1, %set2 : !presburger.set<2,0>, !presburger.set<2,0>
  return %r : i1
}
