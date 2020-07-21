// RUN: mlir-opt %s | mlir-opt | FileCheck %s

func @simple_intersect() {

  // CHECK: %[[S1:.*]] = presburger.set #presburger<"{{.*}}">
  %set1 = presburger.set #presburger<"set(x)[] : (x >= 0 and -x + 4>= 0)">

  // CHECK: %[[S2:.*]] = presburger.set #presburger<"{{.*}}">
  %set2 = presburger.set #presburger<"set(x)[] : (x - 1 >= 0 and -x + 3 >= 0)">

  // CHECK: %{{.*}} = presburger.intersect %[[S1]], %[[S2]] : !presburger.set<1,0>
  %uset = presburger.intersect %set1, %set2 : !presburger.set<1,0>
  return
}

// -----

func @intersect_with_syms() {

  // CHECK: %[[S1:.*]] = presburger.set #presburger<"{{.*}}">
  %set1 = presburger.set #presburger<"set(x)[N] : (x >= 0 and -x + N>= 0)">

  // CHECK: %[[S2:.*]] = presburger.set #presburger<"{{.*}}">
  %set2 = presburger.set #presburger<"set(x)[N] : (x - N >= 0 and -x + 3 >= 0)">

  // CHECK: %{{.*}} = presburger.intersect %[[S1]], %[[S2]] : !presburger.set<1,1>
  %uset = presburger.intersect %set1, %set2 : !presburger.set<1,1>
  return
}
