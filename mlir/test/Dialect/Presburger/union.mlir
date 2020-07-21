// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

func @simple_union() {

  // CHECK: %[[S1:.*]] = presburger.set #presburger<"{{.*}}">
  %set1 = presburger.set #presburger<"set(x)[] : (x >= 0 and -x + 4>= 0)">

  // CHECK: %[[S2:.*]] = presburger.set #presburger<"{{.*}}">
  %set2 = presburger.set #presburger<"set(x)[] : (x - 1 >= 0 and -x + 3 >= 0)">

  // CHECK: %{{.*}} = presburger.union %[[S1]], %[[S2]] : !presburger.set<1,0>
  %uset = presburger.union %set1, %set2 : !presburger.set<1,0>
  return
}

// -----

func @union_with_syms() {

  // CHECK: %[[S1:.*]] = presburger.set #presburger<"{{.*}}">
  %set1 = presburger.set #presburger<"set(x)[N] : (x >= 0 and -x + 2N>= 0)">

  // CHECK: %[[S2:.*]] = presburger.set #presburger<"{{.*}}">
  %set2 = presburger.set #presburger<"set(x)[N] : (x - N >= 0 and -x + 3 >= 0)">

  // CHECK: %{{.*}} = presburger.union %[[S1]], %[[S2]] : !presburger.set<1,1>
  %uset = presburger.union %set1, %set2 : !presburger.set<1,1>
  return
}
