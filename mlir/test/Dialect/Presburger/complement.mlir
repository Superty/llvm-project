// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

func @simple_complement() {

  // CHECK: %[[S1:.*]] = presburger.set #presburger<"{{.*}}">
  %set1 = presburger.set #presburger<"set(x)[] : (x >= 0 and -x + 4>= 0)">

  // CHECK: %{{.*}} = presburger.complement %[[S1]] : !presburger.set<1,0>
  %uset = presburger.complement %set1 : !presburger.set<1,0>
  return
}

// -----

func @complement_with_syms() {

  // CHECK: %[[S1:.*]] = presburger.set #presburger<"{{.*}}">
  %set1 = presburger.set #presburger<"set(x)[N] : (x >= 0 and -x + 2N>= 0)">

  // CHECK: %{{.*}} = presburger.complement %[[S1]] : !presburger.set<1,1>
  %uset = presburger.complement %set1 : !presburger.set<1,1>
  return
}
