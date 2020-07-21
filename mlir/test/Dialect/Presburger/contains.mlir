// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

func @simple_contains() {

  // CHECK: %[[S1:.*]] = presburger.set #presburger<"{{.*}}">
  %set1 = presburger.set #presburger<"set(x)[] : (x >= 0 and -x + 4 >= 0)">

  // CHECK: %[[D1:.*]] = constant 0 : index
  %d1 = constant 0 : index

  // CHECK: %{{.*}} = presburger.contains (%[[D1]]) %[[S1]]
  %c1 = presburger.contains (%d1)[] %set1 

  // CHECK: %{{.*}} = presburger.contains (%[[D1]]) %[[S1]]
  %c2 = presburger.contains (%d1) %set1 
  return
}

// -----

func @simple_contains_with_syms() {

  // CHECK: %[[S1:.*]] = presburger.set #presburger<"{{.*}}">
  %set1 = presburger.set #presburger<"set(x)[s] : (x >= 0 and -x + 4 >= s)">

  // CHECK: %[[D1:.*]] = constant 0 : index
  %d1 = constant 0 : index
  // CHECK: %[[P1:.*]] = constant 1 : index
  %s1 = constant 1 : index

  // CHECK: %{{.*}} = presburger.contains (%[[D1]])[%[[P1]]] %[[S1]]
  %uset = presburger.contains (%d1)[%s1] %set1 
  return
}
