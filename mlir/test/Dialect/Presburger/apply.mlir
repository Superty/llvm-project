// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

func @simple_apply() {

  // CHECK: %[[S1:.*]] = presburger.expr #presburger<"{{.*}}">
  %expr1 = presburger.expr #presburger<"expr(x)[] -> (x) : ()">

  // CHECK: %[[D1:.*]] = constant 0 : index
  %d1 = constant 0 : index

  // CHECK: %{{.*}} = presburger.apply (%[[D1]]) %[[S1]]
  %c1 = presburger.apply (%d1)[] %expr1 

  return
}

// -----

func @simple_apply_with_syms() {

  // CHECK: %[[S1:.*]] = presburger.expr #presburger<"{{.*}}">
  %set1 = presburger.expr #presburger<"expr(x)[s] -> (x + s) : (x >= s) ; (x) : (x - 1 <= s)">

  // CHECK: %[[D1:.*]] = constant 0 : index
  %d1 = constant 0 : index
  // CHECK: %[[P1:.*]] = constant 1 : index
  %s1 = constant 1 : index

  // CHECK: %{{.*}} = presburger.apply (%[[D1]])[%[[P1]]] %[[S1]]
  %uset = presburger.apply (%d1)[%s1] %set1 
  return
}
