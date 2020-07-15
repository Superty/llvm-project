// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

func @simple_coalesce() {

  // CHECK: %[[S1:.*]] = presburger.set #presburger<"{{.*}}">
  %set1 = presburger.set #presburger<"(x)[] : (x >= 0 and -x + 4>= 0)">

  // CHECK: %{{.*}} = presburger.coalesce %[[S1]] : !presburger.set<1,0>
  %uset = presburger.coalesce %set1 : !presburger.set<1,0>
  return
}
