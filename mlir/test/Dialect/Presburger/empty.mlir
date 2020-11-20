// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

func @simple_coalesce() {

  // CHECK: %[[S1:.*]] = presburger.set #presburger<"{{.*}}">
  %set1 = presburger.set #presburger<"set(d0, d1, d2)[p0, p1, p2, p3] : (exists e0 : -d0 + 8192e0  + 8191 >= 0 and 32p2 + -d0 + 8192e0  + 31 >= 0 and d0 + -8192e0  >= 0 and -32p2 + d0 + -8192e0  >= 0)">

  // CHECK: %{{.*}} = presburger.is_empty %[[S1]] : !presburger.set<3,4>
  %uset = presburger.is_empty %set1 : !presburger.set<3,4>
  return
}
