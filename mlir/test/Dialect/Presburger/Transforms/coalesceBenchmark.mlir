// RUN: mlir-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: func @contained
func @contained() -> i1 {
  // CHECK-NEXT: %[[S:.*]] = constant true
  // CHECK-NEXT: return %[[S]]
  %set = presburger.set #presburger<"(x)[] : (x >= 0 and x <= 1 or x >= 2 and x <= 3)">

  %r = presburger.coalesce %set : !presburger.set<1,0>
  %e = presburger.equal %set, %r : !presburger.set<1,0>, !presburger.set<1,0>
  return %e : i1
}

// ----
