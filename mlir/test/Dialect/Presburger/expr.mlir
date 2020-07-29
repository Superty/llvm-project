// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

func @pw() {

  // CHECK: %{{.*}} = presburger.expr #presburger<"expr(d0) -> (d0 + 1) : (d0 >= 0)">
  %expr = presburger.expr #presburger<"expr (x) -> (x + 1) : (x >= 0)">

  // CHECK: %{{.*}} = presburger.expr #presburger<"expr(d0) -> (d0 + 1) : (d0 >= 0) ; (0) : (-d0 + 1 >= 0)">
  %expr1 = presburger.expr #presburger<"expr (x) -> (x + 1) : (x >= 0) ; (0) : (-x + 1 >= 0)">

  // CHECK: %{{.*}} = presburger.expr #presburger<"expr(d0, d1) -> (d0) : (d0 >= 0 and 2d1 >= 0) ; (d1) : (-d0 + 1 >= 0)">
  %expr2 = presburger.expr #presburger<"expr (x,y) -> (x) : (x >= 0 and 2y >= 0) ; (y) : (-x + 1 >= 0)">
  return
}

