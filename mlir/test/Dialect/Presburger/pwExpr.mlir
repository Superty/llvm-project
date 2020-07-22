// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

func @pw() {

  // CHECK: %{{.*}} = presburger.pwExpr #presburger<"pwExpr(d0)[] -> (d0 + 1) : (d0 >= 0)">
  %expr = presburger.pwExpr #presburger<"pwExpr (x)[] -> (x + 1) : (x >= 0)">

  // CHECK: %{{.*}} = presburger.pwExpr #presburger<"pwExpr(d0)[] -> (d0 + 1) : (d0 >= 0) ; (0) : (-d0 + 1 >= 0)">
  %expr1 = presburger.pwExpr #presburger<"pwExpr (x)[] -> (x + 1) : (x >= 0) ; (0) : (-x + 1 >= 0)">

  // CHECK: %{{.*}} = presburger.pwExpr #presburger<"pwExpr(d0, d1)[] -> (d0) : (d0 >= 0 and 2d1 >= 0) ; (d1) : (-d0 + 1 >= 0)">
  %expr2 = presburger.pwExpr #presburger<"pwExpr (x,y)[] -> (x) : (x >= 0 and 2y >= 0) ; (y) : (-x + 1 >= 0)">
  return
}

