// RUN: presburgerc %s --mlir-print-op-generic | presburgerc | FileCheck %s

func @simple_union() {

  // CHECK: %{{.*}} = presburger.set #presburger<"(d0)[] : ()">
  %set0 = presburger.set #presburger<"(x)[] : ()">

  // CHECK: %{{.*}} = presburger.set #presburger<"(d0)[] : (d0 >= 0)">
  %set1 = presburger.set #presburger<"(x)[] : (x >= 0)">

  // CHECK: %{{.*}} = presburger.set #presburger<"(d0)[] : (d0 = 0)">
  %set2 = presburger.set #presburger<"(y)[] : (y = 0)">

  // CHECK: %{{.*}} = presburger.set #presburger<"(d0)[] : (d0 >= 0 and -d0 + 4 >= 0 or d0 - 10 >= 0 and -d0 + 14 >= 0)">
  %set3 = presburger.set #presburger<"(x)[] : (x >= 0 and -x + 4 >= 0 or x >= 10 and -x + 14 >= 0)">

  // CHECK: %{{.*}} = presburger.set #presburger<"(d0)[] : (-d0 + 4 = 0 and d0 >= 0)">
  %set4 = presburger.set #presburger<"(x)[] : (x >= 0 and -x + 4 = 0)">

  // CHECK: %{{.*}} = presburger.set #presburger<"(d0, d1)[] : (d0 >= 0 and d0 + d1 - 10 >= 0)">
  %set5 = presburger.set #presburger<"(x,y)[] : (x >= 0 and x + y - 10 >= 0)">

  // CHECK: %{{.*}} = presburger.set #presburger<"(d0)[s0] : (d0 - s0 >= 0)">
  %set6 = presburger.set #presburger<"(y)[N] : (y >= N)">

  // CHECK: %{{.*}} = presburger.set #presburger<"(d0)[s0, s1] : (d0 - s0 + 1 >= 0 and d0 - s0 + 42s1 + 10 >= 0)">
  %set7 = presburger.set #presburger<"(y)[N,M] : (y >= N - 1 and y + 10 - N + 42M >= 0)">

  return
}

