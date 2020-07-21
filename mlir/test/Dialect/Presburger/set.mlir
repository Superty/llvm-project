// RUN: mlir-opt %s --mlir-print-op-generic | mlir-opt | FileCheck %s

func @simple_union() {

  // CHECK: %{{.*}} = presburger.set #presburger<"set(d0)[] : ()">
  %set0 = presburger.set #presburger<"set(x)[] : ()">

  // CHECK: %{{.*}} = presburger.set #presburger<"set(d0)[] : (d0 >= 0)">
  %set1 = presburger.set #presburger<"set(x)[] : (x >= 0)">

  // CHECK: %{{.*}} = presburger.set #presburger<"set(d0)[] : (d0 = 0)">
  %set2 = presburger.set #presburger<"set(y)[] : (y = 0)">

  // CHECK: %{{.*}} = presburger.set #presburger<"set(d0)[] : (d0 >= 0 and -d0 + 4 >= 0 or d0 - 10 >= 0 and -d0 + 14 >= 0)">
  %set3 = presburger.set #presburger<"set(x)[] : (x >= 0 and -x + 4 >= 0 or x >= 10 and -x + 14 >= 0)">

  // CHECK: %{{.*}} = presburger.set #presburger<"set(d0)[] : (-d0 + 4 = 0 and d0 >= 0)">
  %set4 = presburger.set #presburger<"set(x)[] : (x >= 0 and -x + 4 = 0)">

  // CHECK: %{{.*}} = presburger.set #presburger<"set(d0, d1)[] : (d0 >= 0 and d0 + d1 - 10 >= 0)">
  %set5 = presburger.set #presburger<"set(x,y)[] : (x >= 0 and x + y - 10 >= 0)">

  // CHECK: %{{.*}} = presburger.set #presburger<"set(d0)[s0] : (d0 - s0 >= 0)">
  %set6 = presburger.set #presburger<"set(y)[N] : (y >= N)">

  // CHECK: %{{.*}} = presburger.set #presburger<"set(d0)[s0, s1] : (d0 - s0 + 1 >= 0 and d0 - s0 + 42s1 + 10 >= 0)">
  %set7 = presburger.set #presburger<"set(y)[N,M] : (y >= N - 1 and y + 10 - N + 42M >= 0)">

  // CHECK: %{{.*}} = presburger.set #presburger<"set(d0, d1)[] : (-d0 - 2147483650 >= 0 or d0 - 2499 >= 0 or d0 >= 0 and -d0 + 2498 >= 0 and d1 >= 0)">

  %set8 = presburger.set #presburger<"set(d0, d1)[] : (-d0 + -2147483650 >= 0  or d0 + -2499 >= 0  or d0  >= 0 and -d0 + 2498 >= 0 and d1  >= 0 )">

  return
}

