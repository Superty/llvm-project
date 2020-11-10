// RUN: mlir-opt %s -canonicalize | FileCheck %s

//===----------------------------------------------------------------------===//
// union simplifications
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @union
func @union() -> !presburger.set<1,1> {
  %set1 = presburger.set #presburger<"set(x)[s] : (x >= 0) or (x - s = 42)">

  // CHECK-NEXT: %[[S:.*]] = presburger.set #presburger<"set(d0)[s0] : (d0 >= 0) or (d0 - s0 - 42 = 0)">
  // CHECK-NEXT: return %[[S]]
  %res = presburger.union %set1, %set1 : !presburger.set<1,1>

  return %res : !presburger.set<1,1>
}

// CHECK-LABEL: func @union_no_opt
func @union_no_opt() -> !presburger.set<1,1> {
  %set1 = presburger.set #presburger<"set(x)[s] : (x >= 0) or (x - s = 42)">
  %set2 = presburger.set #presburger<"set(x)[s] : (x >= 0) or (x + s = 42)">

  // CHECK-NEXT: %[[S1:.*]] = presburger.set #presburger<"{{.*}}">
  // CHECK-NEXT: %[[S2:.*]] = presburger.set #presburger<"{{.*}}">
  // CHECK-NEXT: %[[R:.*]] = presburger.union %[[S1]], %[[S2]]
  // CHECK-NEXT: return %[[R]]
  %res = presburger.union %set1, %set2 : !presburger.set<1,1>

  return %res : !presburger.set<1,1>
}

//===----------------------------------------------------------------------===//
// intersect simplifications
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @intersect
func @intersect() -> !presburger.set<1,0> {
  %set1 = presburger.set #presburger<"set(x)[] : (x >= 0 and x <= 42)">

  // CHECK-NEXT: %[[S:.*]] = presburger.set #presburger<"set(d0) : (d0 >= 0 and -d0 + 42 >= 0)">
  // CHECK-NEXT: return %[[S]]
  %res = presburger.intersect %set1, %set1 : !presburger.set<1,0>

  return %res : !presburger.set<1,0>
}

// CHECK-LABEL: func @intersect_no_opt
func @intersect_no_opt() -> !presburger.set<1,1> {
  %set1 = presburger.set #presburger<"set(x)[s] : (x >= 0) or (x - s = 42)">
  %set2 = presburger.set #presburger<"set(x)[s] : (x >= 0) or (x + s = 42)">

  // CHECK-NEXT: %[[S1:.*]] = presburger.set #presburger<"{{.*}}">
  // CHECK-NEXT: %[[S2:.*]] = presburger.set #presburger<"{{.*}}">
  // CHECK-NEXT: %[[R:.*]] = presburger.intersect %[[S1]], %[[S2]]
  // CHECK-NEXT: return %[[R]]
  %res = presburger.intersect %set1, %set2 : !presburger.set<1,1>

  return %res : !presburger.set<1,1>
}

//===----------------------------------------------------------------------===//
// subtract simplifications
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @subtract
func @subtract() -> !presburger.set<1,0> {
  %set1 = presburger.set #presburger<"set(x)[] : (x >= 0 and x <= 42)">

  // CHECK-NEXT: %[[S:.*]] = presburger.set #presburger<"set(d0) : empty">
  // CHECK-NEXT: return %[[S]]
  %res = presburger.subtract %set1, %set1 : !presburger.set<1,0>

  return %res : !presburger.set<1,0>
}

// CHECK-LABEL: func @subtract_no_opt
func @subtract_no_opt() -> !presburger.set<1,1> {
  %set1 = presburger.set #presburger<"set(x)[s] : (x >= 0) or (x - s = 42)">
  %set2 = presburger.set #presburger<"set(x)[s] : (x >= 0) or (x + s = 42)">

  // CHECK-NEXT: %[[S1:.*]] = presburger.set #presburger<"{{.*}}">
  // CHECK-NEXT: %[[S2:.*]] = presburger.set #presburger<"{{.*}}">
  // CHECK-NEXT: %[[R:.*]] = presburger.subtract %[[S1]], %[[S2]]
  // CHECK-NEXT: return %[[R]]
  %res = presburger.subtract %set1, %set2 : !presburger.set<1,1>

  return %res : !presburger.set<1,1>
}

//===----------------------------------------------------------------------===//
// equal simplifications
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @equal
func @equal() -> i1 {
  %set1 = presburger.set #presburger<"set(x)[s] : (x >= 0) or (x - s = 42)">

  // CHECK-NEXT: %[[R:.*]] = constant true
  // CHECK-NEXT: return %[[R]]
  %res = presburger.equal %set1, %set1 : !presburger.set<1,1>

  return %res : i1
}

// CHECK-LABEL: func @equal_no_opt1
func @equal_no_opt1() -> i1 {
  %set1 = presburger.set #presburger<"set(x)[s] : (x >= 0) or (x - s = 42)">
  %set2 = presburger.set #presburger<"set(x)[s] : (x >= 0) or (x - 1 = 42)">

  // CHECK-NEXT: %[[S1:.*]] = presburger.set #presburger<"{{.*}}">
  // CHECK-NEXT: %[[S2:.*]] = presburger.set #presburger<"{{.*}}">
  // CHECK-NEXT: %[[R:.*]] = presburger.equal %[[S1]], %[[S2]]
  // CHECK-NEXT: return %[[R]]
  %res = presburger.equal %set1, %set2 : !presburger.set<1,1>

  return %res : i1
}

// This pass should not perform any nontrivial computations
// CHECK-LABEL: func @equal_no_opt2
func @equal_no_opt2() -> i1 {
  %set1 = presburger.set #presburger<"set(x) : (x >= 0 and x <= 42)">
  %set2 = presburger.set #presburger<"set(x) : (x >= 0 and x <= 41) or (x = 42)">

  // CHECK-NEXT: %[[S1:.*]] = presburger.set #presburger<"{{.*}}">
  // CHECK-NEXT: %[[S2:.*]] = presburger.set #presburger<"{{.*}}">
  // CHECK-NEXT: %[[R:.*]] = presburger.equal %[[S1]], %[[S2]]
  // CHECK-NEXT: return %[[R]]
  %res = presburger.equal %set1, %set2 : !presburger.set<1,0>

  return %res : i1
}

//===----------------------------------------------------------------------===//
// complement simplifications
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @single_complement
func @single_complement() -> !presburger.set<1,0> {
  %set1 = presburger.set #presburger<"set(x)[] : (x >= 0 and x <= 42)">

  // CHECK-NEXT: %[[S:.*]] = presburger.set #presburger<"{{.*}}">
  // CHECK-NEXT: %[[R:.*]] = presburger.complement %[[S]]
  // CHECK-NEXT: return %[[R]]
  %setC = presburger.complement %set1 : !presburger.set<1,0>

  return %setC : !presburger.set<1,0>
}

// CHECK-LABEL: func @double_complement
func @double_complement() -> !presburger.set<1,0> {
  %set1 = presburger.set #presburger<"set(x)[] : (x >= 0 and x <= 42)">

  // CHECK-NEXT: %[[S:.*]] = presburger.set #presburger<"{{.*}}">
  // CHECK-NEXT: return %[[S]]
  %setC = presburger.complement %set1 : !presburger.set<1,0>
  %setCC = presburger.complement %setC : !presburger.set<1,0>

  return %setCC : !presburger.set<1,0>
}

// CHECK-LABEL: func @triple_complement
func @triple_complement() -> !presburger.set<1,0> {
  %set1 = presburger.set #presburger<"set(x)[] : (x >= 0 and x <= 42)">

  // CHECK-NEXT: %[[S:.*]] = presburger.set #presburger<"{{.*}}">
  // CHECK-NEXT: %[[R:.*]] = presburger.complement %[[S]]
  // CHECK-NEXT: return %[[R]]
  %setC = presburger.complement %set1 : !presburger.set<1,0>
  %setCC = presburger.complement %setC : !presburger.set<1,0>
  %setCCC = presburger.complement %setCC : !presburger.set<1,0>

  return %setCCC : !presburger.set<1,0>
}
