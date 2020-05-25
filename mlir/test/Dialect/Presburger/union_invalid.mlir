// RUN: presburgerc %s -split-input-file -verify-diagnostics

func @union_should_match_op_dims() {
  %set1 = presburger.set #presburger<"(x,y)[] : (y >= 0 and -x + 4>= 0)">
  %set2 = presburger.set #presburger<"(x)[] : (x - 1 >= 0 and -x + 3 >= 0)">

  // NOTE: can only be checked in the generic form
  // expected-error @+1 {{'presburger.union' op requires the same type for all operands and results}}
  %uset = "presburger.union"(%set1, %set2)
    : (!presburger.set<2,0>, !presburger.set<1,0>) -> !presburger.set<1,0>
}

// -----

func @union_should_match_op_res_dims() {
  %set1 = presburger.set #presburger<"(x)[] : (x >= 0 and -x + 4>= 0)">

  %set2 = presburger.set #presburger<"(x)[] : (x - 1 >= 0 and -x + 3 >= 0)">

  // NOTE: can only be checked in the generic form
  // expected-error @+1 {{'presburger.union' op requires the same type for all operands and results}}
  %uset = "presburger.union"(%set1, %set2)
    : (!presburger.set<1,0>, !presburger.set<1,0>) -> !presburger.set<2,0>
}
