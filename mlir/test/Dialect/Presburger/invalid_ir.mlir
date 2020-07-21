// RUN: mlir-opt %s -split-input-file -verify-diagnostics

//===--------------------------------------------------------------------------
// Invalid sets
//===--------------------------------------------------------------------------

func @undeclared_dim_var() {
  // expected-error @+1 {{encountered unknown variable name: x}}
  %set1 = presburger.set #presburger<"set(y)[] : (x >= 0)">
}

// -----

func @undeclared_sym_var() {
  // expected-error @+1 {{encountered unknown variable name: M}}
  %set1 = presburger.set #presburger<"set(y)[N] : (y >= M)">
}

// -----

func @wrong_brackets() {
  // expected-error @+1 {{expected '('}}
  %set1 = presburger.set #presburger<"set(y)[N] : {x >= M}">
}

// -----

func @missing_bracket() {
  // expected-error @+1 {{expected ',' or ']'}}
  %set1 = presburger.set #presburger<"set(y)[N : (y <= N)">
}

// -----

func @end_of_set() {
  // expected-error @+1 {{expected to be at the end of the set}}
  %set1 = presburger.set #presburger<"set(y)[N] : (y <= N) (y = 0)">
}

// -----

func @end_of_empty_set() {
  // expected-error @+1 {{expected to be at the end of the set}}
  %set1 = presburger.set #presburger<"set(y)[N] : () a ">
}

// -----

func @empty_dims() {
  // expected-error @+1 {{expected non empty list}}
  %set1 = presburger.set #presburger<"set()[N] : (N = 0)">
}

// -----

func @no_set_definition() {
  // expected-error @+1 {{expected ':' but got}}
  %set1 = presburger.set #presburger<"set(x)[]">
}

// -----

func @no_set_definition() {
  // expected-error @+1 {{expected ')'}}
  %set1 = presburger.set #presburger<"set(x,y)[] : (x + y = 0 = x + y)">
}

// -----

func @no_set_definition() {
  // expected-error @+1 {{expected a valid 64 bit integer}}
  %set = presburger.set #presburger<"set(d0)[] : (d0 >= 9223372036854775808)">
}

// -----

//===--------------------------------------------------------------------------
// non local operands 
//===--------------------------------------------------------------------------

func @union_non_local_set(%set1 : !presburger.set<1,0>) {
  %set2 = presburger.set #presburger<"set(x)[] : (x - 1 >= 0 and -x + 3 >= 0)">

  // expected-error @+1 {{expect local set definitions}}
  %uset = presburger.union %set1, %set2 : !presburger.set<1,0>
}

// -----

func @intersect_non_local_sets(%set1 : !presburger.set<1,0>) {
  %set2 = presburger.set #presburger<"set(x)[] : (x - 1 >= 0 and -x + 3 >= 0)">

  // expected-error @+1 {{expect local set definitions}}
  %uset = presburger.intersect %set1, %set2 : !presburger.set<1,0>
}

// -----

func @subtract_non_local_sets(%set1 : !presburger.set<1,0>) {
  %set2 = presburger.set #presburger<"set(x)[] : (x - 1 >= 0 and -x + 3 >= 0)">

  // expected-error @+1 {{expect local set definitions}}
  %uset = presburger.subtract %set1, %set2 : !presburger.set<1,0>
}

// -----

func @equal_non_local_sets(%set1 : !presburger.set<1,0>) {
  %set2 = presburger.set #presburger<"set(x)[] : (x - 1 >= 0 and -x + 3 >= 0)">

  // expected-error @+1 {{expect local set definitions}}
  %uset = presburger.equal %set1, %set2 : !presburger.set<1,0>, !presburger.set<1,0>
}

// -----

func @complement_non_local_sets(%set : !presburger.set<1,0>) {

  // expected-error @+1 {{expect local set definitions}}
  %uset = presburger.complement %set : !presburger.set<1,0>
}

// -----

func @f() -> !presburger.set<1,0>

func @subtract_non_local_sets() {
  %set1 = call @f() : () -> !presburger.set<1,0>
  %set2 = presburger.set #presburger<"set(x)[] : (x - 1 >= 0 and -x + 3 >= 0)">

  // expected-error @+1 {{expect operand to have trait ProducesPresburgerSet}}
  %uset = presburger.subtract %set1, %set2 : !presburger.set<1,0>
  return
}
