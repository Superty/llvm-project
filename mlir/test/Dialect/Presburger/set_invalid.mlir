// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func @undeclared_dim_var() {
  // expected-error @+1 {{encountered unknown variable name: x}}
  %set1 = presburger.set #presburger<"(y)[] : (x >= 0)">
}

// -----

func @undeclared_sym_var() {
  // expected-error @+1 {{encountered unknown variable name: M}}
  %set1 = presburger.set #presburger<"(y)[N] : (y >= M)">
}

// -----

func @wrong_brackets() {
  // expected-error @+1 {{expected '('}}
  %set1 = presburger.set #presburger<"(y)[N] : {x >= M}">
}

// -----

func @missing_bracket() {
  // expected-error @+1 {{expected ',' or ']'}}
  %set1 = presburger.set #presburger<"(y)[N : (y <= N)">
}

// -----

func @end_of_set() {
  // expected-error @+1 {{expected to be at the end of the set}}
  %set1 = presburger.set #presburger<"(y)[N] : (y <= N) (y = 0)">
}

// -----

func @end_of_empty_set() {
  // expected-error @+1 {{expected to be at the end of the set}}
  %set1 = presburger.set #presburger<"(y)[N] : () a ">
}

// -----

func @empty_dims() {
  // expected-error @+1 {{expected non empty list}}
  %set1 = presburger.set #presburger<"()[N] : (N = 0)">
}

// -----

func @no_set_definition() {
  // expected-error @+1 {{expected ':' but got}}
  %set1 = presburger.set #presburger<"(x)[]">
}

// -----

func @no_set_definition() {
  // expected-error @+1 {{expected ')'}}
  %set1 = presburger.set #presburger<"(x,y)[] : (x + y = 0 = x + y)">
}
