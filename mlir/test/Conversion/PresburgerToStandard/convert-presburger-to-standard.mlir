// RUN: mlir-opt --convert-presburger-to-std %s | FileCheck %s

//-----------------------------------------------------------------------------
// presburger.contains
//-----------------------------------------------------------------------------

// CHECK-LABEL: func @contains_universe(%arg0: index) -> i1 {
// CHECK-NEXT:    {{.*}} = presburger.set #presburger<"{{.*}}">
// CHECK-NEXT:    %true = constant true
// CHECK-NEXT:    return %true : i1
// CHECK-NEXT:  }
func @contains_universe(%x0 : index) -> i1 {
  %set1 = presburger.set #presburger<"set(x)[] : ()">

  %c = presburger.contains (%x0)[] %set1
  return %c : i1
}

// -----

// CHECK-LABEL: func @contains_runtime
// CHECK-NEXT:   {{.*}} = presburger.set #presburger<"{{.*}}">
// CHECK-NEXT:   %false = constant false
// CHECK-NEXT:   %true = constant true
// CHECK-NEXT:   %c0 = constant 0 : index
// CHECK-NEXT:   %c1 = constant 1 : index
// CHECK-NEXT:   %1 = muli %c1, %arg0 : index
// CHECK-NEXT:   %2 = addi %c0, %1 : index
// CHECK-NEXT:   %c1_0 = constant 1 : index
// CHECK-NEXT:   %3 = muli %c1_0, %arg1 : index
// CHECK-NEXT:   %4 = addi %2, %3 : index
// CHECK-NEXT:   %c0_1 = constant 0 : index
// CHECK-NEXT:   %5 = muli %c0_1, %arg2 : index
// CHECK-NEXT:   %6 = addi %4, %5 : index
// CHECK-NEXT:   %c0_2 = constant 0 : index
// CHECK-NEXT:   %7 = cmpi "sge", %6, %c0_2 : index
// CHECK-NEXT:   %8 = and %true, %7 : i1
// CHECK-NEXT:   %c4 = constant 4 : index
// CHECK-NEXT:   %c-1 = constant -1 : index
// CHECK-NEXT:   %9 = muli %c-1, %arg0 : index
// CHECK-NEXT:   %10 = addi %c4, %9 : index
// CHECK-NEXT:   %c0_3 = constant 0 : index
// CHECK-NEXT:   %11 = muli %c0_3, %arg1 : index
// CHECK-NEXT:   %12 = addi %10, %11 : index
// CHECK-NEXT:   %c-1_4 = constant -1 : index
// CHECK-NEXT:   %13 = muli %c-1_4, %arg2 : index
// CHECK-NEXT:   %14 = addi %12, %13 : index
// CHECK-NEXT:   %c0_5 = constant 0 : index
// CHECK-NEXT:   %15 = cmpi "sge", %14, %c0_5 : index
// CHECK-NEXT:   %16 = and %8, %15 : i1
// CHECK-NEXT:   %c0_6 = constant 0 : index
// CHECK-NEXT:   %c1_7 = constant 1 : index
// CHECK-NEXT:   %17 = muli %c1_7, %arg0 : index
// CHECK-NEXT:   %18 = addi %c0_6, %17 : index
// CHECK-NEXT:   %c-1_8 = constant -1 : index
// CHECK-NEXT:   %19 = muli %c-1_8, %arg1 : index
// CHECK-NEXT:   %20 = addi %18, %19 : index
// CHECK-NEXT:   %c0_9 = constant 0 : index
// CHECK-NEXT:   %21 = muli %c0_9, %arg2 : index
// CHECK-NEXT:   %22 = addi %20, %21 : index
// CHECK-NEXT:   %c0_10 = constant 0 : index
// CHECK-NEXT:   %23 = cmpi "sge", %22, %c0_10 : index
// CHECK-NEXT:   %24 = and %16, %23 : i1
// CHECK-NEXT:   %25 = or %false, %24 : i1
// CHECK-NEXT:   return %25 : i1
// CHECK-NEXT: }
func @contains_runtime(%x0 : index, %x1 : index, %s0 : index) -> i1 {
  %set1 = presburger.set #presburger<"set(x, y)[s] : (x + y >= 0 and -x + 4 >= s and x - y >= 0)">

  %c = presburger.contains (%x0, %x1)[%s0] %set1
  return %c : i1
}

//-----------------------------------------------------------------------------
// presburger.apply
//-----------------------------------------------------------------------------

// TODO: these tests might be too specific

// CHECK-LABEL: func @simple_apply(%arg0: index) -> index {
// CHECK-NEXT:    %{{.*}} = presburger.expr #presburger<"{{.*}}">
// CHECK-NEXT:    br ^bb2
// CHECK-NEXT:  ^bb1(%1: index):  
// CHECK-NEXT:    return %1 : index
// CHECK-NEXT:  ^bb2:  
// CHECK-NEXT:    %true = constant true
// CHECK-NEXT:    %c0 = constant 0 : index
// CHECK-NEXT:    cond_br %true, ^bb3, ^bb1(%c0 : index)
// CHECK-NEXT:  ^bb3:  
// CHECK-NEXT:    %c0_0 = constant 0 : index
// CHECK-NEXT:    %c1 = constant 1 : index
// CHECK-NEXT:    %2 = muli %c1, %arg0 : index
// CHECK-NEXT:    %3 = addi %c0_0, %2 : index
// CHECK-NEXT:    br ^bb1(%3 : index)
// CHECK-NEXT:  }
func @simple_apply(%d1 : index) -> index {

  %expr1 = presburger.expr #presburger<"expr(x)[] -> (x) : ()">
  %c1 = presburger.apply (%d1)[] %expr1 

  return %c1 : index
}


// CHECK-LABEL: func @expression_apply(%arg0: index) -> index {
// CHECK-NEXT:    %0 = presburger.expr #presburger<"{{.*}}">
// CHECK-NEXT:    br ^bb2
// CHECK-NEXT:  ^bb1(%1: index):  
// CHECK-NEXT:    return %1 : index
// CHECK-NEXT:  ^bb2:  
// CHECK-NEXT:    %true = constant true
// CHECK-NEXT:    %c0 = constant 0 : index
// CHECK-NEXT:    cond_br %true, ^bb3, ^bb1(%c0 : index)
// CHECK-NEXT:  ^bb3:  
// CHECK-NEXT:    %c10 = constant 10 : index
// CHECK-NEXT:    %c2 = constant 2 : index
// CHECK-NEXT:    %2 = muli %c2, %arg0 : index
// CHECK-NEXT:    %3 = addi %c10, %2 : index
// CHECK-NEXT:    br ^bb1(%3 : index)
// CHECK-NEXT:  }
func @expression_apply(%d1 : index) -> index {

  %expr1 = presburger.expr #presburger<"expr(x)[] -> (2x + 10) : ()">
  %c1 = presburger.apply (%d1)[] %expr1 

  return %c1 : index
}

// CHECK-LABEL:  func @two_domain_apply(%arg0: index) -> index {
// CHECK-NEXT:    %0 = presburger.expr #presburger<"{{.*}}">
// CHECK-NEXT:    br ^bb2
// CHECK-NEXT:  ^bb1(%1: index):  
// CHECK-NEXT:    return %1 : index
// CHECK-NEXT:  ^bb2:  
// CHECK-NEXT:    %false = constant false
// CHECK-NEXT:    %true = constant true
// CHECK-NEXT:    %c0 = constant 0 : index
// CHECK-NEXT:    %c-1 = constant -1 : index
// CHECK-NEXT:    %2 = muli %c-1, %arg0 : index
// CHECK-NEXT:    %3 = addi %c0, %2 : index
// CHECK-NEXT:    %c0_0 = constant 0 : index
// CHECK-NEXT:    %4 = cmpi "sge", %3, %c0_0 : index
// CHECK-NEXT:    %5 = and %true, %4 : i1
// CHECK-NEXT:    %6 = or %false, %5 : i1
// CHECK-NEXT:    cond_br %6, ^bb3, ^bb4
// CHECK-NEXT:  ^bb3: 
// CHECK-NEXT:    %c1 = constant 1 : index
// CHECK-NEXT:    %c1_1 = constant 1 : index
// CHECK-NEXT:    %7 = muli %c1_1, %arg0 : index
// CHECK-NEXT:    %8 = addi %c1, %7 : index
// CHECK-NEXT:    br ^bb1(%8 : index)
// CHECK-NEXT:  ^bb4:
// CHECK-NEXT:    %false_2 = constant false
// CHECK-NEXT:    %true_3 = constant true
// CHECK-NEXT:    %c-1_4 = constant -1 : index
// CHECK-NEXT:    %c1_5 = constant 1 : index
// CHECK-NEXT:    %9 = muli %c1_5, %arg0 : index
// CHECK-NEXT:    %10 = addi %c-1_4, %9 : index
// CHECK-NEXT:    %c0_6 = constant 0 : index
// CHECK-NEXT:    %11 = cmpi "sge", %10, %c0_6 : index
// CHECK-NEXT:    %12 = and %true_3, %11 : i1
// CHECK-NEXT:    %13 = or %false_2, %12 : i1
// CHECK-NEXT:    %c0_7 = constant 0 : index
// CHECK-NEXT:    cond_br %13, ^bb5, ^bb1(%c0_7 : index)
// CHECK-NEXT:  ^bb5:
// CHECK-NEXT:    %c-10 = constant -10 : index
// CHECK-NEXT:    %c1_8 = constant 1 : index
// CHECK-NEXT:    %14 = muli %c1_8, %arg0 : index
// CHECK-NEXT:    %15 = addi %c-10, %14 : index
// CHECK-NEXT:    br ^bb1(%15 : index)
// CHECK-NEXT:  }
func @two_domain_apply(%d1 : index) -> index {

  %expr1 = presburger.expr #presburger<"expr(x)[] -> (x + 1) : (x <= 0); (x - 10) : (x >= 1)">
  %c1 = presburger.apply (%d1)[] %expr1 

  return %c1 : index
}
