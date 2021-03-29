add_mlir_library(MLIRPresburger
  Simplex.cpp
  Set.cpp
  ParamLexSimplex.cpp
  PresburgerBasicSet.cpp
  Expr.cpp
  Printer.cpp
  ISLPrinter.cpp
  Matrix.cpp
  LinearTransform.cpp
  Coalesce.cpp
  )

target_link_libraries(obj.MLIRPresburger PRIVATE /usr/lib/llvm-11/lib/clang/11.0.0/lib/linux/libclang_rt.builtins-x86_64.a)
