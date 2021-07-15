add_mlir_library(MLIRPresburger
  Simplex.cpp
  Set.cpp
  ParamLexSimplex.cpp
  PresburgerBasicSet.cpp
  PresburgerBasicMap.cpp
  Expr.cpp
  Printer.cpp
  ISLPrinter.cpp
  Matrix.cpp
  LinearTransform.cpp
  Coalesce.cpp
  )

target_compile_options(obj.MLIRPresburger PRIVATE -ftrapv)
