add_mlir_library(MLIRPresburger
  Simplex.cpp
  Matrix.cpp
  Coalesce.cpp

  DEPENDS
  MLIRBuiltinLocationAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )
