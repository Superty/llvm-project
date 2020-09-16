add_mlir_library(MLIRPresburger
  Simplex.cpp
  Matrix.cpp
  Parser.cpp

  DEPENDS
  MLIRBuiltinLocationAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRParser
  )
