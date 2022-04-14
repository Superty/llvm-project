add_mlir_library(MLIRPresburger
  IntegerRelation.cpp
  LinearTransform.cpp
  Matrix.cpp
  PresburgerRelation.cpp
  PresburgerSpace.cpp
  PWMAFunction.cpp
  Simplex.cpp
  MPInt.cpp
  Utils.cpp

  LINK_LIBS PUBLIC
  MLIRSupport
  )
