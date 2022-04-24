add_mlir_library(MLIRPresburger
  MPInt.cpp
  MPAPInt.cpp
  
  LINK_LIBS PUBLIC
  MLIRSupport
  )
