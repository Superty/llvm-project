add_mlir_library(MLIRPresburger
  Simplex.cpp
  )

target_link_libraries(MLIRPresburger
  PUBLIC
  MLIRAnalysis
  )

