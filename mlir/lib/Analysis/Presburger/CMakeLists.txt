add_mlir_library(MLIRPresburger
  Simplex.cpp
  LinearTransform.cpp
  Coalesce.cpp
  )

target_link_libraries(MLIRPresburger
  PUBLIC
  MLIRAnalysis
  )

