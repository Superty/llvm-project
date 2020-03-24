add_mlir_unittest(MLIRAnalysisTests
  MatrixTest.cpp
  SimplexTest.cpp
)

target_link_libraries(MLIRAnalysisTests
  PRIVATE MLIRAnalysis)