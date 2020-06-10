add_mlir_unittest(MLIRPresburgerTests
  MatrixTest.cpp
  SimplexTest.cpp
  CoalesceTest.cpp
)

target_link_libraries(MLIRPresburgerTests
  PRIVATE MLIRPresburger
  MLIRPresburgerDialect
  MLIRAnalysis
  MLIRLoopAnalysis)
