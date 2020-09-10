add_mlir_unittest(MLIRPresburgerTests
  MatrixTest.cpp
  SimplexTest.cpp
  SetTest.cpp
)

target_link_libraries(MLIRPresburgerTests
  PRIVATE MLIRLoopAnalysis)

