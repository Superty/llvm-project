add_mlir_unittest(MLIRAnalysisTests
  AffineStructuresTest.cpp
  LinearTransformTest.cpp
  PresburgerSetTest.cpp
  ParamLexSimplexTest.cpp
)

target_link_libraries(MLIRAnalysisTests
  PRIVATE MLIRLoopAnalysis)

add_subdirectory(Presburger)
