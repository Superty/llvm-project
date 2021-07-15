add_mlir_unittest(MLIRPresburgerTests
  MatrixTest.cpp
  SimplexTest.cpp
  ParamLexSimplexTest.cpp
  CoalesceTest.cpp
  SetTest.cpp
  PresburgerBasicMapTest.cpp
)

target_link_libraries(MLIRPresburgerTests
  PRIVATE MLIRPresburger
  MLIRPresburgerDialect
  MLIRAnalysis
  MLIRLoopAnalysis)
