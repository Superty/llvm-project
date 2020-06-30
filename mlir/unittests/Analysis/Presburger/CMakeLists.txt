add_mlir_unittest(MLIRPresburgerTests
  MatrixTest.cpp
  SimplexTest.cpp
  ParamLexSimplexTest.cpp
)

target_link_libraries(MLIRPresburgerTests
  PRIVATE MLIRPresburger)
