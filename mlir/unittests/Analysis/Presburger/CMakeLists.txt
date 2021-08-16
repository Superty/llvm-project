add_mlir_unittest(MLIRPresburgerTests
  MatrixTest.cpp
  SimplexTest.cpp
  ParamLexSimplexTest.cpp
  CoalesceTest.cpp
  SetTest.cpp
  PresburgerMapTest.cpp
)

target_link_libraries(MLIRPresburgerTests
  PRIVATE MLIRPresburgerDialect
  gmpxx
)
