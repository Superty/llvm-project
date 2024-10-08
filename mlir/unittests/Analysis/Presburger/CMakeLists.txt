add_mlir_unittest(MLIRPresburgerBench
  IntegerPolyhedronTest.cpp
)

add_mlir_unittest(MLIRPresburgerTests
  BarvinokTest.cpp
  FractionTest.cpp
  GeneratingFunctionTest.cpp
  IntegerRelationTest.cpp
  LinearTransformTest.cpp
  MatrixTest.cpp
  Parser.h
  ParserTest.cpp
  PresburgerSetTest.cpp
  PresburgerRelationTest.cpp
  PresburgerSpaceTest.cpp
  PWMAFunctionTest.cpp
  QuasiPolynomialTest.cpp
  SimplexTest.cpp
  UtilsTest.cpp
)


target_link_libraries(MLIRPresburgerTests
  PRIVATE MLIRPresburger
  MLIRAffineAnalysis
  MLIRParser
  )

target_link_libraries(MLIRPresburgerBench
  PRIVATE MLIRPresburger
  MLIRAffineAnalysis
  MLIRParser
  )
  
