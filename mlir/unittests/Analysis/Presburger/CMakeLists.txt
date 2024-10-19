add_mlir_unittest(MLIRPresburgerTests
  IntegerPolyhedronTest.cpp
  SimplexTest.cpp
)

target_link_libraries(MLIRPresburgerTests
  PRIVATE MLIRPresburger
  MLIRAffineAnalysis
  MLIRParser
  )
 
