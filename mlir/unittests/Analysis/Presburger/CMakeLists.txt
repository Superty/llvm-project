add_mlir_unittest(MLIRPresburgerTests
  IntegerPolyhedronTest.cpp
)

target_link_libraries(MLIRPresburgerTests
  PRIVATE MLIRPresburger
  MLIRAffineAnalysis
  MLIRParser
  )
 
