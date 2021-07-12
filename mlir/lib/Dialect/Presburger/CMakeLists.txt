add_mlir_dialect_library(MLIRPresburgerDialect
  PresburgerDialect.cpp
  PresburgerOps.cpp
  PresburgerOptions.cpp
  Attributes.cpp
  Types.cpp
  Transforms/Evaluation.cpp
  Transforms/Canonicalization.cpp
  
  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/include

  DEPENDS
  MLIRStandardOpsIncGen
  MLIRPresburgerOpsIncGen
  MLIRPresburgerPassIncGen
  MLIRPresburgerEvaluationPatterns
  MLIRPresburgerCanonicalizationPatterns
)

target_link_libraries(MLIRPresburgerDialect PUBLIC MLIRIR)

target_include_directories(MLIRPresburgerDialect PUBLIC "${MLIR_MAIN_INCLUDE_DIR}")
target_include_directories(MLIRPresburgerDialect PUBLIC "${MLIR_MAIN_INCLUDE_DIR}")


