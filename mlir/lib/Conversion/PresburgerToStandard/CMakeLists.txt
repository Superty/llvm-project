add_mlir_conversion_library(MLIRPresburgerToStandard
  PresburgerToStandard.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/PresburgerToStandard

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRPresburgerDialect
  MLIRPass
  MLIRStandardOps
  MLIRTransforms
  MLIRIR
  )
