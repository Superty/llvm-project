set(LLVM_TARGET_DEFINITIONS Transforms.td )
mlir_tablegen(Transforms.cpp.inc -gen-rewriters
    "-I${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Presburger")
add_public_tablegen_target(MLIRPresburgerTransforms)
