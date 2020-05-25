get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRPresburger
        )

add_llvm_executable(presburgerc
 presburgerc.cpp
)

llvm_update_compile_flags(presburgerc)
target_link_libraries(presburgerc PRIVATE ${LIBS})
