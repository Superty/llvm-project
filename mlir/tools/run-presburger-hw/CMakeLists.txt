set (LIBS
MLIRPresburgerDialect
)

add_llvm_tool(run-presburger-hw
  run-presburger-hw.cpp

  DEPENDS
  ${LIBS}
  )

target_link_libraries(run-presburger-hw PRIVATE ${LIBS} /usr/lib/llvm-11/lib/clang/11.0.0/lib/linux/libclang_rt.builtins-x86_64.a gmp gmpxx)

llvm_update_compile_flags(run-presburger-hw)
mlir_check_all_link_libraries(run-presburger-hw)
