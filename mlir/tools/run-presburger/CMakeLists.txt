set (LIBS
MLIRLoopAnalysis
MLIRPresburger
MLIRPresburgerDialect
)

add_llvm_tool(run-presburger
  run-presburger.cpp

  DEPENDS
  ${LIBS}
  )

target_link_libraries(run-presburger PRIVATE ${LIBS} /usr/lib/llvm-11/lib/clang/11.0.0/lib/linux/libclang_rt.builtins-x86_64.a)

llvm_update_compile_flags(run-presburger)
mlir_check_all_link_libraries(run-presburger)
