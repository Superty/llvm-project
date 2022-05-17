add_benchmark(MPIntBenchmark mpint.bench.cpp)

target_link_libraries(MPIntBenchmark
  PRIVATE
  MLIRPresburger
  LLVMSupport
  )

