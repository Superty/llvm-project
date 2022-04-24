//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include "mlir/Analysis/Presburger/MPInt.h"

#include "benchmark/benchmark.h"

#define N (1024 * 16)

using namespace mlir::presburger;

static void mul_i64(benchmark::State &State) {
  long A[N];
  long B[N];
  long C[N];
  for (int i = 0; i < N; i++) {
	A[i] = i;
	B[i] = i;
  }

  for (auto _ : State)
 	for (int i = 0; i < N; i+=16) {
	    __sync_synchronize();
	    C[i] = A[i] * B[i];
	}

  benchmark::DoNotOptimize(C[42]);
}
BENCHMARK(mul_i64);

static void mul_mpint(benchmark::State &State) {
  MPInt A[N];
  MPInt B[N];
  MPInt C[N];
  for (int i = 0; i < N; i++) {
	A[i] = i;
	B[i] = i;
  }

  for (auto _ : State)
 	for (int i = 0; i < N; i+=16) {
	    __sync_synchronize();
	    C[i] = A[i] * B[i];
	}

  benchmark::DoNotOptimize(C[42]);
}
BENCHMARK(mul_mpint);

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;

  benchmark::RunSpecifiedBenchmarks();
}
