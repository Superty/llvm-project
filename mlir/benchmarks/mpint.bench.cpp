//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Simplex.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "benchmark/benchmark.h"

#define N (1024 * 16)

using namespace mlir::presburger;

static void mulI64(benchmark::State &state) {
  long a[N];
  long b[N];
  long c[N];
  for (int i = 0; i < N; i++) {
	a[i] = i;
	b[i] = i;
  }

  for (auto _ : state)
  // #pragma nounroll
 	for (int i = 0; i < N; i+=16) {
    // __sync_synchronize();
    c[i] = a[i] * b[i];
	}

  benchmark::DoNotOptimize(c[42]);
}
BENCHMARK(mulI64);

static void matrix(benchmark::State &state) {
  for (auto _ : state) {
    Matrix mat(4, 200);
    mat.fillRow(3, 1);
    benchmark::DoNotOptimize(matrix);
	}
}
BENCHMARK(matrix);

static void simplex(benchmark::State &state) {
  llvm::SmallVector<int64_t, 8> ineq1 = {1, -1};
  llvm::SmallVector<int64_t, 8> ineq2 = {-1, 0};
  for (auto _ : state) {
    Simplex simplex(1);
    simplex.addInequality(ineq1);
    simplex.addInequality(ineq2);
    benchmark::DoNotOptimize(matrix);
	}
}
BENCHMARK(simplex);

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;

  benchmark::RunSpecifiedBenchmarks();
}
