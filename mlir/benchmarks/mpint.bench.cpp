//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/MPInt.h"
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

static void mulMpint(benchmark::State &state) {
  MPInt a[N];
  MPInt b[N];
  MPInt c[N];
  for (int i = 0; i < N; i++) {
	a[i] = i;
	b[i] = i;
  }

  for (auto _ : state)
  // #pragma nounroll
 	for (int i = 0; i < N; i+=16) {
	    c[i] = a[i] * b[i];
	}

  benchmark::DoNotOptimize(c[42]);
}
BENCHMARK(mulMpint);

static void matrix(benchmark::State &state) {
  Matrix mat(4, 16);
  for (auto _ : state) {
    mat.swapColumns(1, 3);
    mat.swapColumns(1, 3);
    benchmark::DoNotOptimize(mat);
  }
}
BENCHMARK(matrix);

static void simplex(benchmark::State &state) {
  auto ineq = getMPIntVec({1, -1});
  auto coeffs = getMPIntVec({-1, 0});
  Simplex simplex(1);
  simplex.addInequality(ineq);
  for (auto _ : state) {
    simplex.computeOptimum(Simplex::Direction::Up, coeffs);
    benchmark::DoNotOptimize(simplex);
  }
}
BENCHMARK(simplex);

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;

  benchmark::RunSpecifiedBenchmarks();
}
