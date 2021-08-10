#include "mlir/Analysis/Presburger/Coalesce.h"
#include "mlir/Analysis/Presburger/Set.h"
#include "mlir/Analysis/Presburger/TransprecSet.h"
#include "mlir/Dialect/Presburger/Parser.h"
#include "mlir/Analysis/Presburger/Presburger-impl.h"
#include <iostream>
#include <string>
#include <x86intrin.h>
#include <fstream>

using namespace mlir;
using namespace mlir::presburger;

unsigned TransprecSet::waterline = 0;

template <typename Int>
Optional<PresburgerSet<Int>> setFromString(StringRef string) {
  ErrorCallback callback = [&](SMLoc loc, const Twine &message) {
    // This is a hack to make the Parser compile
    // These have to be commented out currently because "errors" are raised
    // When an integer that can't fit in 32 bits appears in the input.
    // This is detected and handled by the transprecision infrastructures.
    // Unfortunately we do not yet check if it is an integer overflow error or
    // some other error, and all errors are assumed to be integer overflow errors.
    // If modifying something that might cause a different error here, note that
    // you have to uncomment the following to make the error be printed.
    // llvm::errs() << "Parsing error " << message << " at " << loc.getPointer()
    //              << '\n';
    // llvm::errs() << "invalid input " << string << '\n';

    // llvm_unreachable("PARSING ERROR!!");
    MLIRContext context;
    return mlir::emitError(UnknownLoc::get(&context), "");
  };
  Parser<Int> parser(string, callback);
  PresburgerParser<Int> setParser(parser);
  PresburgerSet<Int> res;
  if (failed(setParser.parsePresburgerSet(res)))
    return {};
  return res;
}

void dumpStats(std::ofstream &f, TransprecSet &a) {
  // a.dumpISL();
  // return;
  std::visit([&](auto &&set) {
    unsigned ids = set.getNumDims() + set.getNumSyms(), nDivs = 0, nEqs = 0, nIneqs = 0, nBS = 0;
    for (auto &bs : set.getBasicSets()) {
      ids = std::max(ids, bs.getNumTotalDims());
      nDivs += bs.getDivisions().size();
      nEqs += bs.getNumEqualities();
      nIneqs += bs.getNumInequalities();
      nBS += 1;
    }
    f << ids << ' ' << nBS << ' ' << nDivs << ' ' << nIneqs << ' ' << nEqs << '\n';
  }, a.setvar);
}

void consumeLine(unsigned cnt = 1) {
  while (cnt--) {
    char str[1'000'000];
    std::cin.getline(str, 1'000'000);
    // std::cerr << "Consumed '" << str << "'\n";
  }
}

// Exits the program if cin reached EOF.
TransprecSet getTransprecSetFromString(StringRef str) {
  // std::cerr << "Read '" << str << "'\n";
  if (auto set = setFromString<SafeInteger<int16_t>>(str))
    return TransprecSet(*set);
  else if (auto set = setFromString<SafeInteger<int64_t>>(str))
    return TransprecSet(*set);
  else if (auto set = setFromString<SafeInteger<__int128_t>>(str))
    return TransprecSet(*set);
  else if (auto set = setFromString<mpz_class>(str))
    return TransprecSet(*set);
  else
    llvm_unreachable("Input did not fit in 128-bits!");
}

template <typename Set>
Set getSetFromInput() {
  char str[1'000'000];
  std::cin.getline(str, 1'000'000);
  if constexpr (std::is_same_v<Set, TransprecSet>) {
    return getTransprecSetFromString(str);
  } else {
    if (auto set = setFromString<typename Set::UnderlyingInt>(str)) {
      return *set;
    } else
      llvm_unreachable("Input did not fit in specified precision!");
  }
}

void consumeNewline() {
  char c;
  std::cin.get(c);
  if (c != '\n') {
    std::cerr << "Expected newline!\n";
    exit(1);
  }
}

template <typename Set, bool printAuxInfo>
void run(std::string op, std::string suffix, std::optional<unsigned> maxWaterline) {
  if (!suffix.empty())
    assert(!printAuxInfo && "NYI");
  if (printAuxInfo)
    assert(!maxWaterline && "NYI");

  const unsigned numRuns = 5;
  unsigned numCases;
  std::cin >> numCases;
  consumeNewline();

  if (!suffix.empty())
    suffix = "_" + suffix;
  std::ifstream fwaterlineIn("data/waterline_fpl_" + op + ".txt");
  std::ofstream fruntime("data/runtime_fpl_" + op + suffix + ".txt");
  std::ofstream fwaterline, fstat;
  std::error_code EC;
  llvm::raw_fd_ostream fout(printAuxInfo ? "data/outputs_fpl_" + op + ".txt" : "data/empty_file_used_for_a_hack", EC);
  if (printAuxInfo) {
    fwaterline = std::ofstream("data/waterline_fpl_" + op + ".txt");
    fstat = std::ofstream("data/stats_fpl_" + op + ".txt");
    if (EC) {
      std::cerr << "Could not open outputs_fpl_" + op + ".txt!\n";
      std::abort();
    }
    fout << numCases << '\n';
  }

  for (unsigned j = 0; j < numCases; ++j) {
    int times[numRuns];
    if (j % 50000 == 0)
      std::cerr << op << ' ' << j << '/' << numCases << '\n';

    if (maxWaterline) {
      unsigned waterline;
      fwaterlineIn >> waterline;
      if (waterline > *maxWaterline) {
        consumeLine();
        consumeLine();
        if (op == "subtract" || op == "union" || op == "intersect" || op == "equal")
          consumeLine();
        continue;
      }
    }

    if constexpr (printAuxInfo)
      Set::waterline = 0;
    if (op == "empty") {
      Set setA = getSetFromInput<Set>();
      for (unsigned i = 0; i < numRuns; ++i) {
        auto a = setA;
        unsigned int dummy;
        unsigned long long start = __rdtscp(&dummy);
        volatile auto res = a.isIntegerEmpty();
        res = res;
        unsigned long long end = __rdtscp(&dummy);
        times[i] = end - start;
        if (i == numRuns - 1) {
          std::sort(times, times + numRuns);
          fruntime << times[numRuns/2] << '\n';
          if constexpr (printAuxInfo) {
            fwaterline << Set::waterline << '\n';
            fout << res << '\n';
          }
        }
      }
    } else if (op == "equal") {
      Set setA = getSetFromInput<Set>();
      Set setB = getSetFromInput<Set>();
      for (unsigned i = 0; i < numRuns; ++i) {
        auto a = setA;
        auto b = setB;
        unsigned int dummy;
        unsigned long long start = __rdtscp(&dummy);
        volatile auto res = Set::equal(a, b);
        res = res;
        unsigned long long end = __rdtscp(&dummy);
        times[i] = end - start;
        if (i == numRuns - 1) {
          std::sort(times, times + numRuns);
          fruntime << times[numRuns/2] << '\n';
          if constexpr (printAuxInfo) {
            fwaterline << Set::waterline << '\n';
            fout << res << '\n';
          }
        }
      }
    } else if (op == "union") {
      Set setA = getSetFromInput<Set>();
      Set setB = getSetFromInput<Set>();
      for (unsigned i = 0; i < numRuns; ++i) {
        auto a = setA;
        auto b = setB;
        unsigned int dummy;
        unsigned long long start = __rdtscp(&dummy);
        a.unionSet(b);
        unsigned long long end = __rdtscp(&dummy);
        times[i] = end - start;
        if (i == numRuns - 1) {
          std::sort(times, times + numRuns);
          fruntime << times[numRuns/2] << '\n';
          if constexpr (printAuxInfo) {
            fwaterline << Set::waterline << '\n';
            dumpStats(fstat, a);
            a.printISL(fout);
            fout << '\n';
          }
        }
      }
    } else if (op == "intersect") {
      Set setA = getSetFromInput<Set>();
      Set setB = getSetFromInput<Set>();
      for (unsigned i = 0; i < numRuns; ++i) {
        auto a = setA;
        auto b = setB;
        unsigned int dummy;
        unsigned long long start = __rdtscp(&dummy);
        a.intersectSet(b);
        unsigned long long end = __rdtscp(&dummy);
        times[i] = end - start;
        if (i == numRuns - 1) {
          std::sort(times, times + numRuns);
          fruntime << times[numRuns/2] << '\n';
          if constexpr (printAuxInfo) {
            fwaterline << Set::waterline << '\n';
            dumpStats(fstat, a);
            a.printISL(fout);
            fout << '\n';
          }
        }
      }
    } else if (op == "subtract") {
      Set setA = getSetFromInput<Set>();
      Set setB = getSetFromInput<Set>();
      for (unsigned i = 0; i < numRuns; ++i) {
        auto a = setA;
        auto b = setB;
        unsigned int dummy;
        unsigned long long start = __rdtscp(&dummy);
        a.subtract(b);
        unsigned long long end = __rdtscp(&dummy);
        times[i] = end - start;
        if (i == numRuns - 1) {
          std::sort(times, times + numRuns);
          fruntime << times[numRuns/2] << '\n';
          if constexpr (printAuxInfo) {
            fwaterline << Set::waterline << '\n';
            dumpStats(fstat, a);
            a.printISL(fout);
            fout << '\n';
          }
        }
      }
    } else if (op == "coalesce") {
      Set setA = getSetFromInput<Set>();
      for (unsigned i = 0; i < numRuns; ++i) {
        auto a = setA;
        unsigned int dummy;
        unsigned long long start = __rdtscp(&dummy);
        Set res = coalesce(a);
        unsigned long long end = __rdtscp(&dummy);
        times[i] = end - start;
        if (i == numRuns - 1) {
          std::sort(times, times + numRuns);
          fruntime << times[numRuns/2] << '\n';
          if constexpr (printAuxInfo) {
            fwaterline << Set::waterline << '\n';
            dumpStats(fstat, res);
            res.printISL(fout);
            fout << '\n';
          }
        }
      }
    } else if (op == "complement") {
      Set setA = getSetFromInput<Set>();
      for (unsigned i = 0; i < numRuns; ++i) {
        auto a = setA;
        unsigned int dummy;
        unsigned long long start = __rdtscp(&dummy);
        auto res = Set::complement(a);
        unsigned long long end = __rdtscp(&dummy);
        times[i] = end - start;
        if (i == numRuns - 1) {
          std::sort(times, times + numRuns);
          fruntime << times[numRuns/2] << '\n';
          if constexpr (printAuxInfo) {
            fwaterline << Set::waterline << '\n';
            dumpStats(fstat, a);
            res.printISL(fout);
            fout << '\n';
          }
        }
      }
    } else if (op == "eliminate") {
      Set setA = getSetFromInput<Set>();
      for (unsigned i = 0; i < numRuns; ++i) {
        auto a = setA;
        unsigned int dummy;
        unsigned long long start = __rdtscp(&dummy);
        auto res = Set::eliminateExistentials(a);
        unsigned long long end = __rdtscp(&dummy);
        times[i] = end - start;
        if (i == numRuns - 1) {
          std::sort(times, times + numRuns);
          fruntime << times[numRuns/2] << '\n';
          if constexpr (printAuxInfo) {
            fwaterline << Set::waterline << '\n';
            dumpStats(fstat, a);
            a.printISL(fout);
            fout << '\n';
          }
        }
      }
    } else {
      std::cerr << "Unsupported operation " << op << "!\n";
      std::abort();
    }
    consumeLine();
  }
}

int main(int argc, char **argv) {
  if (argc != 2 && argc != 3) {
    std::cerr << "usage: ./run-presburger <op> [precision:16/64/128/gmp/T]\nPass input to stdin.\n";
    return 1;
  }

  std::string op = argv[1];
  std::string prec = argv[2];
  if (prec == "16")
    run<PresburgerSet<int16_t>, false>(op, "16", 0);
  else if (prec == "64")
    run<PresburgerSet<int64_t>, false>(op, "64", 1);
  else if (prec == "128")
    run<PresburgerSet<__int128_t>, false>(op, "128", 2);
  else if (prec == "gmp")
    run<PresburgerSet<mpz_class>, false>(op, "gmp", 3);
  else if (prec == "T")
    run<TransprecSet, true>(op, "", {});
}

