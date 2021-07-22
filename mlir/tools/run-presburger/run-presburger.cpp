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
Optional<PresburgerSet<SafeInt<Int>>> setFromString(StringRef string) {
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
  PresburgerSet<SafeInt<Int>> res;
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
TransprecSet getSetFromInput() {
  char str[1'000'000];
  std::cin.getline(str, 1'000'000);
  // std::cerr << "Read '" << str << "'\n";
  if (auto set = setFromString<int16_t>(str))
    return TransprecSet(*set);
  else if (auto set = setFromString<int64_t>(str))
    return TransprecSet(*set);
  else if (auto set = setFromString<__int128_t>(str))
    return TransprecSet(*set);
  else if (auto set = setFromString<mpz_class>(str))
    return TransprecSet(*set);
  else
    llvm_unreachable("Input did not fit in 128-bits!");
  // return setFromString(str);
}

void consumeNewline() {
  char c;
  std::cin.get(c);
  if (c != '\n') {
    std::cerr << "Expected newline!\n";
    exit(1);
  }
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: ./run-presburger <op>\nPass input to stdin.\n";
    return 1;
  }

  const unsigned numRuns = 5;
  std::string op = argv[1];

  std::ofstream fruntime("data/runtime_fpl_" + op + ".txt");
  std::ofstream fwaterline("data/waterline_fpl_" + op + ".txt");
  std::ofstream fstat("data/stats_fpl_" + op + ".txt");
  std::error_code ec;
  llvm::raw_fd_ostream fout("data/outputs_fpl_" + op + ".txt", ec);
  if (ec) {
    std::cerr << "Could not open data/outputs_fpl_" + op + ".txt!\nTry creating the directory data.\n";
    return 1;
  }

  unsigned numCases;
  std::cin >> numCases;
  consumeNewline();
  fout << numCases << '\n';

  for (unsigned j = 0; j < numCases; ++j) {
    int times[numRuns];
    if (j % 50000 == 0)
      std::cerr << op << ' ' << j << '/' << numCases << '\n';

    TransprecSet::waterline = 0;
    if (op == "empty") {
      TransprecSet setA = getSetFromInput();
      for (unsigned i = 0; i < numRuns; ++i) {
        auto a = setA;
        unsigned int dummy;
        unsigned long long start = __rdtscp(&dummy);
        volatile auto res = a.isIntegerEmpty();
        res = res;
        unsigned long long end = __rdtscp(&dummy);
        times[i] = end - start;
        if (i == numRuns - 1) {
          fwaterline << TransprecSet::waterline << '\n';
          std::sort(times, times + numRuns);
          fruntime << times[numRuns/2] << '\n';
          fout << res << '\n';
        }
      }
    } else if (op == "equal") {
      TransprecSet setA = getSetFromInput();
      TransprecSet setB = getSetFromInput();
      for (unsigned i = 0; i < numRuns; ++i) {
        auto a = setA;
        auto b = setB;
        unsigned int dummy;
        unsigned long long start = __rdtscp(&dummy);
        volatile auto res = a.equal(b);
        res = res;
        unsigned long long end = __rdtscp(&dummy);
        times[i] = end - start;
        if (i == numRuns - 1) {
          fwaterline << TransprecSet::waterline << '\n';
          std::sort(times, times + numRuns);
          fruntime << times[numRuns/2] << '\n';
          fout << res << '\n';
        }
      }
    } else if (op == "union") {
      TransprecSet setA = getSetFromInput();
      TransprecSet setB = getSetFromInput();
      for (unsigned i = 0; i < numRuns; ++i) {
        auto a = setA;
        auto b = setB;
        unsigned int dummy;
        unsigned long long start = __rdtscp(&dummy);
        a.unionSet(b);
        unsigned long long end = __rdtscp(&dummy);
        times[i] = end - start;
        if (i == numRuns - 1) {
          fwaterline << TransprecSet::waterline << '\n';
          std::sort(times, times + numRuns);
          fruntime << times[numRuns/2] << '\n';
          dumpStats(fstat, a);
          a.printISL(fout);
          fout << '\n';
        }
      }
    } else if (op == "intersect") {
      TransprecSet setA = getSetFromInput();
      TransprecSet setB = getSetFromInput();
      for (unsigned i = 0; i < numRuns; ++i) {
        auto a = setA;
        auto b = setB;
        unsigned int dummy;
        unsigned long long start = __rdtscp(&dummy);
        a.intersectSet(b);
        unsigned long long end = __rdtscp(&dummy);
        times[i] = end - start;
        if (i == numRuns - 1) {
          fwaterline << TransprecSet::waterline << '\n';
          std::sort(times, times + numRuns);
          fruntime << times[numRuns/2] << '\n';
          dumpStats(fstat, a);
          a.printISL(fout);
          fout << '\n';
        }
      }
    } else if (op == "subtract") {
      TransprecSet setA = getSetFromInput();
      TransprecSet setB = getSetFromInput();
      for (unsigned i = 0; i < numRuns; ++i) {
        auto a = setA;
        auto b = setB;
        unsigned int dummy;
        unsigned long long start = __rdtscp(&dummy);
        a.subtract(b);
        unsigned long long end = __rdtscp(&dummy);
        times[i] = end - start;
        if (i == numRuns - 1) {
          fwaterline << TransprecSet::waterline << '\n';
          std::sort(times, times + numRuns);
          fruntime << times[numRuns/2] << '\n';
          dumpStats(fstat, a);
          a.printISL(fout);
          fout << '\n';
        }
      }
    } else if (op == "coalesce") {
      TransprecSet setA = getSetFromInput();
      for (unsigned i = 0; i < numRuns; ++i) {
        auto a = setA;
        unsigned int dummy;
        unsigned long long start = __rdtscp(&dummy);
        auto res = a.coalesce();
        unsigned long long end = __rdtscp(&dummy);
        times[i] = end - start;
        if (i == numRuns - 1) {
          fwaterline << TransprecSet::waterline << '\n';
          std::sort(times, times + numRuns);
          fruntime << times[numRuns/2] << '\n';
          dumpStats(fstat, res);
          res.printISL(fout);
          fout << '\n';
        }
      }
    } else if (op == "complement") {
      TransprecSet setA = getSetFromInput();
      for (unsigned i = 0; i < numRuns; ++i) {
        auto a = setA;
        unsigned int dummy;
        unsigned long long start = __rdtscp(&dummy);
        auto res = a.complement();
        unsigned long long end = __rdtscp(&dummy);
        times[i] = end - start;
        if (i == numRuns - 1) {
          fwaterline << TransprecSet::waterline << '\n';
          std::sort(times, times + numRuns);
          fruntime << times[numRuns/2] << '\n';
          dumpStats(fstat, a);
          res.printISL(fout);
          fout << '\n';
        }
      }
    } else if (op == "eliminate") {
      TransprecSet setA = getSetFromInput();
      for (unsigned i = 0; i < numRuns; ++i) {
        auto a = setA;
        unsigned int dummy;
        unsigned long long start = __rdtscp(&dummy);
        auto res = a.eliminateExistentials();
        unsigned long long end = __rdtscp(&dummy);
        times[i] = end - start;
        if (i == numRuns - 1) {
          fwaterline << TransprecSet::waterline << '\n';
          std::sort(times, times + numRuns);
          fruntime << times[numRuns/2] << '\n';
          dumpStats(fstat, a);
          a.printISL(fout);
          fout << '\n';
        }
      }
    } else {
      std::cerr << "Unsupported operation " << op << "!\n";
      return 1;
    }
    consumeLine();
  }
}
