#include "mlir/Analysis/Presburger/Coalesce.h"
#include "mlir/Analysis/Presburger/LinearTransform.h"
#include "mlir/Analysis/Presburger/Set.h"
#include "mlir/Dialect/Presburger/Parser.h"
#include <iostream>
#include <string>
#include <x86intrin.h>

using namespace mlir;
using namespace mlir::presburger;

static PresburgerSet setFromString(StringRef string) {
  ErrorCallback callback = [&](SMLoc loc, const Twine &message) {
    // This is a hack to make the Parser compile
    llvm::errs() << "Parsing error " << message << " at " << loc.getPointer()
                 << '\n';
    llvm::errs() << "invalid input " << string << '\n';

    llvm_unreachable("PARSING ERROR!!");
    MLIRContext context;
    return mlir::emitError(UnknownLoc::get(&context), message);
  };
  Parser parser(string, callback);
  PresburgerParser setParser(parser);
  PresburgerSet res;
  setParser.parsePresburgerSet(res);
  return res;
}

PresburgerSet getSetFromInput() {
  char str[1'000'000];
  std::cin.getline(str, 1'000'000);
  return setFromString(str);
}

unsigned long long getTime(unsigned long long start, unsigned long long end) {
  auto ret = fnTimer<PivotKey>::time;

  fnTimer<Simplex>::time = 0;
  fnTimer<FindPivotKey>::time = 0;
  fnTimer<AddRowKey>::time = 0;
  fnTimer<PivotKey>::time = 0;
  fnTimer<LinearTransform>::time = 0;
  return ret;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: ./run-presburger <op>\nPass input to stdin.\n";
    return 1;
  }

  const unsigned numRuns = 5;
  std::string op = argv[1];
  if (op == "empty") {
    PresburgerSet setA = getSetFromInput();
    for (unsigned i = 0; i < numRuns; ++i) {
      auto a = setA;
      unsigned int dummy;
      unsigned long long start = __rdtscp(&dummy);
      auto res = a.isIntegerEmpty();
      unsigned long long end = __rdtscp(&dummy);
      std::cerr << getTime(start, end) << '\n';
      if (i == numRuns - 1)
        std::cerr << res << '\n';
    }
  } else if (op == "equal") {
    PresburgerSet setA = getSetFromInput();
    PresburgerSet setB = getSetFromInput();
    for (unsigned i = 0; i < numRuns; ++i) {
      auto a = setA;
      auto b = setB;
      unsigned int dummy;
      unsigned long long start = __rdtscp(&dummy);
      auto res = PresburgerSet::equal(a, b);
      unsigned long long end = __rdtscp(&dummy);
      std::cerr << getTime(start, end) << '\n';
      if (i == numRuns - 1)
        llvm::errs() << res << '\n';
    }
  } else if (op == "union") {
    PresburgerSet setA = getSetFromInput();
    PresburgerSet setB = getSetFromInput();
    for (unsigned i = 0; i < numRuns; ++i) {
      auto a = setA;
      auto b = setB;
      unsigned int dummy;
      unsigned long long start = __rdtscp(&dummy);
      a.unionSet(std::move(b));
      unsigned long long end = __rdtscp(&dummy);
      std::cerr << getTime(start, end) << '\n';
      if (i == numRuns - 1)
        a.dumpISL();
    }
  } else if (op == "intersect") {
    PresburgerSet setA = getSetFromInput();
    PresburgerSet setB = getSetFromInput();
    for (unsigned i = 0; i < numRuns; ++i) {
      auto a = setA;
      auto b = setB;
      unsigned int dummy;
      unsigned long long start = __rdtscp(&dummy);
      a.intersectSet(std::move(b));
      unsigned long long end = __rdtscp(&dummy);
      std::cerr << getTime(start, end) << '\n';
      if (i == numRuns - 1)
        a.dumpISL();
    }
  } else if (op == "subtract") {
    PresburgerSet setA = getSetFromInput();
    PresburgerSet setB = getSetFromInput();
    for (unsigned i = 0; i < numRuns; ++i) {
      auto a = setA;
      auto b = setB;
      unsigned int dummy;
      unsigned long long start = __rdtscp(&dummy);
      a.subtract(std::move(b));
      unsigned long long end = __rdtscp(&dummy);
      std::cerr << getTime(start, end) << '\n';
      if (i == numRuns - 1)
        a.dumpISL();
    }
  } else if (op == "coalesce") {
    PresburgerSet setA = getSetFromInput();
    for (unsigned i = 0; i < numRuns; ++i) {
      auto a = setA;
      unsigned int dummy;
      unsigned long long start = __rdtscp(&dummy);
      auto res = coalesce(a);
      unsigned long long end = __rdtscp(&dummy);
      std::cerr << getTime(start, end) << '\n';
      if (i == numRuns - 1)
        res.dumpISL();
    }
  } else if (op == "complement") {
    PresburgerSet setA = getSetFromInput();
    for (unsigned i = 0; i < numRuns; ++i) {
      auto a = setA;
      unsigned int dummy;
      unsigned long long start = __rdtscp(&dummy);
      auto res = PresburgerSet::complement(a);
      unsigned long long end = __rdtscp(&dummy);
      std::cerr << getTime(start, end) << '\n';
      if (i == numRuns - 1)
        res.dumpISL();
    }
  } else if (op == "eliminate") {
    PresburgerSet setA = getSetFromInput();
    for (unsigned i = 0; i < numRuns; ++i) {
      auto a = setA;
      unsigned int dummy;
      unsigned long long start = __rdtscp(&dummy);
      auto res = PresburgerSet::eliminateExistentials(a);
      unsigned long long end = __rdtscp(&dummy);
      std::cerr << getTime(start, end) << '\n';
      if (i == numRuns - 1)
        a.dumpISL();
    }
  } else {
    std::cout << "Unsupported operation " << op << "!\n";
    return 1;
  }
}
