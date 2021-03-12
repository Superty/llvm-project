#ifndef MLIR_ANALYSIS_PRESBURGER_FNTIMER_H
#define MLIR_ANALYSIS_PRESBURGER_FNTIMER_H

#include <x86intrin.h>

template <typename T>
class fnTimer {
public:
  static unsigned long long time;
  fnTimer() {
    unsigned int dummy;
    start = __rdtscp(&dummy);
    timerDepth++;
  };
  ~fnTimer() {
    unsigned int dummy;
    unsigned long long end = __rdtscp(&dummy);
    timerDepth--;
    if (timerDepth == 0)
      fnTimer<T>::time += end - start;
  }
private:
  unsigned long long start;
  static unsigned timerDepth;
};

template <typename T>
unsigned long long fnTimer<T>::time = 0;
template <typename T>
unsigned fnTimer<T>::timerDepth = 0;

#endif // MLIR_ANALYSIS_PRESBURGER_FNTIMER_H

