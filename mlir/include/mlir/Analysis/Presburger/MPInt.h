//===- MPInt.h - MLIR MPInt Class -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a simple class to represent arbitrary precision signed integers.
// Unlike APInt2, one does not have to specify a fixed maximum size, and the
// integer can take on any aribtrary values.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_MPINT_H
#define MLIR_ANALYSIS_PRESBURGER_MPINT_H

#include "mlir/Analysis/Presburger/MPAPInt.h"
#include "mlir/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <unistd.h>
#include <variant>

namespace mlir {
namespace presburger {

/// This class provides support for multi-precision arithmetic.
///
/// Unlike APInt2, this extends the precision as necessary to prevent overflows
/// and supports operations between objects with differing internal precisions.
///
/// Since it uses APInt2 internally, MPInt (MultiPrecision Integer) stores
/// values in a 64-bit machine integer for small values and uses slower
/// arbitrary-precision arithmetic only for larger values.
class MPInt {
public:
  __attribute__((always_inline))
  explicit MPInt(int64_t val) : val64(val), holdsAP(false) {}
  __attribute__((always_inline))
  MPInt() : MPInt(0) {}
  ~MPInt() {
    if (isLarge())
      valAP.detail::MPAPInt::~MPAPInt();
  }
  __attribute__((always_inline))
  MPInt(const MPInt &o) : val64(o.val64), holdsAP(false) {
    if (o.isLarge())
      initAP(o.valAP);
  }
  __attribute__((always_inline))
  MPInt &operator=(const MPInt &o) {
    if (o.isLarge())
      initAP(o.valAP);
    else
      init64(o.val64);
    return *this;
  }
  MPInt &operator=(int x) {
    init64(x);
    return *this;
  }

  MPInt operator*(const MPInt &o) const;

private:
  explicit MPInt(const detail::MPAPInt &val) : valAP(val), holdsAP(true) {}
  __attribute__((always_inline)) bool isSmall() const { return !holdsAP; }
  __attribute__((always_inline)) bool isLarge() const { return holdsAP; }
  __attribute__((always_inline)) int64_t get64() const {
    assert(isSmall());
    return val64;
  }
  __attribute__((always_inline)) int64_t &get64() {
    assert(isSmall());
    return val64;
  }
  __attribute__((always_inline)) const detail::MPAPInt &getAP() const {
    assert(isLarge());
    return valAP;
  }
  __attribute__((always_inline)) detail::MPAPInt &getAP() {
    assert(isLarge());
    return valAP;
  }
  explicit operator detail::MPAPInt() const {
    if (isSmall())
      return detail::MPAPInt(get64());
    return getAP();
  }
  detail::MPAPInt getAsAP() const {
    return detail::MPAPInt(*this);
  }

  union {
    int64_t val64;
    detail::MPAPInt valAP;
  };
  void init64(int64_t o) {
    val64 = o;
    holdsAP = false;
  }
  void initAP(const detail::MPAPInt &o) {
    valAP = o;
    holdsAP = true;
  }
  bool holdsAP;
};

__attribute__((always_inline))
inline MPInt MPInt::operator*(const MPInt &o) const {
  if (isSmall() && o.isSmall()) {
    MPInt result;
    bool overflow = __builtin_mul_overflow(get64(), o.get64(), &result.get64());
    if (!overflow) {
      return result;
    }
  }
  return MPInt(getAsAP() * o.getAsAP());
}

} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_MPINT_H
