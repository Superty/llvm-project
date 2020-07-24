/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 * SPDX-FileCopyrightText: 2019 Arjun Pitchanathan <arjunpitchanathan@gmail.com>
 */

#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/Presburger/LinearTransform.h"

namespace mlir {

FlatAffineConstraints
LinearTransform::postMultiplyBasicSet(const FlatAffineConstraints &bs) {
  FlatAffineConstraints result(bs.getNumDimIds());

  for (unsigned i = 0; i < bs.getNumEqualities(); ++i) {
    ArrayRef<int64_t> eq = bs.getEquality(i);

    int64_t c = eq.back();

    SmallVector<int64_t, 8> newEq = postMultiplyRow(eq.drop_back());
    newEq.push_back(c);
    result.addEquality(newEq);
  }

  for (unsigned i = 0; i < bs.getNumInequalities(); ++i) {
    ArrayRef<int64_t> ineq = bs.getInequality(i);

    int64_t c = ineq.back();

    SmallVector<int64_t, 8> newIneq = postMultiplyRow(ineq.drop_back());
    newIneq.push_back(c);
    result.addInequality(newIneq);
  }

  // bs.simplify(); // isl does this here
  return result;
}

} // namespace mlir