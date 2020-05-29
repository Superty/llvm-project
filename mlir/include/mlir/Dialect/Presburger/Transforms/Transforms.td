#ifndef PRESBURGER_TRANSFORMS
#define PRESBURGER_TRANSFORMS

include "PresburgerOps.td"

//===----------------------------------------------------------------------===//
// Presburger dialect transformations
//===----------------------------------------------------------------------===//

def UnionSets : NativeCodeCall<
    "unionSets($_builder, $0.getDefiningOp(), $1, $2)">;
def FoldUnionPattern : Pat<
  (Presburger_UnionOp:$op (Presburger_SetOp $set1), (Presburger_SetOp $set2)),
  (UnionSets $op, $set1, $set2)>;

def IntersectSets : NativeCodeCall<
    "intersectSets($_builder, $0.getDefiningOp(), $1, $2)">;
def FoldIntersectPattern : Pat<
  (Presburger_IntersectOp:$op (Presburger_SetOp $set1), (Presburger_SetOp $set2)),
  (IntersectSets $op, $set1, $set2)>;

def SubtractSets : NativeCodeCall<
    "subtractSets($_builder, $0.getDefiningOp(), $1, $2)">;
def FoldSubtractPattern : Pat<
  (Presburger_SubtractOp:$op (Presburger_SetOp $set1), (Presburger_SetOp $set2)),
  (SubtractSets $op, $set1, $set2)>;
#endif // PRESBURGER_TRANSFORMS
