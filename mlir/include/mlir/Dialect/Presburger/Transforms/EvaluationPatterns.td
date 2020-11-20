#ifndef MLIR_DIALECT_PRESBURGER_TRANSFORMS_EVALUATION_PATTERNS
#define MLIR_DIALECT_PRESBURGER_TRANSFORMS_EVALUATION_PATTERNS

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

def CoalesceSet : NativeCodeCall<
    "coalesceSet($_builder, $0.getDefiningOp(), $1)">;
def FoldCoalescePattern : Pat<
  (Presburger_CoalesceOp:$op (Presburger_SetOp $set)),
  (CoalesceSet $op, $set)>;

def ComplementSet : NativeCodeCall<
    "complementSet($_builder, $0.getDefiningOp(), $1)">;
def FoldComplementPattern : Pat<
  (Presburger_ComplementOp:$op (Presburger_SetOp $set)),
  (ComplementSet $op, $set)>;

def EqualSets : NativeCodeCall<
    "areEqualSets($_builder, $0.getDefiningOp(), $1, $2)">;
def FoldEqualPattern : Pat<
  (Presburger_EqualOp:$op (Presburger_SetOp $set1), (Presburger_SetOp $set2)),
  (EqualSets $op, $set1, $set2)>;

def EmptySet : NativeCodeCall<
    "emptySet($_builder, $0.getDefiningOp(), $1)">;
def FoldEmptyPattern : Pat<
  (Presburger_EmptyOp:$op (Presburger_SetOp $set)),
  (EmptySet $op, $set)>;

def EliminateExistentialsSet : NativeCodeCall<
    "eliminateExistentialsSet($_builder, $0.getDefiningOp(), $1)">;
def FoldEliminateExPattern : Pat<
  (Presburger_EliminateExOp:$op (Presburger_SetOp $set)),
  (EliminateExistentialsSet $op, $set)>;
#endif // MLIR_DIALECT_PRESBURGER_TRANSFORMS_EVALUATION_PATTERNS
