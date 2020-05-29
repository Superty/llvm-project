#ifndef Presburger_OPS
#define Presburger_OPS

include "PresburgerDialect.td"
include "mlir/Interfaces/SideEffects.td"


def Presburger_SetOp : Presburger_Op<"set", [NoSideEffect]> {
    let summary = "presburger set";

    let description = [{
        The `presburger.set` operator allows us to introduce a new presburger
        set as an SSA value. The type of the SSA value is `!presburger.set<dimC,symC>`,
        where `dimC` represents the amount of dimensions and `symC` the amount
        of symbols of the Presburger set.
        It requires a `PresburgerSetAttr` that captures a Presburger Set
        (similar to integer sets of the affine dialect).

        Note that no type is required, because it can be extracted from the attribute.

        **Examples:**

        ```mlir
        %0 = presburger.set #presburger<"(d0) : (d0 >= 0 and -d0 + 4 >= 0)">
        %1 = presburger.set #presburger<"(d0, d1)[s0, s1] : (d0 >= 0 and -d0 + s0 - 1 >= 0 or d1 >= 0 and -d1 + s1 - 1 >= 0)">
        ```
    }];

    let arguments = (ins PresburgerSetAttr:$set);

    let results = (outs PresburgerSetType:$res);
    let verifier = [{ return ::verify(*this); }];
}


def Presburger_UnionOp : Presburger_BinSetOp<"union"> {
    let summary = "union of two presburger sets";

    let description = [{
        The `presburger.union` operator allows us to perfom a union of two
        Presburger sets.
        It requires two operands of type `!presburger.set<dimC, symC>` that have
        matching `dimC` and `symC`.

        **Examples:**

        ```mlir
        %0 = presburger.set #presburger<"(d0) : (d0 >= 0 or -d0 + 4 >= 0)">
        %1 = presburger.set #presburger<"(d0) : (d0 - 1 >= 0 and -d0 + 3 >= 0)">
        %2 = presburger.union %0, %1 : !presburger.set<1,0>
        ```
    }];

    let hasCanonicalizer = 1;
}


def Presburger_IntersectOp : Presburger_BinSetOp<"intersect"> {
    let summary = "intersection of two presburger sets";

    let description = [{
        The `presburger.intersect` operator allows us to perfom an intersection of two
        Presburger sets.
        It requires two operands of type `!presburger.set<dimC, symC>` that have
        matching `dimC` and `symC`.

        **Examples:**

        ```mlir
        %0 = presburger.set #presburger<"(d0)[N] : (d0 - N >= 0 and (-d0) + 4 >= 0)">
        %1 = presburger.set #presburger<"(d0)[N] : (d0 - 1 >= 0 and -d0 + 3 >= 0)">
        %2 = presburger.intersect %0, %1 : !presburger.set<1,1>
        ```
    }];

    let hasCanonicalizer = 1;
}


def Presburger_SubtractOp : Presburger_BinSetOp<"subtract"> {
    let summary = "subtraction of two presburger sets";

    let description = [{
        The `presburger.subtract` operator allows us to perfom a subtraction of two
        Presburger sets.
        It requires two operands of type `!presburger.set<dimC, symC>` that have
        matching `dimC` and `symC`.

        **Examples:**
        // TODO change example

        ```mlir
        %0 = presburger.set #presburger<"(d0)[N] : (d0 - N >= 0 and (-d0) + 4 >= 0)">
        %1 = presburger.set #presburger<"(d0)[N] : (d0 - 1 >= 0 and -d0 + 3 >= 0)">
        %2 = presburger.intersect %0, %1 : !presburger.set<1,1>
        ```
    }];

    let hasCanonicalizer = 1;
}


#endif // Presburger_OPS
