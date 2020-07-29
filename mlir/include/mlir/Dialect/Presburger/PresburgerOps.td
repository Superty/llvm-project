#ifndef Presburger_OPS
#define Presburger_OPS

include "PresburgerDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


def Presburger_SetOp : Presburger_Op<"set", [NoSideEffect, ProducesPresburgerSet]> {
    let summary = "presburger set";

    let description = [{
        Syntax:
        ```
        operation     ::= `presburger.set #presburger<"set ` pb-set `">` 
        pb-set        ::= dim-and-symbol-use-list `:` `(` pb-or-expr? `)`
        pb-or-expr    ::= pb-and-expr (`or` pb-and-expr)*
        pb-and-expr   ::= pb-constraint (`and` pb-constraint)*
        pb-constraint ::= pb-sum (`>=` | `=` | `<=`) pb-sum
        pb-sum        ::= pb-term (('+' | '-') pb-term)*
        pb-term       ::= '-'? pb-int? pb-var
                      ::= '-'? pb-int
        pb-var        ::= letter (digit | letter)*
        pb-int        ::= digit+
        ```

        The `presburger.set` operator allows us to introduce a new presburger
        set as an SSA value. The type of the SSA value is `!presburger.set<dimC,symC>`,
        where `dimC` represents the amount of dimensions and `symC` the amount
        of symbols of the Presburger set.
        It requires a `PresburgerSetAttr` that captures a Presburger Set
        (similar to integer sets of the affine dialect).

        Note that no type is required, because it can be extracted from the attribute.

        Note that for now the expressions only support affine once, more is comming
        soon.

        **Examples:**

        ```mlir
        %0 = presburger.set #presburger<"(d0) : (d0 >= 0 and -d0 + 4 >= 0)">
        %1 = presburger.set #presburger<"(d0, d1)[s0, s1] : (d0 >= 0 and -d0 + s0 - 1 >= 0 or d1 >= 0 and -d1 + s1 - 1 >= 0)">
        ```
    }];

    let arguments = (ins PresburgerSetAttr:$set);

    let results = (outs PresburgerSetType:$res);
}

def Presburger_ExprOp : Presburger_Op<"expr", [NoSideEffect]> {
    let summary = "piecewise Presburger expression";

    let description = [{
        Syntax:
        ```
        operation  ::= `presburger.expr #presburger<"expr` pb-pw-expr `">` 
        pb-pw-expr ::= dim-and-symbol-use-list `->` piece (`;` piece)* 
        piece      ::= `(`pb-sum`) : (`pb-or-expr`)`
        ```

        The `presburger.expr` operator allows us to introduce a new piecewise
        Presburger expression as an SSA value. The type of the SSA value is 
        `!presburger.expr<dimC,symC>`, where `dimC` represents the amount of 
        dimensions and `symC` the amount of symbols of the Presburger expressions.
        It requires a `PresburgerExprAttr` that captures a piecewise Presburger 
        expression.

        Note that no type is required, because it can be extracted from the 
        attribute.

        TODO: define the behavior of values outside of domains. 

        **Examples:**

        ```mlir
        %expr1 = presburger.expr #presburger<"expr (x)[] -> (x + 1) : (x >= 0) ; (0) : (-x + 1 >= 0)">
        %expr2 = presburger.expr #presburger<"expr (x,y)[] -> (x) : (x >= 0 and 2y >= 0) ; (y) : (-x + 1 >= 0)">
        ```
    }];

    let arguments = (ins PresburgerExprAttr:$expr);
    let results = (outs PresburgerExprType:$res);
}


def Presburger_UnionOp : Presburger_BinSetOp<"union"> {
    let summary = "union of two presburger sets";

    let description = [{
        The `presburger.union` operator allows us to perfom a union of two
        Presburger sets.
        It requires two operands of type `!presburger.set<dimC, symC>` that have
        matching `dimC` and `symC`.
        The sets need to be defined locally, otherwise this operation is invalid. 

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
        The `presburger.intersect` operator allows us to perfom an intersection 
        of two Presburger sets.
        It requires two operands of type `!presburger.set<dimC, symC>` that have
        matching `dimC` and `symC`.
        The sets need to be defined locally, otherwise this operation is invalid. 

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
        The sets need to be defined locally, otherwise this operation is invalid. 


        **Examples:**
        // TODO change example

        ```mlir
        %0 = presburger.set #presburger<"(d0)[N] : (d0 - N >= 0 and (-d0) + 4 >= 0)">
        %1 = presburger.set #presburger<"(d0)[N] : (d0 - 1 >= 0 and -d0 + 3 >= 0)">
        %2 = presburger.subtract %0, %1 : !presburger.set<1,1>
        ```
    }];

    let hasCanonicalizer = 1;
}


def Presburger_ComplementOp : Presburger_Op<"complement", [NoSideEffect,
                              SameOperandsAndResultType, ProducesPresburgerSet]> {
    let summary = "complement of a Presburger set";

    let description = [{
        The `presburger.complement` operator allows us to compute the complement 
        of a given Presburger set. The set need to be defined locally, otherwise 
        this operation is invalid. 

        **Examples:**

        TODO

        ```mlir
        %0 = presburger.set #presburger<"(d0) : (d0 >= 0 and -d0 + 4 >= 0)">
        %1 = presburger.complement %0 : !presburger<1,0>
        ```
    }];

    let arguments = (ins PresburgerSetType:$set);

    let results = (outs PresburgerSetType:$res);

    let hasCanonicalizer = 1;
}

def Presburger_EqualOp : Presburger_Op<"equal", [NoSideEffect, 
                                                      SameTypeOperands]> {
    let summary = "checks set equality";

    let description = [{
        The `presburger.equal` operator performs a semantic equality check. 
        If both Presburger sets contain the examt same points they are considered
        equal. The result type is i1.

        **Examples:**


        ```mlir
        %0 = presburger.set #presburger<"(d0)[] : (d0 >= 0 and -d0 + 4 >= 0)">
        %1 = presburger.set #presburger<"(d0)[] : (d0 - 1 >= 0 and -d0 + 3 >= 0)">
        %2 = presburger.equal %0, %1 : !presburger.set<1,0>, !presburger.set<1,0>
        ```
    }];

    let arguments = (ins PresburgerSetType:$set1, PresburgerSetType:$set2);

    let results = (outs BoolLike:$res);
    let verifier = [{ return ::verifyBinSetOp(*this); }];

    let hasCanonicalizer = 1;
}

def Presburger_ContainsOp : Presburger_Op<"contains", [NoSideEffect]> {
    let summary = "checks if a certain point is inside the set";

    let description = [{
        The presburger.contains operation checks if the provided point is 
        contained in the set. If the set has symbols, they need to be bound 
        as well. All the provided SSA values - except the Presburger set - 
        must all have `index` type.

        Example:

        ```mlir
        %set = presburger.set #presburger<"(x)[s] : (x >= 0 and -x + 4 >= s)">
        ...

        %contained = presburger.contains (%d1)[%s1] %set 
        ```
    }];

    let arguments = (ins Variadic<Index>:$dimAndSyms, PresburgerSetType:$set);

    let results = (outs BoolLike:$res);

}

#endif // Presburger_OPS
