#ifndef Presburger_DIALECT
#define Presburger_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Presburger dialect definition
//===----------------------------------------------------------------------===//

def Presburger_Dialect : Dialect {
    let name = "presburger";

    let description = [{
        A presburger dialect for MLIR.
    }];

    let cppNamespace = "presburger";
}

//===----------------------------------------------------------------------===//
// Presburger type definitions
//===----------------------------------------------------------------------===//

def PresburgerSetType : Type<CPred<"$_self.isa<PresburgerSetType>()">, "Presburger set type">;


def PresburgerSetAttr : Attr<CPred<"$_self.isa<PresburgerSetAttr>()">, "Presburger set attr"> {
    let storageType = [{ PresburgerSetAttr }];
    let returnType = [{ PresburgerSet }];
}

def PresburgerExprType : Type<CPred<"$_self.isa<PresburgerExprType>()">, "Presburger expression type">;


def PresburgerExprAttr : Attr<CPred<"$_self.isa<PresburgerExprAttr>()">, "Presburger expression attr"> {
    let storageType = [{ PresburgerExprAttr }];
    let returnType = [{ PresburgerExpr }];
}
//===----------------------------------------------------------------------===//
// Presburger trait definition
//===----------------------------------------------------------------------===//

def ProducesPresburgerSet : NativeOpTrait<"ProducesPresburgerSet">;

//===----------------------------------------------------------------------===//
// Presburger op definition
//===----------------------------------------------------------------------===//

// Base class for all Presburger ops.
class Presburger_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Presburger_Dialect, mnemonic, traits> {

    // For each Presburger op, the following static functions need to be defined in
    // PresburgerOps.cpp:
    let parser = [{ return ::parse$cppClass(parser, result); }];
    let printer = [{ ::print(p, *this); }];
    let verifier = [{ return ::verify(*this); }];
}

class Presburger_BinSetOp<string mnemonic, list<OpTrait> traits = [
        NoSideEffect, SameOperandsAndResultType, ProducesPresburgerSet]> :
    Presburger_Op<mnemonic, traits> {

    let arguments = (ins PresburgerSetType:$set1, PresburgerSetType:$set2);

    let results = (outs PresburgerSetType:$res);

    let parser = [{ return ::parseBinSetOp(parser, result); }];
    let verifier = [{ return ::verifyBinSetOp(*this); }];
}


#endif // Presburger_DIALECT
