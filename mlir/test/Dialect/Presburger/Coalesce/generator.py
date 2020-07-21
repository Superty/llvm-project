testFile = open("new_tests.txt", "r")

i = 0
for line in testFile:
    print(i)
    # used to only get a subset of the tests.
    # if (i < 4191 or i > 4201):
    #    i += 1
    #    continue
    string = "coalesceBenchmark" + str(i) + ".test"
    writeFile = open(string, "w")
    line_arr = line.split(":")
    dims = line_arr[0]
    dims_arr = dims.split(")")
    params = dims_arr[0]
    syms = dims_arr[1]
    param_count = params.count("d")
    syms_count = syms.count("p")
    line = line[:len(line)-1]
    writeFile.write("// RUN: mlir-opt -canonicalize %s | FileCheck %s\n"
            "\n"
            "// CHECK-LABEL: func @performance" + str(i) + "\n"
            "func @performance" + str(i) + "() -> i1 {\n"
            "  // CHECK-NEXT: %[[S:.*]] = constant true\n"
            "  // CHECK-NEXT: return %[[S]]\n"
            "  %set = presburger.set #presburger<\"" + line + "\">\n"
            "\n"
            "  %r = presburger.coalesce %set : !presburger.set<" + str(param_count) + "," + str(syms_count) + ">\n"
            "  %e = presburger.equal %set, %r : !presburger.set<" + str(param_count) + "," + str(syms_count) + ">, !presburger.set<" + str(param_count) + "," + str(syms_count) + ">\n"
            "  return %e : i1\n"
            "}\n"
            "\n"
            "// ----\n")
    i += 1
