// RUN: %clang_cc1 -triple riscv64 -target-feature +v \
// RUN:   -target-feature +experimental-zvvfmm \
// RUN:   -dwarf-version=4 -debug-info-kind=limited -emit-llvm -o - %s \
// RUN:   | FileCheck --check-prefix=DEBUGINFO %s

// IME (Zvvm) OFP8 vector types: scalable arrays of the i8 container element
// (no FP8 DWARF base type), named after the __rvv_* builtin type.

__rvv_float8e4m3m2_t f1(__rvv_float8e4m3m2_t arg) { return arg; }

// DEBUGINFO: !DIDerivedType(tag: DW_TAG_typedef, name: "__rvv_float8e4m3m2_t", {{.*}}baseType: ![[E4M3M2:[0-9]+]])
// DEBUGINFO: ![[E4M3M2]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[UCHAR:[0-9]+]], flags: DIFlagVector, elements: ![[E4M3M2ELTS:[0-9]+]])
// DEBUGINFO: ![[UCHAR]] = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
// DEBUGINFO: ![[E4M3M2ELTS]] = !{![[E4M3M2SUB:[0-9]+]]}
// DEBUGINFO: ![[E4M3M2SUB]] = !DISubrange(lowerBound: 0, upperBound: !DIExpression(DW_OP_bregx, 7202, 0, DW_OP_constu, 1, DW_OP_div, DW_OP_constu, 2, DW_OP_mul, DW_OP_constu, 1, DW_OP_minus))

__rvv_float8e5m2mf2_t f2(__rvv_float8e5m2mf2_t arg) { return arg; }

// DEBUGINFO: !DIDerivedType(tag: DW_TAG_typedef, name: "__rvv_float8e5m2mf2_t"
// DEBUGINFO: !DISubrange(lowerBound: 0, upperBound: !DIExpression(DW_OP_bregx, 7202, 0, DW_OP_constu, 1, DW_OP_div, DW_OP_constu, 2, DW_OP_div, DW_OP_constu, 1, DW_OP_minus))
