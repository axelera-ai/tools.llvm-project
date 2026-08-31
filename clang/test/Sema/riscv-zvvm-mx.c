// Microscaled (MX) integer-input MAC builtin gating and typing.
//
// The BS=32 cells are gated on Zvvxi8*; the BS=16 cells on Zvvxni8* (which
// imply their BS=32 siblings). MXINT inputs are always signed. There is no
// `_lm{N}` qualifier on these names (the `_i8m{N}` input token carries the
// A/B LMUL) and no `_bs8` block size.
//
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -target-feature +zvfhmin \
// RUN:   -target-feature +experimental-zvvxi8fp16mm \
// RUN:   -fsyntax-only -verify=bs32only %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -target-feature +zvfhmin \
// RUN:   -target-feature +experimental-zvvxni8fp16mm \
// RUN:   -fsyntax-only -verify=bs16 %s

#pragma clang riscv intrinsic zvvm_vector

#include <riscv_vector.h>

vfloat16m1_t test_bs32(vfloat16m1_t vd, vint8m1_t vs1, vint8m1_t vs2,
                       vuint16m1_t v0, size_t vl) {
  // Legal under both feature sets (Zvvxni8fp16mm implies Zvvxi8fp16mm).
  return __riscv_vfwimmacc_vv_f16m1_i8m1_bs32(vd, vs1, vs2, v0, vl);
}

vfloat16m1_t test_bs16(vfloat16m1_t vd, vint8m1_t vs1, vint8m1_t vs2,
                       vuint16m1_t v0, size_t vl) {
  // BS=16 requires the Zvvxn* extension.
  return __riscv_vfwimmacc_vv_f16m1_i8m1_bs16(vd, vs1, vs2, v0, vl); // bs32only-error {{builtin requires at least one of the following extensions: experimental-zvvxni8fp16mm}}
}

vfloat16m1_t test_unsigned_input(vfloat16m1_t vd, vuint8m1_t vs1,
                                 vuint8m1_t vs2, vuint16m1_t v0, size_t vl) {
  // MXINT inputs are always signed; a vuint8m1_t argument mismatches.
  return __riscv_vfwimmacc_vv_f16m1_i8m1_bs32(vd, vs1, vs2, v0, vl); // bs32only-error {{passing 'vuint8m1_t' (aka '__rvv_uint8m1_t') to parameter of incompatible type '__rvv_int8m1_t'}} bs16-error {{passing 'vuint8m1_t' (aka '__rvv_uint8m1_t') to parameter of incompatible type '__rvv_int8m1_t'}} bs32only-note {{passing argument to parameter here}} bs16-note {{passing argument to parameter here}}
}

vfloat16m1_t test_wrong_scale_type(vfloat16m1_t vd, vint8m1_t vs1,
                                   vint8m1_t vs2, vuint16m2_t v0, size_t vl) {
  // The paired-scale operand is exactly one register: vuint16m1_t.
  return __riscv_vfwimmacc_vv_f16m1_i8m1_bs32(vd, vs1, vs2, v0, vl); // bs32only-error {{passing 'vuint16m2_t' (aka '__rvv_uint16m2_t') to parameter of incompatible type '__rvv_uint16m1_t'}} bs16-error {{passing 'vuint16m2_t' (aka '__rvv_uint16m2_t') to parameter of incompatible type '__rvv_uint16m1_t'}} bs32only-note {{passing argument to parameter here}} bs16-note {{passing argument to parameter here}}
}

vfloat16m1_t test_no_bs8(vfloat16m1_t vd, vint8m1_t vs1, vint8m1_t vs2,
                         vuint16m1_t v0, size_t vl) {
  // No `_bs8` block size exists.
  return __riscv_vfwimmacc_vv_f16m1_i8m1_bs8(vd, vs1, vs2, v0, vl); // bs32only-error {{call to undeclared function '__riscv_vfwimmacc_vv_f16m1_i8m1_bs8'}} bs16-error {{call to undeclared function '__riscv_vfwimmacc_vv_f16m1_i8m1_bs8'}} bs32only-error {{returning 'int'}} bs16-error {{returning 'int'}}
}

vfloat16m1_t test_no_lm_qualifier(vfloat16m1_t vd, vint8m2_t vs1,
                                  vint8m2_t vs2, vuint16m1_t v0, size_t vl) {
  // The input token carries the LMUL; no `_lm{N}` spelling exists.
  return __riscv_vfwimmacc_vv_f16m1_lm2_bs32(vd, vs1, vs2, v0, vl); // bs32only-error {{call to undeclared function '__riscv_vfwimmacc_vv_f16m1_lm2_bs32'}} bs16-error {{call to undeclared function '__riscv_vfwimmacc_vv_f16m1_lm2_bs32'}} bs32only-error {{returning 'int'}} bs16-error {{returning 'int'}}
}
