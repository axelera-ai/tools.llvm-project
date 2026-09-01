// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +v \
// RUN:   -target-feature +experimental-zvvi32mm \
// RUN:   -target-feature +experimental-zvvfp32mm \
// RUN:   -fsyntax-only -verify %s

#pragma clang riscv intrinsic zvvm_vector

#include <riscv_vector.h>

// Sema surface of the IME m16 (EMUL_C = 16) types and intrinsics: the
// v<elt>m16_t types are distinct from every m1..m8 type, the m16 MAC cells
// type-check both operand axes, and pair/unpair only accept the matching
// m8 halves / m16 source.

vint32m16_t good_mac(vint32m16_t vd, vint32m1_t vs1, vint32m1_t vs2,
                     size_t vl) {
  return __riscv_vmmacc_vv_i32m16(vd, vs1, vs2, vl);
}

vint32m16_t bad_accumulator(vint32m8_t vd, vint32m1_t vs1, vint32m1_t vs2,
                            size_t vl) {
  // expected-error@+2 {{passing 'vint32m8_t' (aka '__rvv_int32m8_t') to parameter of incompatible type '__rvv_int32m16_t'}}
  // expected-note@+1 {{passing argument to parameter here}}
  return __riscv_vmmacc_vv_i32m16(vd, vs1, vs2, vl);
}

vint32m16_t bad_input_lmul(vint32m16_t vd, vint32m2_t vs1, vint32m2_t vs2,
                           size_t vl) {
  // The unqualified name is the LMUL=1 cell; m2 inputs need _lm2.
  // expected-error@+2 {{passing 'vint32m2_t' (aka '__rvv_int32m2_t') to parameter of incompatible type '__rvv_int32m1_t'}}
  // expected-note@+1 {{passing argument to parameter here}}
  return __riscv_vmmacc_vv_i32m16(vd, vs1, vs2, vl);
}

vint32m16_t bad_sign(vint32m16_t vd, vuint32m1_t vs1, vint32m1_t vs2,
                     size_t vl) {
  // Unsigned A on the ss cell: the _us spelling is required.
  // expected-error@+2 {{passing 'vuint32m1_t' (aka '__rvv_uint32m1_t') to parameter of incompatible type '__rvv_int32m1_t'}}
  // expected-note@+1 {{passing argument to parameter here}}
  return __riscv_vmmacc_vv_i32m16(vd, vs1, vs2, vl);
}

vint32m16_t bad_pair_half(vint32m4_t lo, vint32m8_t hi) {
  // Pair halves must be m8 groups.
  // expected-error@+2 {{passing 'vint32m4_t' (aka '__rvv_int32m4_t') to parameter of incompatible type '__rvv_int32m8_t'}}
  // expected-note@+1 {{passing argument to parameter here}}
  return __riscv_ime_vpair_i32m16(lo, hi);
}

vint32m8_t bad_unpair_source(vint32m8_t src) {
  // Unpair requires an m16 source, not an m8 group.
  // expected-error@+2 {{passing 'vint32m8_t' (aka '__rvv_int32m8_t') to parameter of incompatible type '__rvv_int32m16_t'}}
  // expected-note@+1 {{passing argument to parameter here}}
  return __riscv_ime_vunpairlo_i32m16(src);
}

vint32m16_t mixed_m16_types(vint32m16_t vd, vfloat32m16_t fd) {
  // m16 types are distinct per element type.
  // expected-error@+1 {{returning 'vfloat32m16_t' (aka '__rvv_float32m16_t') from a function with incompatible result type 'vint32m16_t' (aka '__rvv_int32m16_t')}}
  return fd;
}

void no_m16_tile_ls(int32_t *base, size_t ld, vint32m16_t v, size_t vl) {
  // No single-instruction m16 tile load/store exists (LMUL=16 is not a legal
  // vtype setting); software goes through pair/unpair + two m8 accesses.
  // expected-error@+1 {{call to undeclared function '__riscv_vmts_v_i32m16'}}
  __riscv_vmts_v_i32m16(base, ld, v, vl);
}

vint32m16_t no_lm16(vint32m16_t vd, vint32m1_t vs1, vint32m1_t vs2,
                    size_t vl) {
  // _lm1 stays omitted for LMUL=1 even at EMUL_C=16.
  // expected-error@+2 {{call to undeclared function '__riscv_vmmacc_vv_i32m16_lm1'}}
  // expected-error@+1 {{returning 'int' from a function with incompatible result type 'vint32m16_t' (aka '__rvv_int32m16_t')}}
  return __riscv_vmmacc_vv_i32m16_lm1(vd, vs1, vs2, vl);
}
