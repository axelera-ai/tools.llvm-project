// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -target-feature +experimental-zvvmm \
// RUN:   -target-feature +experimental-zvvfmm -fsyntax-only -verify %s

// Verify the (EMUL_C, LMUL) MAC C-surface: distinct builtins per cell, with
// per-operand types pinned to the LMUL suffix in the name. Wrong-LMUL inputs
// must diagnose; no _lm16 / non-power-of-two suffix exists. Per the IME
// spec's canonical suffix order the LMUL=1 name carries no qualifier — an
// explicit _lm1 spelling does not exist.

#pragma clang riscv intrinsic zvvm_vector

#include <riscv_vector.h>

vint32m1_t ok_m1(vint32m1_t vd, vint32m1_t vs1, vint32m1_t vs2, size_t vl) {
  return __riscv_vmmacc_vv_i32m1(vd, vs1, vs2, vl); // ok
}

vint32m1_t ok_m1_lm2(vint32m1_t vd, vint32m2_t vs1, vint32m2_t vs2, size_t vl) {
  return __riscv_vmmacc_vv_i32m1_lm2(vd, vs1, vs2, vl); // ok
}

vint32m1_t bad_ab_lmul(vint32m1_t vd, vint32m1_t vs1, vint32m1_t vs2,
                       size_t vl) {
  // Builtin expects vs1/vs2 at LMUL=2 (= __rvv_int32m2_t); passing m1 must
  // diagnose. Sema stops at the first wrong operand, so only one error is
  // reported (with a "passing argument to parameter here" note).
  return __riscv_vmmacc_vv_i32m1_lm2(vd, vs1, vs2, vl);
  // expected-error@-1 {{passing 'vint32m1_t'}}
  // expected-note@-2 {{passing argument to parameter here}}
}

vint32m2_t bad_c_emul_c(vint32m1_t vd, vint32m2_t vs1, vint32m2_t vs2,
                        size_t vl) {
  // Builtin expects vd at EMUL_C=2 (= __rvv_int32m2_t); passing m1 must
  // diagnose.
  return __riscv_vmmacc_vv_i32m2_lm2(vd, vs1, vs2, vl);
  // expected-error@-1 {{passing 'vint32m1_t'}}
  // expected-note@-2 {{passing argument to parameter here}}
}

vint32m1_t no_lm16(vint32m1_t vd, vint32m1_t vs1, vint32m1_t vs2, size_t vl) {
  // _lm16 is not in the EMUL_C / LMUL ∈ {1,2,4,8} cross-product — there's no
  // VRM16 register class upstream and the IME spec caps LMUL at 8.
  return __riscv_vmmacc_vv_i32m1_lm16(vd, vs1, vs2, vl);
  // expected-error@-1 {{call to undeclared function '__riscv_vmmacc_vv_i32m1_lm16'}}
  // expected-error@-2 {{returning 'int' from a function with incompatible result type}}
}

vint32m1_t no_lm3(vint32m1_t vd, vint32m1_t vs1, vint32m1_t vs2, size_t vl) {
  // Non-power-of-two LMUL suffix does not exist.
  return __riscv_vmmacc_vv_i32m1_lm3(vd, vs1, vs2, vl);
  // expected-error@-1 {{call to undeclared function '__riscv_vmmacc_vv_i32m1_lm3'}}
  // expected-error@-2 {{returning 'int' from a function with incompatible result type}}
}

vint32m1_t no_lm1(vint32m1_t vd, vint32m1_t vs1, vint32m1_t vs2, size_t vl) {
  // The LMUL=1 form is the unqualified name; an explicit _lm1 spelling is
  // not provided (spec: the _lm1 qualifier is omitted).
  return __riscv_vmmacc_vv_i32m1_lm1(vd, vs1, vs2, vl);
  // expected-error@-1 {{call to undeclared function '__riscv_vmmacc_vv_i32m1_lm1'}}
  // expected-error@-2 {{returning 'int' from a function with incompatible result type}}
}
