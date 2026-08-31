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

// ─── Widening MAC cells ─────────────────────────────────────────────────

vint32m2_t w_ok(vint32m2_t vd, vint16m4_t vs1, vint16m4_t vs2, size_t vl) {
  return __riscv_vwmmacc_vv_i32m2_lm4(vd, vs1, vs2, vl); // ok
}

vint32m2_t w_bad_input_sew(vint32m2_t vd, vint32m4_t vs1, vint32m4_t vs2,
                           size_t vl) {
  // vwmmacc inputs must be SEW(C)/2 = i16; passing i32 inputs must diagnose.
  return __riscv_vwmmacc_vv_i32m2_lm4(vd, vs1, vs2, vl);
  // expected-error@-1 {{passing 'vint32m4_t'}}
  // expected-note@-2 {{passing argument to parameter here}}
}

vint32m2_t w_bad_input_lmul(vint32m2_t vd, vint16m1_t vs1, vint16m1_t vs2,
                            size_t vl) {
  // The _lm4 cell pins vs1/vs2 to LMUL=4 (= __rvv_int16m4_t); passing m1
  // must diagnose.
  return __riscv_vwmmacc_vv_i32m2_lm4(vd, vs1, vs2, vl);
  // expected-error@-1 {{passing 'vint16m1_t'}}
  // expected-note@-2 {{passing argument to parameter here}}
}

vint32m4_t w_bad_c_emul_c(vint32m2_t vd, vint16m1_t vs1, vint16m1_t vs2,
                          size_t vl) {
  // Builtin expects vd at EMUL_C=4 (= __rvv_int32m4_t); passing m2 must
  // diagnose.
  return __riscv_vwmmacc_vv_i32m4(vd, vs1, vs2, vl);
  // expected-error@-1 {{passing 'vint32m2_t'}}
  // expected-note@-2 {{passing argument to parameter here}}
}

vint64m1_t q_ok(vint64m1_t vd, vint16m8_t vs1, vint16m8_t vs2, size_t vl) {
  return __riscv_vqmmacc_vv_i64m1_lm8(vd, vs1, vs2, vl); // ok
}

vint32m1_t q_bad_input_sew(vint32m1_t vd, vint16m1_t vs1, vint16m1_t vs2,
                           size_t vl) {
  // vqmmacc inputs must be SEW(C)/4 = i8; passing i16 inputs must diagnose.
  return __riscv_vqmmacc_vv_i32m1(vd, vs1, vs2, vl);
  // expected-error@-1 {{passing 'vint16m1_t'}}
  // expected-note@-2 {{passing argument to parameter here}}
}

vint64m1_t no_v8w_i32(vint32m1_t vd, vint8m1_t vs1, vint8m1_t vs2, size_t vl) {
  // v8wmmacc with an i32 accumulator would need Int4 inputs — no 4-bit
  // element type exists, so no such builtin is declared.
  return __riscv_v8wmmacc_vv_i32m1(vd, vs1, vs2, vl);
  // expected-error@-1 {{call to undeclared function '__riscv_v8wmmacc_vv_i32m1'}}
  // expected-error@-2 {{returning 'int' from a function with incompatible result type}}
}

// --- Sign-combination variants (Phase 5) ---

vuint32m2_t sign_ok_uu(vuint32m2_t vd, vuint32m1_t vs1, vuint32m1_t vs2,
                       size_t vl) {
  return __riscv_vmmacc_vv_u32m2(vd, vs1, vs2, vl); // ok
}

vint32m2_t sign_ok_su_lm2(vint32m2_t vd, vint32m2_t vs1, vuint32m2_t vs2,
                          size_t vl) {
  return __riscv_vmmacc_vv_i32m2_su_lm2(vd, vs1, vs2, vl); // ok
}

vint32m2_t sign_bad_su_vs2(vint32m2_t vd, vint32m1_t vs1, vint32m1_t vs2,
                           size_t vl) {
  // _su expects an UNSIGNED vs2; passing vint32m1_t must diagnose.
  return __riscv_vmmacc_vv_i32m2_su(vd, vs1, vs2, vl);
  // expected-error@-1 {{passing 'vint32m1_t'}}
  // expected-note@-2 {{passing argument to parameter here}}
}

vint32m2_t sign_bad_us_vs1(vint32m2_t vd, vint32m1_t vs1, vint32m1_t vs2,
                           size_t vl) {
  // _us expects an UNSIGNED vs1.
  return __riscv_vmmacc_vv_i32m2_us(vd, vs1, vs2, vl);
  // expected-error@-1 {{passing 'vint32m1_t'}}
  // expected-note@-2 {{passing argument to parameter here}}
}

vuint32m2_t sign_bad_uu_vd(vint32m2_t vd, vuint32m1_t vs1, vuint32m1_t vs2,
                           size_t vl) {
  // The u-token accumulator variant expects an unsigned vd.
  return __riscv_vmmacc_vv_u32m2(vd, vs1, vs2, vl);
  // expected-error@-1 {{passing 'vint32m2_t'}}
  // expected-note@-2 {{passing argument to parameter here}}
}

vint32m2_t no_uu_su_combo(vuint32m2_t vd, vint32m1_t vs1, vuint32m1_t vs2,
                          size_t vl) {
  // No `u{sew}..._su` spelling exists: mixed-sign variants use the signed
  // accumulator token per the spec.
  return __riscv_vmmacc_vv_u32m2_su(vd, vs1, vs2, vl);
  // expected-error@-1 {{call to undeclared function '__riscv_vmmacc_vv_u32m2_su'}}
  // expected-error@-2 {{returning 'int' from a function with incompatible result type}}
}

vint32m2_t widening_sign_ok(vint32m2_t vd, vuint16m1_t vs1, vint16m1_t vs2,
                            size_t vl) {
  return __riscv_vwmmacc_vv_i32m2_us(vd, vs1, vs2, vl); // ok
}

vint32m2_t widening_sign_bad_eew(vint32m2_t vd, vint32m1_t vs1,
                                 vuint32m1_t vs2, size_t vl) {
  // vwmmacc _su expects SEW/2 inputs (i16/u16), not full-SEW i32/u32.
  return __riscv_vwmmacc_vv_i32m2_su(vd, vs1, vs2, vl);
  // expected-error@-1 {{passing 'vint32m1_t'}}
  // expected-note@-2 {{passing argument to parameter here}}
}

vint32m1_t no_su_lm1(vint32m1_t vd, vint32m1_t vs1, vuint32m1_t vs2,
                     size_t vl) {
  // _lm1 stays nonexistent for the sign variants too.
  return __riscv_vmmacc_vv_i32m1_su_lm1(vd, vs1, vs2, vl);
  // expected-error@-1 {{call to undeclared function '__riscv_vmmacc_vv_i32m1_su_lm1'}}
  // expected-error@-2 {{returning 'int' from a function with incompatible result type}}
}
