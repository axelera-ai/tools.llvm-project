// OFP8-input FP MAC builtins: naming, type, and per-type gating checks.
//
//   * Every name follows {type-suffix}[_{inputA}[_{inputB}]]: the OFP8 input
//     token carries the A/B LMUL, so no `_lm{N}` qualifier exists on the
//     OFP8-input cells; a mixed-format cell names A then B; vfmmacc names
//     its inputs only when they differ from the OFP8 accumulator format.
//   * The vs1/vs2 formats, LMUL, and the vd EMUL_C are encoded in the
//     argument types; a mismatch diagnoses on the long form and finds no
//     candidate on the short form.
//   * Each row is gated on its own per-type extension (Zvvofp8fp16mm etc.).
//
// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -target-feature +zvfh \
// RUN:   -target-feature +zvfbfmin -target-feature +experimental-zvvofp8mm \
// RUN:   -target-feature +experimental-zvvofp8fp16mm \
// RUN:   -target-feature +experimental-zvvofp8bf16mm \
// RUN:   -target-feature +experimental-zvvofp8fp32mm \
// RUN:   -target-feature +experimental-zvvofp8fp64mm \
// RUN:   -fsyntax-only -verify=all %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -target-feature +zvfh \
// RUN:   -target-feature +zvfbfmin \
// RUN:   -target-feature +experimental-zvvofp8fp16mm \
// RUN:   -fsyntax-only -verify=fp16only -DGATING %s

#pragma clang riscv intrinsic zvvm_vector

#include <riscv_vector.h>

#ifndef GATING

// --- Wrong input format on the long form. -------------------------------

vfloat16m4_t wrong_fmt_b(vfloat16m4_t vd, vfloat8e4m3m1_t a,
                         vfloat8e5m2m1_t b, size_t vl) {
  // E4M3 x E4M3 cell called with an E5M2 B operand.
  return __riscv_vfwmmacc_vv_f16m4_f8e4m3m1(vd, a, b, vl); // all-error {{passing 'vfloat8e5m2m1_t' (aka '__rvv_float8e5m2m1_t') to parameter of incompatible type '__rvv_float8e4m3m1_t'}} all-note {{passing argument to parameter here}}
}

vfloat32m2_t wrong_fmt_order(vfloat32m2_t vd, vfloat8e4m3m1_t a,
                             vfloat8e5m2m1_t b, size_t vl) {
  // The mixed cell's A/B order is part of the name: E5M2 x E4M3 here.
  return __riscv_vfqmmacc_vv_f32m2_f8e5m2m1_f8e4m3m1(vd, a, b, vl); // all-error {{passing 'vfloat8e4m3m1_t' (aka '__rvv_float8e4m3m1_t') to parameter of incompatible type '__rvv_float8e5m2m1_t'}} all-note {{passing argument to parameter here}}
}

vfloat16m4_t uint8_input(vfloat16m4_t vd, vuint8m1_t a, vuint8m1_t b,
                         size_t vl) {
  // OFP8 vectors are distinct from vuint8 even though both are i8 in IR.
  return __riscv_vfwmmacc_vv_f16m4_f8e4m3m1(vd, a, b, vl); // all-error {{passing 'vuint8m1_t' (aka '__rvv_uint8m1_t') to parameter of incompatible type '__rvv_float8e4m3m1_t'}} all-note {{passing argument to parameter here}}
}

vfloat8e4m3m1_t vfmmacc_wrong_acc(vfloat8e5m2m1_t vd, vfloat8e4m3m1_t a,
                                  vfloat8e4m3m1_t b, size_t vl) {
  // The OFP8 accumulator format is part of the type-suffix.
  return __riscv_vfmmacc_vv_f8e4m3m1(vd, a, b, vl); // all-error {{passing 'vfloat8e5m2m1_t' (aka '__rvv_float8e5m2m1_t') to parameter of incompatible type '__rvv_float8e4m3m1_t'}} all-note {{passing argument to parameter here}}
}

// --- Wrong LMUL / EMUL_C on the long form. ------------------------------

vfloat16m4_t wrong_lmul(vfloat16m4_t vd, vfloat8e4m3m2_t a,
                        vfloat8e4m3m2_t b, size_t vl) {
  return __riscv_vfwmmacc_vv_f16m4_f8e4m3m1(vd, a, b, vl); // all-error {{passing 'vfloat8e4m3m2_t' (aka '__rvv_float8e4m3m2_t') to parameter of incompatible type '__rvv_float8e4m3m1_t'}} all-note {{passing argument to parameter here}}
}

vbfloat16m1_t wrong_emul_c(vbfloat16m1_t vd, vfloat8e4m3m4_t a,
                           vfloat8e4m3m4_t b, size_t vl) {
  return __riscv_vfwmmacc_vv_bf16m2_f8e4m3m4(vd, a, b, vl); // all-error {{passing 'vbfloat16m1_t' (aka '__rvv_bfloat16m1_t') to parameter of incompatible type '__rvv_bfloat16m2_t'}} all-note {{passing argument to parameter here}}
}

vbfloat16m1_t bf16_vs_f16(vbfloat16m1_t vd, vfloat8e4m3m4_t a,
                          vfloat8e4m3m4_t b, size_t vl) {
  // FP16 and BF16 accumulators are distinct cells.
  return __riscv_vfwmmacc_vv_f16m1_f8e4m3m4(vd, a, b, vl); // all-error {{passing 'vbfloat16m1_t' (aka '__rvv_bfloat16m1_t') to parameter of incompatible type '__rvv_float16m1_t'}} all-note {{passing argument to parameter here}}
}

// --- Spellings the spec does not define. --------------------------------

void bad_names(vfloat16m1_t c16, vfloat8e4m3m4_t a, vfloat8e4m3m8_t c8,
               vfloat8e4m3m1_t a1, size_t vl) {
  // The input token carries LMUL; there is no `_lm{N}` qualifier.
  __riscv_vfwmmacc_vv_f16m1_f8e4m3m4_lm4(c16, a, a, vl); // all-error {{call to undeclared function '__riscv_vfwmmacc_vv_f16m1_f8e4m3m4_lm4'}}
  // ... nor an input-less OFP8-input widening name.
  __riscv_vfwmmacc_vv_f16m1_lm4(c16, a, a, vl); // all-error {{call to undeclared function '__riscv_vfwmmacc_vv_f16m1_lm4'}}
  // A = B repeats no token.
  __riscv_vfwmmacc_vv_f16m1_f8e4m3m4_f8e4m3m4(c16, a, a, vl); // all-error {{call to undeclared function '__riscv_vfwmmacc_vv_f16m1_f8e4m3m4_f8e4m3m4'}}
  // vfmmacc with inputs in the accumulator's own format has no input token.
  __riscv_vfmmacc_vv_f8e4m3m8_f8e4m3m1(c8, a1, a1, vl); // all-error {{call to undeclared function '__riscv_vfmmacc_vv_f8e4m3m8_f8e4m3m1'}}
  // Mixed-format vfmmacc names the inputs, not `_lm{N}`.
  __riscv_vfmmacc_vv_f8e4m3m8_f8e5m2m1_lm1(c8, a1, a1, vl); // all-error {{call to undeclared function '__riscv_vfmmacc_vv_f8e4m3m8_f8e5m2m1_lm1'}}
  // No OFP8 sign or `_alt` qualifiers.
  __riscv_vfwmmacc_vv_f16m1_f8e4m3m4_su(c16, a, a, vl); // all-error {{call to undeclared function '__riscv_vfwmmacc_vv_f16m1_f8e4m3m4_su'}}
}

// --- Short forms: argument types must match one cell. -------------------

vfloat32m2_t short_mixed_lmul(vfloat32m2_t vd, vfloat8e4m3m4_t a,
                              vfloat8e4m3m2_t b, size_t vl) {
  // A and B must share the LMUL.
  return __riscv_vfqmmacc_vv(vd, a, b, vl); // all-error {{no matching function for call to '__riscv_vfqmmacc_vv'}} all-note + {{candidate function not viable}}
}

vfloat64m1_t short_wrong_width(vfloat64m1_t vd, vfloat8e4m3m4_t a,
                               vfloat8e4m3m4_t b, size_t vl) {
  // OFP8 inputs into FP64 is vf8wmmacc (W=8), not vfqmmacc.
  return __riscv_vfqmmacc_vv(vd, a, b, vl); // all-error {{no matching function for call to '__riscv_vfqmmacc_vv'}} all-note + {{candidate function not viable}}
}

vfloat8e4m3m1_t short_ok(vfloat8e4m3m1_t vd, vfloat8e5m2m8_t a,
                         vfloat8e4m3m8_t b, size_t vl) {
  return __riscv_vfmmacc_vv(vd, a, b, vl);
}

#else // GATING

// Only Zvvofp8fp16mm is enabled: its cells are available, the sibling rows
// (BF16 / FP32 / FP64 accumulator, OFP8 accumulator) are not.

vfloat16m4_t gate_fp16(vfloat16m4_t vd, vfloat8e4m3m1_t a, vfloat8e5m2m1_t b,
                       size_t vl) {
  return __riscv_vfwmmacc_vv_f16m4_f8e4m3m1_f8e5m2m1(vd, a, b, vl);
}

vbfloat16m4_t gate_bf16(vbfloat16m4_t vd, vfloat8e4m3m1_t a,
                        vfloat8e4m3m1_t b, size_t vl) {
  return __riscv_vfwmmacc_vv_bf16m4_f8e4m3m1(vd, a, b, vl); // fp16only-error {{builtin requires at least one of the following extensions: experimental-zvvofp8bf16mm}}
}

vfloat32m2_t gate_fp32(vfloat32m2_t vd, vfloat8e4m3m4_t a, vfloat8e4m3m4_t b,
                       size_t vl) {
  return __riscv_vfqmmacc_vv_f32m2_f8e4m3m4(vd, a, b, vl); // fp16only-error {{builtin requires at least one of the following extensions: experimental-zvvofp8fp32mm}}
}

vfloat64m1_t gate_fp64(vfloat64m1_t vd, vfloat8e4m3m1_t a, vfloat8e4m3m1_t b,
                       size_t vl) {
  return __riscv_vf8wmmacc_vv_f64m1_f8e4m3m1(vd, a, b, vl); // fp16only-error {{builtin requires at least one of the following extensions: experimental-zvvofp8fp64mm}}
}

vfloat8e4m3m8_t gate_ofp8(vfloat8e4m3m8_t vd, vfloat8e4m3m1_t a,
                          vfloat8e4m3m1_t b, size_t vl) {
  return __riscv_vfmmacc_vv_f8e4m3m8(vd, a, b, vl); // fp16only-error {{builtin requires at least one of the following extensions: experimental-zvvofp8mm}}
}

vfloat32m2_t gate_fp32_short(vfloat32m2_t vd, vfloat8e4m3m4_t a,
                             vfloat8e4m3m4_t b, size_t vl) {
  // The short form of a gated cell is gated too.
  return __riscv_vfqmmacc_vv(vd, a, b, vl); // fp16only-error {{builtin requires at least one of the following extensions: experimental-zvvofp8fp32mm}}
}

#endif
