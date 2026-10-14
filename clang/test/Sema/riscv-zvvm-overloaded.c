// Overloaded short forms: negative cases.
//
// The spec's short forms resolve solely from argument types. Consequences
// checked here:
//   * Unmasked (TA) tile loads vary only in the return type, so their short
//     form is inherently unresolvable — every call is ambiguous (this
//     mirrors the upstream unmasked-`vle` behavior). Use the long name.
//   * The bare mnemonic without the `.v` / `.vv` token is not a name the
//     spec defines — it must stay undeclared.
//   * `_L{N}` lambda-override variants share the default-lambda C
//     signature, so they are reachable only through their long names; the
//     short name resolves to the dynamic-lambda variant (checked in the
//     CodeGen test) and no `_L{N}`-qualified short spelling exists.
//   * The integer-input MX MACs keep the `_bs{N}` qualifier: a bare
//     `{mnemonic}_vv` short form would be ambiguous between the two block
//     sizes and is not provided.
//
// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -target-feature +zvfhmin \
// RUN:   -target-feature +experimental-zvvi32mm \
// RUN:   -target-feature +experimental-zvvxni8fp16mm \
// RUN:   -target-feature +experimental-zvvmtls \
// RUN:   -fsyntax-only -verify %s

#pragma clang riscv intrinsic zvvm_vector

#include <riscv_vector.h>

vint32m1_t ta_load_short(const int32_t *base, size_t ld, size_t vl) {
  return __riscv_vmtl_v(base, ld, vl); // expected-error {{call to '__riscv_vmtl_v' is ambiguous}} expected-note + {{candidate function}}
}

vint32m1_t bare_mnemonic(const int32_t *base, size_t ld, size_t vl) {
  return __riscv_vmtl(base, ld, vl); // expected-error {{call to undeclared function '__riscv_vmtl'}} expected-error {{returning 'int' from a function with incompatible result type 'vint32m1_t'}}
}

vint32m1_t lambda_qualified_short(const int32_t *base, size_t ld, size_t vl) {
  return __riscv_vmtl_v_L4(base, ld, vl); // expected-error {{call to undeclared function '__riscv_vmtl_v_L4'}} expected-error {{returning 'int' from a function with incompatible result type 'vint32m1_t'}}
}

vfloat16m1_t mx_without_bs(vfloat16m1_t vd, vint8m1_t vs1, vint8m1_t vs2,
                           vuint16m1_t v0, size_t vl) {
  return __riscv_vfwimmacc_vv(vd, vs1, vs2, v0, vl); // expected-error {{call to undeclared function '__riscv_vfwimmacc_vv'}} expected-error {{returning 'int' from a function with incompatible result type 'vfloat16m1_t'}}
}

vint32m1_t wrong_types_no_candidate(vint32m1_t vd, vint32m2_t vs1,
                                    vint32m4_t vs2, size_t vl) {
  // vs1/vs2 LMULs disagree: no cell matches.
  return __riscv_vmmacc_vv(vd, vs1, vs2, vl); // expected-error {{no matching function for call to '__riscv_vmmacc_vv'}} expected-note + {{candidate function not viable}}
}

// OFP8 masked tile loads have no short form: (vbool8_t, const uint8_t *) is
// shared with the u8m1 cell, which keeps the short call resolving to u8m1.
vfloat8e4m3m1_t ofp8_masked_load_short(vbool8_t mask, const uint8_t *base,
                                       size_t ld, size_t vl) {
  return __riscv_vmtl_v(mask, base, ld, vl); // expected-error {{returning '__rvv_uint8m1_t' from a function with incompatible result type 'vfloat8e4m3m1_t' (aka '__rvv_float8e4m3m1_t')}}
}

// Consequently no `_tu` short form for OFP8 loads either (the TA/_tu/_m
// variants of one record share its OverloadedName); use the long name.
vfloat8e4m3m1_t ofp8_tu_load_short(vfloat8e4m3m1_t pt, const uint8_t *base,
                                   size_t ld, size_t vl) {
  return __riscv_vmtl_v_tu(pt, base, ld, vl); // expected-error {{no matching function for call to '__riscv_vmtl_v_tu'}} expected-note + {{candidate function not viable}}
}
