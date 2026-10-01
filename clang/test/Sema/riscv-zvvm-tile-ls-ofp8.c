// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +v \
// RUN:   -target-feature +experimental-zvvmtls \
// RUN:   -target-feature +experimental-zvvmttls \
// RUN:   -fsyntax-only -verify %s

// OFP8 tile load/store builtins: negative cases.
//   * The natural (and `_as_e8`) forms take a uint8_t base; `_as_e{S}` takes
//     uint{S}_t. A pointer of another width is rejected.
//   * E4M3 and E5M2 are distinct types: passing one format's tile to the
//     other's store, or returning one from the other's load, is an error.
//   * `_as_e{S}` exists only for S ∈ {8,16,32,64} and only on the
//     order-preserving pair (no transposing `_as_e{S}`).
//   * OFP8 masked tile loads have no short form: their signature would
//     collide with the u8 masked load, so `__riscv_vmtl_v(vbool8_t, const
//     uint8_t *, ...)` keeps resolving to the u8 cell.

#pragma clang riscv intrinsic zvvm_vector

#include <riscv_vector.h>

vfloat8e4m3m1_t bad_ptr_load(const uint16_t *b16, size_t ld, size_t vl) {
  return __riscv_vmtl_v_f8e4m3m1(b16, ld, vl); // expected-error {{incompatible pointer types passing 'const uint16_t *' (aka 'const unsigned short *') to parameter of type 'const unsigned char *'}} expected-note {{passing argument to parameter here}}
}

vfloat8e4m3m1_t bad_ptr_as_e32(const uint8_t *b8, size_t ld, size_t vl) {
  return __riscv_vmtl_v_f8e4m3m1_as_e32(b8, ld, vl); // expected-error {{incompatible pointer types passing 'const uint8_t *' (aka 'const unsigned char *') to parameter of type 'const unsigned int *'}} expected-note {{passing argument to parameter here}}
}

void bad_ptr_store(uint32_t *b32, size_t ld, vfloat8e5m2m1_t v, size_t vl) {
  __riscv_vmts_v_f8e5m2m1(b32, ld, v, vl); // expected-error {{incompatible pointer types passing 'uint32_t *' (aka 'unsigned int *') to parameter of type 'unsigned char *'}} expected-note {{passing argument to parameter here}}
}

void bad_ptr_store_as_e16(uint32_t *b32, size_t ld, vfloat8e5m2m1_t v,
                          size_t vl) {
  __riscv_vmts_v_f8e5m2m1_as_e16(b32, ld, v, vl); // expected-error {{incompatible pointer types passing 'uint32_t *' (aka 'unsigned int *') to parameter of type 'unsigned short *'}} expected-note {{passing argument to parameter here}}
}

void wrong_format_store(uint8_t *b, size_t ld, vfloat8e4m3m1_t e4,
                        size_t vl) {
  __riscv_vmts_v_f8e5m2m1(b, ld, e4, vl); // expected-error {{passing 'vfloat8e4m3m1_t' (aka '__rvv_float8e4m3m1_t') to parameter of incompatible type '__rvv_float8e5m2m1_t'}} expected-note {{passing argument to parameter here}}
}

void wrong_format_store_L4(uint8_t *b, size_t ld, vfloat8e5m2m2_t e5,
                           size_t vl) {
  // The `_L{N}` long name is an overload set (unmasked + masked).
  __riscv_vmtts_v_f8e4m3m2_L4(b, ld, e5, vl); // expected-error {{no matching function for call to '__riscv_vmtts_v_f8e4m3m2_L4'}} expected-note {{no known conversion from 'vfloat8e5m2m2_t' (aka '__rvv_float8e5m2m2_t') to '__rvv_float8e4m3m2_t' for 3rd argument}} expected-note {{requires 5 arguments, but 4 were provided}}
}

vfloat8e5m2m1_t wrong_format_load(const uint32_t *b, size_t ld, size_t vl) {
  return __riscv_vmtl_v_f8e4m3m1_as_e32(b, ld, vl); // expected-error {{returning '__rvv_float8e4m3m1_t' from a function with incompatible result type 'vfloat8e5m2m1_t' (aka '__rvv_float8e5m2m1_t')}}
}

vfloat8e4m3m1_t bad_as_e4(const uint8_t *b, size_t ld, size_t vl) {
  return __riscv_vmtl_v_f8e4m3m1_as_e4(b, ld, vl); // expected-error {{call to undeclared function '__riscv_vmtl_v_f8e4m3m1_as_e4'}} expected-error {{returning 'int' from a function with incompatible result type 'vfloat8e4m3m1_t' (aka '__rvv_float8e4m3m1_t')}}
}

vfloat8e4m3m1_t bad_as_e128(const uint8_t *b, size_t ld, size_t vl) {
  return __riscv_vmtl_v_f8e4m3m1_as_e128(b, ld, vl); // expected-error {{call to undeclared function '__riscv_vmtl_v_f8e4m3m1_as_e128'}} expected-error {{returning 'int' from a function with incompatible result type 'vfloat8e4m3m1_t' (aka '__rvv_float8e4m3m1_t')}}
}

vfloat8e4m3m1_t no_transposing_as_e(const uint32_t *b, size_t ld, size_t vl) {
  return __riscv_vmttl_v_f8e4m3m1_as_e32(b, ld, vl); // expected-error {{call to undeclared function '__riscv_vmttl_v_f8e4m3m1_as_e32'}} expected-error {{returning 'int' from a function with incompatible result type 'vfloat8e4m3m1_t' (aka '__rvv_float8e4m3m1_t')}}
}

void no_transposing_store_as_e(uint8_t *b, size_t ld, vfloat8e5m2m1_t v,
                               size_t vl) {
  __riscv_vmtts_v_f8e5m2m1_as_e8(b, ld, v, vl); // expected-error {{call to undeclared function '__riscv_vmtts_v_f8e5m2m1_as_e8'}}
}

vfloat8e4m3m1_t no_fractional_lmul(const uint8_t *b, size_t ld, size_t vl) {
  return __riscv_vmtl_v_f8e4m3mf2(b, ld, vl); // expected-error {{call to undeclared function '__riscv_vmtl_v_f8e4m3mf2'}} expected-error {{returning 'int' from a function with incompatible result type 'vfloat8e4m3m1_t' (aka '__rvv_float8e4m3m1_t')}}
}

vfloat8e4m3m1_t masked_short_is_u8(vbool8_t m, const uint8_t *b, size_t ld,
                                   size_t vl) {
  return __riscv_vmtl_v(m, b, ld, vl); // expected-error {{returning '__rvv_uint8m1_t' from a function with incompatible result type 'vfloat8e4m3m1_t' (aka '__rvv_float8e4m3m1_t')}}
}
