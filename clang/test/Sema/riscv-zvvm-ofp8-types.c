// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +v \
// RUN:   -target-feature +experimental-zvvfmm -mvscale-min=2 -mvscale-max=2 -fsyntax-only -verify=ok %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +v \
// RUN:   -target-feature +experimental-zvvmtls -mvscale-min=2 -mvscale-max=2 -fsyntax-only -verify=ok %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +v \
// RUN:   -target-feature +experimental-zvvmttls -mvscale-min=2 -mvscale-max=2 -fsyntax-only -verify=ok %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +v \
// RUN:   -target-feature +experimental-zvfofp8min -mvscale-min=2 -mvscale-max=2 -fsyntax-only -verify=ok %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +v \
// RUN:   -target-feature +experimental-zvvofp8fp16mm -mvscale-min=2 -mvscale-max=2 -fsyntax-only -verify=ok %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +v \
// RUN:   -fsyntax-only -verify=nofeat -DNOFEAT %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +zve32x \
// RUN:   -target-feature +experimental-zvvmtls -fsyntax-only -verify=zve32x \
// RUN:   -DZVE32X %s

// The IME (Zvvm) OFP8 vector types are distinct C types: they share the i8
// container with vint8/vuint8 in IR, but never convert implicitly to or from
// them, or between E4M3 and E5M2. They are sizeless and have no fixed-length
// (riscv_rvv_vector_bits) form. They are available with any extension that
// produces or consumes OFP8 vector data: Zvvfmm (implied by every OFP8 per-type
// IME extension), Zvvmtls, Zvvmttls or Zvfofp8min.

#include <riscv_vector.h>

#if defined(NOFEAT)
void nofeat(void) {
  vfloat8e4m3m1_t a; // nofeat-error {{RISC-V type 'vfloat8e4m3m1_t' (aka '__rvv_float8e4m3m1_t') requires the 'zvvfmm, zvvmtls, zvvmttls or zvfofp8min' extension}}
  vfloat8e5m2m16_t b; // nofeat-error {{RISC-V type 'vfloat8e5m2m16_t' (aka '__rvv_float8e5m2m16_t') requires the 'zvvfmm, zvvmtls, zvvmttls or zvfofp8min' extension}}
  vint8m1_t ok; // integer types stay available
}
#elif defined(ZVE32X)
void zve32x(void) {
  vfloat8e4m3m1_t a;
  vfloat8e4m3mf8_t b; // zve32x-error {{RISC-V type 'vfloat8e4m3mf8_t' (aka '__rvv_float8e4m3mf8_t') requires the 'zve64x' extension}}
}
#else
void distinct(vfloat8e4m3m1_t e4, vfloat8e5m2m1_t e5, vint8m1_t i8,
              vuint8m1_t u8) {
  vfloat8e4m3m1_t a = i8; // ok-error {{initializing 'vfloat8e4m3m1_t' (aka '__rvv_float8e4m3m1_t') with an expression of incompatible type 'vint8m1_t' (aka '__rvv_int8m1_t')}}
  vfloat8e4m3m1_t b = u8; // ok-error {{initializing 'vfloat8e4m3m1_t' (aka '__rvv_float8e4m3m1_t') with an expression of incompatible type 'vuint8m1_t' (aka '__rvv_uint8m1_t')}}
  vfloat8e4m3m1_t c = e5; // ok-error {{initializing 'vfloat8e4m3m1_t' (aka '__rvv_float8e4m3m1_t') with an expression of incompatible type 'vfloat8e5m2m1_t' (aka '__rvv_float8e5m2m1_t')}}
  vuint8m1_t d = e4;      // ok-error {{initializing 'vuint8m1_t' (aka '__rvv_uint8m1_t') with an expression of incompatible type 'vfloat8e4m3m1_t' (aka '__rvv_float8e4m3m1_t')}}
  vint8m1_t e = e5;       // ok-error {{initializing 'vint8m1_t' (aka '__rvv_int8m1_t') with an expression of incompatible type 'vfloat8e5m2m1_t' (aka '__rvv_float8e5m2m1_t')}}
  e4 = (vfloat8e4m3m1_t)u8; // ok-error {{used type 'vfloat8e4m3m1_t' (aka '__rvv_float8e4m3m1_t') where arithmetic or pointer type is required}}
  vfloat8e4m3m2_t f = e4; // ok-error {{initializing 'vfloat8e4m3m2_t' (aka '__rvv_float8e4m3m2_t') with an expression of incompatible type 'vfloat8e4m3m1_t' (aka '__rvv_float8e4m3m1_t')}}
  // The explicit, bit-preserving route.
  vfloat8e4m3m1_t g = __riscv_vreinterpret_v_u8m1_f8e4m3m1(u8);
  vfloat8e5m2m1_t h = __riscv_vreinterpret_f8e5m2m1(e4);
}

void sizeless(vfloat8e4m3m1_t e4) {
  (void)sizeof(e4); // ok-error {{invalid application of 'sizeof' to sizeless type 'vfloat8e4m3m1_t' (aka '__rvv_float8e4m3m1_t')}}
  vfloat8e4m3m1_t arr[2]; // ok-error {{array has sizeless element type 'vfloat8e4m3m1_t' (aka '__rvv_float8e4m3m1_t')}}
  (void)(e4 + e4); // ok-error {{invalid operands to binary expression ('vfloat8e4m3m1_t' (aka '__rvv_float8e4m3m1_t') and 'vfloat8e4m3m1_t')}}
}

struct S {
  vfloat8e5m2m1_t v; // ok-error {{field has sizeless type 'vfloat8e5m2m1_t' (aka '__rvv_float8e5m2m1_t')}}
};

vfloat8e4m3m1_t global; // ok-error {{non-local variable with sizeless type 'vfloat8e4m3m1_t' (aka '__rvv_float8e4m3m1_t')}}

// No fixed-length OFP8 types: they would be indistinguishable from the
// fixed-length vuint8 ones.
typedef vfloat8e4m3m1_t fixed_e4m3 __attribute__((riscv_rvv_vector_bits(128))); // ok-error {{'riscv_rvv_vector_bits' attribute applied to non-RVV type 'vfloat8e4m3m1_t' (aka '__rvv_float8e4m3m1_t')}}

void builtins(vfloat8e4m3m8_t e4m8, vint8m4_t i8m4) {
  // vget/vset only accept the matching OFP8 types.
  vfloat8e4m3m4_t a = __riscv_vget_v_f8e4m3m8_f8e4m3m4(e4m8, 0);
  vint8m4_t b = __riscv_vget_v_f8e4m3m8_f8e4m3m4(e4m8, 0); // ok-error {{initializing 'vint8m4_t' (aka '__rvv_int8m4_t') with an expression of incompatible type '__rvv_float8e4m3m4_t'}}
  e4m8 = __riscv_vset_v_f8e4m3m4_f8e4m3m8(e4m8, 0, i8m4); // ok-error {{passing 'vint8m4_t' (aka '__rvv_int8m4_t') to parameter of incompatible type '__rvv_float8e4m3m4_t'}}
  // ok-note@* {{passing argument to parameter here}}
  (void)__riscv_vget_v_f8e4m3m8_f8e4m3m4(e4m8, 2); // ok-error {{argument value 2 is outside the valid range [0, 1]}}
}
#endif
