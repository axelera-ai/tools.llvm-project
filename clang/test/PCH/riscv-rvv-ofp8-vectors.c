// REQUIRES: riscv-registered-target
// RUN: rm -rf %t
// RUN: split-file %s %t

// The IME (Zvvm) OFP8 vector types round-trip through a precompiled header
// as their own predefined types (not as the vuint8 types sharing their i8
// container).
// RUN: %clang_cc1 -triple riscv64-linux-gnu -target-feature +v \
// RUN:   -target-feature +experimental-zvvfmm -emit-pch -o %t/test_pch.pch \
// RUN:   %t/test_pch.h
// RUN: %clang_cc1 -triple riscv64-linux-gnu -target-feature +v \
// RUN:   -target-feature +experimental-zvvfmm -include-pch %t/test_pch.pch \
// RUN:   -fsyntax-only -verify %t/test_pch_src.c

//--- test_pch.h
#include <riscv_vector.h>
vfloat8e4m3m4_t ofp8_low(vfloat8e4m3m8_t v);
vfloat8e5m2m16_t ofp8_pair(vfloat8e5m2m8_t lo, vfloat8e5m2m8_t hi);

//--- test_pch_src.c
vfloat8e4m3m4_t good(vfloat8e4m3m8_t v) { return ofp8_low(v); }

vuint8m4_t bad(vfloat8e4m3m8_t v) {
  return ofp8_low(v); // expected-error {{returning 'vfloat8e4m3m4_t' (aka '__rvv_float8e4m3m4_t') from a function with incompatible result type 'vuint8m4_t' (aka '__rvv_uint8m4_t')}}
}

vfloat8e5m2m16_t pair(vfloat8e5m2m8_t lo, vfloat8e5m2m8_t hi) {
  return ofp8_pair(lo, hi);
}
