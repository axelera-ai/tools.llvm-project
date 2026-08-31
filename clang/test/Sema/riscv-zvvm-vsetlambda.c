// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -target-feature +experimental-zvvmm \
// RUN:   -fsyntax-only -verify %s

// Sema validation for __riscv_vsetlambda: a compile-time-constant argument
// must be 0 (preserve-or-initialize) or a power of two in
// {1, 2, 4, 8, 16, 32, 64}. A runtime argument is accepted per the IME spec
// (an out-of-domain runtime value is undefined behavior).

#pragma clang riscv intrinsic zvvm_vector

#include <riscv_vector.h>

size_t accepts_zero(void)     { return __riscv_vsetlambda(0);  }
size_t accepts_l1(void)       { return __riscv_vsetlambda(1);  }
size_t accepts_l2(void)       { return __riscv_vsetlambda(2);  }
size_t accepts_l4(void)       { return __riscv_vsetlambda(4);  }
size_t accepts_l8(void)       { return __riscv_vsetlambda(8);  }
size_t accepts_l16(void)      { return __riscv_vsetlambda(16); }
size_t accepts_l32(void)      { return __riscv_vsetlambda(32); }
size_t accepts_l64(void)      { return __riscv_vsetlambda(64); }

size_t rejects_3(void)        { return __riscv_vsetlambda(3);  }   // expected-error {{lambda argument must be 0 (preserve-or-initialize) or one of 1, 2, 4, 8, 16, 32, 64}}
size_t rejects_5(void)        { return __riscv_vsetlambda(5);  }   // expected-error {{lambda argument must be 0 (preserve-or-initialize) or one of 1, 2, 4, 8, 16, 32, 64}}
size_t rejects_128(void)      { return __riscv_vsetlambda(128); }  // expected-error {{lambda argument must be 0 (preserve-or-initialize) or one of 1, 2, 4, 8, 16, 32, 64}}
size_t rejects_neg(void)      { return __riscv_vsetlambda(-1); }   // expected-error {{lambda argument must be 0 (preserve-or-initialize) or one of 1, 2, 4, 8, 16, 32, 64}}

// Runtime requests are legal per the spec (out-of-domain values are UB at
// execution time, not a compile error).
size_t accepts_runtime(size_t L) {
  return __riscv_vsetlambda(L);
}

// The geometry-query intrinsics take no arguments.
size_t query_lambda(void) { return __riscv_ime_lambda(); }
size_t query_vlen(void)   { return __riscv_ime_vlen(); }
