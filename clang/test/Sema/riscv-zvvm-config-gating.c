// REQUIRES: riscv-registered-target
// The vtype configuration / query primitives are available under ANY Zvvm
// family member (the vtype fields belong to the family, not to Zvvmm alone),
// so FP-only and tile-only configurations can program lambda / bs / altfmt.
// RUN: %clang_cc1 -triple riscv64 -target-feature +v \
// RUN:   -target-feature +experimental-zvvmm -fsyntax-only -verify=ok %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +v \
// RUN:   -target-feature +experimental-zvvfmm -fsyntax-only -verify=ok %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +v \
// RUN:   -target-feature +experimental-zvvmtls -fsyntax-only -verify=ok %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +v \
// RUN:   -target-feature +experimental-zvvmttls -fsyntax-only -verify=ok %s
// A per-type FP extension implies Zvvfmm, which is enough on its own.
// RUN: %clang_cc1 -triple riscv64 -target-feature +v \
// RUN:   -target-feature +experimental-zvvofp8fp32mm -fsyntax-only -verify=ok %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -fsyntax-only \
// RUN:   -verify=none %s

// ok-no-diagnostics

#include <riscv_vector.h>
#pragma clang riscv intrinsic zvvm_vector

size_t config(size_t avl) {
  size_t vl = __riscv_vsetvl_matrix(avl, 2, 0, 1, 1, 3, 0, 0, 0); // none-error {{builtin requires at least one of the following extensions: experimental-zvvmm, experimental-zvvfmm, experimental-zvvmtls, experimental-zvvmttls}}
  size_t l = __riscv_vsetlambda(4);  // none-error {{builtin requires at least one of the following extensions: experimental-zvvmm, experimental-zvvfmm, experimental-zvvmtls, experimental-zvvmttls}}
  size_t q = __riscv_ime_lambda();   // none-error {{builtin requires at least one of the following extensions: experimental-zvvmm, experimental-zvvfmm, experimental-zvvmtls, experimental-zvvmttls}}
  size_t v = __riscv_ime_vlen();     // none-error {{builtin requires at least one of the following extensions: experimental-zvvmm, experimental-zvvfmm, experimental-zvvmtls, experimental-zvvmttls}}
  return vl + l + q + v;
}
