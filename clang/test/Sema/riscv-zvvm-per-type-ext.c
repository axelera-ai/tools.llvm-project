// Per-type IME extension gating (spec tbl-extensions).
//
// MAC builtins are gated on their per-type extension; the configuration
// primitives (__riscv_vsetvl_matrix etc.) are gated on the family flag,
// which every per-type extension implies. Clang expands the implication
// when building the feature map, so a per-type flag alone is sufficient
// for both — while the family flag alone must NOT enable any per-type
// MAC builtin.
//
// RUN: %clang_cc1 -triple riscv64 -target-feature +v \
// RUN:   -target-feature +experimental-zvvi32mm \
// RUN:   -fsyntax-only -verify=pertype %s
// RUN: %clang_cc1 -triple riscv64 -target-feature +v \
// RUN:   -target-feature +experimental-zvvmm \
// RUN:   -fsyntax-only -verify=familyonly %s

#pragma clang riscv intrinsic zvvm_vector

typedef __SIZE_TYPE__ size_t;
typedef __rvv_int8m1_t vint8m1_t;
typedef __rvv_int32m1_t vint32m1_t;

// The implied family flag makes the config primitive available in both
// configurations.
size_t config(size_t avl) {
  return __riscv_vsetvl_matrix(avl, 2, 0, 1, 1, 2, 0, 0, 0); // no diagnostic
}

vint32m1_t mac_i32(vint32m1_t vd, vint32m1_t vs1, vint32m1_t vs2, size_t vl) {
  return __riscv_vmmacc_vv_i32m1(vd, vs1, vs2, vl);
  // familyonly-error@-1 {{builtin requires at least one of the following extensions: experimental-zvvi32mm}}
}

vint8m1_t mac_i8(vint8m1_t vd, vint8m1_t vs1, vint8m1_t vs2, size_t vl) {
  // A different per-type extension is NOT implied by zvvi32mm (nor by the
  // family flag).
  return __riscv_vmmacc_vv_i8m1(vd, vs1, vs2, vl);
  // pertype-error@-1 {{builtin requires at least one of the following extensions: experimental-zvvi8mm}}
  // familyonly-error@-2 {{builtin requires at least one of the following extensions: experimental-zvvi8mm}}
}
