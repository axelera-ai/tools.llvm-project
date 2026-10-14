// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 \
// RUN:   -target-feature +v -target-feature +experimental-zvvmm \
// RUN:   -target-feature +experimental-zvvmtls \
// RUN:   -target-feature +experimental-zvvmttls \
// RUN:   -O2 -emit-llvm %s -o - | FileCheck %s

// Masked tile load/store builtins (`_m`, and `_L{N}_m` composed with the
// per-instruction lambda override; `_m` is always last per the IME spec's
// canonical suffix order). The mask is the leading C argument with the
// standard RVV mask-ratio type vbool{SEW/LMUL}_t. The masked load carries
// no maskedoff argument: the passthru is poison and the trailing policy
// operand of the IR intrinsic is the TAMA bits (3) — architecturally, tile
// loads always leave inactive and tail elements undisturbed.

#pragma clang riscv intrinsic zvvm_vector

#include <riscv_vector.h>

// CHECK-LABEL: define dso_local <vscale x 2 x i32> @test_vmtl_m_i32m1
// CHECK:         tail call <vscale x 2 x i32> @llvm.riscv.vmtl.mask.nxv2i32.p0.i64(<vscale x 2 x i32> poison, ptr %{{.*}}, i64 %{{.*}}, <vscale x 2 x i1> %{{.*}}, i64 %{{.*}}, i64 3)
vint32m1_t test_vmtl_m_i32m1(vbool32_t mask, const int *ptr, size_t ld,
                             size_t vl) {
  return __riscv_vmtl_v_i32m1_m(mask, ptr, ld, vl);
}

// Mask ratio follows SEW/LMUL: i32 at LMUL=4 uses vbool8_t.
//
// CHECK-LABEL: define dso_local <vscale x 8 x i32> @test_vmtl_m_i32m4
// CHECK:         tail call <vscale x 8 x i32> @llvm.riscv.vmtl.mask.nxv8i32.p0.i64(<vscale x 8 x i32> poison, ptr %{{.*}}, i64 %{{.*}}, <vscale x 8 x i1> %{{.*}}, i64 %{{.*}}, i64 3)
vint32m4_t test_vmtl_m_i32m4(vbool8_t mask, const int *ptr, size_t ld,
                             size_t vl) {
  return __riscv_vmtl_v_i32m4_m(mask, ptr, ld, vl);
}

// Masked order-preserving store: builtin (mask, ptr, ld, value, vl) maps to
// the IR shape (value, ptr, ld, mask, vl); no policy operand on stores.
//
// CHECK-LABEL: define dso_local void @test_vmts_m_i32m1
// CHECK:         tail call void @llvm.riscv.vmts.mask.nxv2i32.p0.i64(<vscale x 2 x i32> %{{.*}}, ptr %{{.*}}, i64 %{{.*}}, <vscale x 2 x i1> %{{.*}}, i64 %{{.*}})
void test_vmts_m_i32m1(vbool32_t mask, int *ptr, size_t ld, vint32m1_t v,
                       size_t vl) {
  __riscv_vmts_v_i32m1_m(mask, ptr, ld, v, vl);
}

// Masked transposing load, FP element type.
//
// CHECK-LABEL: define dso_local <vscale x 2 x float> @test_vmttl_m_f32m1
// CHECK:         tail call <vscale x 2 x float> @llvm.riscv.vmttl.mask.nxv2f32.p0.i64(<vscale x 2 x float> poison, ptr %{{.*}}, i64 %{{.*}}, <vscale x 2 x i1> %{{.*}}, i64 %{{.*}}, i64 3)
vfloat32m1_t test_vmttl_m_f32m1(vbool32_t mask, const float *ptr, size_t ld,
                                size_t vl) {
  return __riscv_vmttl_v_f32m1_m(mask, ptr, ld, vl);
}

// Masked transposing store at LMUL=4: f64 at LMUL=4 gives mask ratio
// 64/4 = 16, i.e. vbool16_t.
//
// CHECK-LABEL: define dso_local void @test_vmtts_m_f64m4
// CHECK:         tail call void @llvm.riscv.vmtts.mask.nxv4f64.p0.i64(<vscale x 4 x double> %{{.*}}, ptr %{{.*}}, i64 %{{.*}}, <vscale x 4 x i1> %{{.*}}, i64 %{{.*}})
void test_vmtts_m_f64m4(vbool16_t mask, double *ptr, size_t ld,
                        vfloat64m4_t v, size_t vl) {
  __riscv_vmtts_v_f64m4_m(mask, ptr, ld, v, vl);
}

// `_L{N}` and `_m` compose, `_m` last: lambda override in the instruction
// immediate plus element masking.
//
// CHECK-LABEL: define dso_local <vscale x 2 x i32> @test_vmtl_L4_m
// CHECK:         tail call <vscale x 2 x i32> @llvm.riscv.vmtl.l4.mask.nxv2i32.p0.i64(<vscale x 2 x i32> poison, ptr %{{.*}}, i64 %{{.*}}, <vscale x 2 x i1> %{{.*}}, i64 %{{.*}}, i64 3)
vint32m1_t test_vmtl_L4_m(vbool32_t mask, const int *ptr, size_t ld,
                          size_t vl) {
  return __riscv_vmtl_v_i32m1_L4_m(mask, ptr, ld, vl);
}

// CHECK-LABEL: define dso_local void @test_vmtts_L2_m_i16m2
// CHECK:         tail call void @llvm.riscv.vmtts.l2.mask.nxv8i16.p0.i64(<vscale x 8 x i16> %{{.*}}, ptr %{{.*}}, i64 %{{.*}}, <vscale x 8 x i1> %{{.*}}, i64 %{{.*}})
void test_vmtts_L2_m_i16m2(vbool8_t mask, short *ptr, size_t ld,
                           vint16m2_t v, size_t vl) {
  __riscv_vmtts_v_i16m2_L2_m(mask, ptr, ld, v, vl);
}
