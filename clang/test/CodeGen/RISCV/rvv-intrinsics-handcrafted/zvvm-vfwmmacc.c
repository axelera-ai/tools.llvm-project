// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -target-feature +zvfh \
// RUN:   -target-feature +experimental-zvvfmm \
// RUN:   -O2 -emit-llvm %s -o - | FileCheck %s

#pragma clang riscv intrinsic zvvm_vector

#include <riscv_vector.h>

// Widening (W=2) FP MAC: EEW(A/B) = SEW(C)/2. FP16 inputs into an FP32
// accumulator and FP32 inputs into an FP64 accumulator (OFP8 inputs are
// deferred until FP8 vector types exist). BF16 inputs are selected by
// vtype.altfmt_A/altfmt_B at execution time, not by the C type. Spot-check
// representative (SEW, EMUL_C, LMUL) cells.

// CHECK-LABEL: define dso_local <vscale x 2 x float> @test_vfwmmacc_vv_f32m1
// CHECK-SAME:    (<vscale x 2 x float> [[VD:%.*]], <vscale x 4 x half> [[VS1:%.*]], <vscale x 4 x half> [[VS2:%.*]], i64 noundef [[VL:%.*]]) {{.*}} {
// CHECK:         [[TMP0:%.*]] = tail call <vscale x 2 x float> @llvm.riscv.vfwmmacc.nxv2f32.nxv4f16.i64(<vscale x 2 x float> [[VD]], <vscale x 4 x half> [[VS1]], <vscale x 4 x half> [[VS2]], i64 [[VL]])
// CHECK-NEXT:    ret <vscale x 2 x float> [[TMP0]]
//
vfloat32m1_t test_vfwmmacc_vv_f32m1(vfloat32m1_t vd, vfloat16m1_t vs1,
                                    vfloat16m1_t vs2, size_t vl) {
  return __riscv_vfwmmacc_vv_f32m1(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 8 x float> @test_vfwmmacc_vv_f32m4_lm2
// CHECK-SAME:    (<vscale x 8 x float> [[VD:%.*]], <vscale x 8 x half> [[VS1:%.*]], <vscale x 8 x half> [[VS2:%.*]], i64 noundef [[VL:%.*]]) {{.*}} {
// CHECK:         [[TMP0:%.*]] = tail call <vscale x 8 x float> @llvm.riscv.vfwmmacc.nxv8f32.nxv8f16.i64(<vscale x 8 x float> [[VD]], <vscale x 8 x half> [[VS1]], <vscale x 8 x half> [[VS2]], i64 [[VL]])
// CHECK-NEXT:    ret <vscale x 8 x float> [[TMP0]]
//
vfloat32m4_t test_vfwmmacc_vv_f32m4_lm2(vfloat32m4_t vd, vfloat16m2_t vs1,
                                        vfloat16m2_t vs2, size_t vl) {
  return __riscv_vfwmmacc_vv_f32m4_lm2(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 2 x double> @test_vfwmmacc_vv_f64m2
// CHECK-SAME:    (<vscale x 2 x double> [[VD:%.*]], <vscale x 2 x float> [[VS1:%.*]], <vscale x 2 x float> [[VS2:%.*]], i64 noundef [[VL:%.*]]) {{.*}} {
// CHECK:         [[TMP0:%.*]] = tail call <vscale x 2 x double> @llvm.riscv.vfwmmacc.nxv2f64.nxv2f32.i64(<vscale x 2 x double> [[VD]], <vscale x 2 x float> [[VS1]], <vscale x 2 x float> [[VS2]], i64 [[VL]])
// CHECK-NEXT:    ret <vscale x 2 x double> [[TMP0]]
//
vfloat64m2_t test_vfwmmacc_vv_f64m2(vfloat64m2_t vd, vfloat32m1_t vs1,
                                    vfloat32m1_t vs2, size_t vl) {
  return __riscv_vfwmmacc_vv_f64m2(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 1 x double> @test_vfwmmacc_vv_f64m1_lm8
// CHECK-SAME:    (<vscale x 1 x double> [[VD:%.*]], <vscale x 16 x float> [[VS1:%.*]], <vscale x 16 x float> [[VS2:%.*]], i64 noundef [[VL:%.*]]) {{.*}} {
// CHECK:         [[TMP0:%.*]] = tail call <vscale x 1 x double> @llvm.riscv.vfwmmacc.nxv1f64.nxv16f32.i64(<vscale x 1 x double> [[VD]], <vscale x 16 x float> [[VS1]], <vscale x 16 x float> [[VS2]], i64 [[VL]])
// CHECK-NEXT:    ret <vscale x 1 x double> [[TMP0]]
//
vfloat64m1_t test_vfwmmacc_vv_f64m1_lm8(vfloat64m1_t vd, vfloat32m8_t vs1,
                                        vfloat32m8_t vs2, size_t vl) {
  return __riscv_vfwmmacc_vv_f64m1_lm8(vd, vs1, vs2, vl);
}
