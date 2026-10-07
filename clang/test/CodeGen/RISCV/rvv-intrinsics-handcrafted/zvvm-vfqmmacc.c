// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -target-feature +zvfh \
// RUN:   -target-feature +experimental-zvvfp16fp64mm \
// RUN:   -O2 -emit-llvm %s -o - | FileCheck %s

#pragma clang riscv intrinsic zvvm_vector

#include <riscv_vector.h>

// Quad-widening (W=4) FP MAC: EEW(A/B) = SEW(C)/4, so the only covered
// combination is FP16 inputs into an FP64 accumulator (OFP4/OFP8 inputs are
// deferred until 4-/8-bit FP vector types exist). BF16 inputs are selected
// by vtype.altfmt_A/altfmt_B at execution time. Spot-check representative
// (EMUL_C, LMUL) cells.

// CHECK-LABEL: define dso_local <vscale x 1 x double> @test_vfqmmacc_vv_f64m1
// CHECK-SAME:    (<vscale x 1 x double> [[VD:%.*]], <vscale x 4 x half> [[VS1:%.*]], <vscale x 4 x half> [[VS2:%.*]], i64 noundef [[VL:%.*]]) {{.*}} {
// CHECK:         [[TMP0:%.*]] = tail call <vscale x 1 x double> @llvm.riscv.vfqmmacc.nxv1f64.nxv4f16.i64(<vscale x 1 x double> [[VD]], <vscale x 4 x half> [[VS1]], <vscale x 4 x half> [[VS2]], i64 [[VL]])
// CHECK-NEXT:    ret <vscale x 1 x double> [[TMP0]]
//
vfloat64m1_t test_vfqmmacc_vv_f64m1(vfloat64m1_t vd, vfloat16m1_t vs1,
                                    vfloat16m1_t vs2, size_t vl) {
  return __riscv_vfqmmacc_vv_f64m1(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 8 x double> @test_vfqmmacc_vv_f64m8_lm4
// CHECK-SAME:    (<vscale x 8 x double> [[VD:%.*]], <vscale x 16 x half> [[VS1:%.*]], <vscale x 16 x half> [[VS2:%.*]], i64 noundef [[VL:%.*]]) {{.*}} {
// CHECK:         [[TMP0:%.*]] = tail call <vscale x 8 x double> @llvm.riscv.vfqmmacc.nxv8f64.nxv16f16.i64(<vscale x 8 x double> [[VD]], <vscale x 16 x half> [[VS1]], <vscale x 16 x half> [[VS2]], i64 [[VL]])
// CHECK-NEXT:    ret <vscale x 8 x double> [[TMP0]]
//
vfloat64m8_t test_vfqmmacc_vv_f64m8_lm4(vfloat64m8_t vd, vfloat16m4_t vs1,
                                        vfloat16m4_t vs2, size_t vl) {
  return __riscv_vfqmmacc_vv_f64m8_lm4(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 2 x double> @test_vfqmmacc_vv_f64m2_lm2
// CHECK-SAME:    (<vscale x 2 x double> [[VD:%.*]], <vscale x 8 x half> [[VS1:%.*]], <vscale x 8 x half> [[VS2:%.*]], i64 noundef [[VL:%.*]]) {{.*}} {
// CHECK:         [[TMP0:%.*]] = tail call <vscale x 2 x double> @llvm.riscv.vfqmmacc.nxv2f64.nxv8f16.i64(<vscale x 2 x double> [[VD]], <vscale x 8 x half> [[VS1]], <vscale x 8 x half> [[VS2]], i64 [[VL]])
// CHECK-NEXT:    ret <vscale x 2 x double> [[TMP0]]
//
vfloat64m2_t test_vfqmmacc_vv_f64m2_lm2(vfloat64m2_t vd, vfloat16m2_t vs1,
                                        vfloat16m2_t vs2, size_t vl) {
  return __riscv_vfqmmacc_vv_f64m2_lm2(vd, vs1, vs2, vl);
}
