// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -target-feature +experimental-zvvi8i32mm -target-feature +experimental-zvvi16i64mm \
// RUN:   -O2 -emit-llvm %s -o - | FileCheck %s

#pragma clang riscv intrinsic zvvm_vector

#include <riscv_vector.h>

// Quad-widening (W=4) integer MAC: EEW(A/B) = SEW(C)/4. Spot-check
// representative (SEW, EMUL_C, LMUL) cells.

// CHECK-LABEL: define dso_local <vscale x 2 x i32> @test_vqmmacc_vv_i32m1
// CHECK-SAME:    (<vscale x 2 x i32> [[VD:%.*]], <vscale x 8 x i8> [[VS1:%.*]], <vscale x 8 x i8> [[VS2:%.*]], i64 noundef [[VL:%.*]]) {{.*}} {
// CHECK:         [[TMP0:%.*]] = tail call <vscale x 2 x i32> @llvm.riscv.vqmmacc.nxv2i32.nxv8i8.i64(<vscale x 2 x i32> [[VD]], <vscale x 8 x i8> [[VS1]], <vscale x 8 x i8> [[VS2]], i64 [[VL]])
// CHECK-NEXT:    ret <vscale x 2 x i32> [[TMP0]]
//
vint32m1_t test_vqmmacc_vv_i32m1(vint32m1_t vd, vint8m1_t vs1,
                                 vint8m1_t vs2, size_t vl) {
  return __riscv_vqmmacc_vv_i32m1(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 4 x i32> @test_vqmmacc_vv_i32m2_lm8
// CHECK-SAME:    (<vscale x 4 x i32> [[VD:%.*]], <vscale x 64 x i8> [[VS1:%.*]], <vscale x 64 x i8> [[VS2:%.*]], i64 noundef [[VL:%.*]]) {{.*}} {
// CHECK:         [[TMP0:%.*]] = tail call <vscale x 4 x i32> @llvm.riscv.vqmmacc.nxv4i32.nxv64i8.i64(<vscale x 4 x i32> [[VD]], <vscale x 64 x i8> [[VS1]], <vscale x 64 x i8> [[VS2]], i64 [[VL]])
// CHECK-NEXT:    ret <vscale x 4 x i32> [[TMP0]]
//
vint32m2_t test_vqmmacc_vv_i32m2_lm8(vint32m2_t vd, vint8m8_t vs1,
                                     vint8m8_t vs2, size_t vl) {
  return __riscv_vqmmacc_vv_i32m2_lm8(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 4 x i64> @test_vqmmacc_vv_i64m4
// CHECK-SAME:    (<vscale x 4 x i64> [[VD:%.*]], <vscale x 4 x i16> [[VS1:%.*]], <vscale x 4 x i16> [[VS2:%.*]], i64 noundef [[VL:%.*]]) {{.*}} {
// CHECK:         [[TMP0:%.*]] = tail call <vscale x 4 x i64> @llvm.riscv.vqmmacc.nxv4i64.nxv4i16.i64(<vscale x 4 x i64> [[VD]], <vscale x 4 x i16> [[VS1]], <vscale x 4 x i16> [[VS2]], i64 [[VL]])
// CHECK-NEXT:    ret <vscale x 4 x i64> [[TMP0]]
//
vint64m4_t test_vqmmacc_vv_i64m4(vint64m4_t vd, vint16m1_t vs1,
                                 vint16m1_t vs2, size_t vl) {
  return __riscv_vqmmacc_vv_i64m4(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 1 x i64> @test_vqmmacc_vv_i64m1_lm2
// CHECK-SAME:    (<vscale x 1 x i64> [[VD:%.*]], <vscale x 8 x i16> [[VS1:%.*]], <vscale x 8 x i16> [[VS2:%.*]], i64 noundef [[VL:%.*]]) {{.*}} {
// CHECK:         [[TMP0:%.*]] = tail call <vscale x 1 x i64> @llvm.riscv.vqmmacc.nxv1i64.nxv8i16.i64(<vscale x 1 x i64> [[VD]], <vscale x 8 x i16> [[VS1]], <vscale x 8 x i16> [[VS2]], i64 [[VL]])
// CHECK-NEXT:    ret <vscale x 1 x i64> [[TMP0]]
//
vint64m1_t test_vqmmacc_vv_i64m1_lm2(vint64m1_t vd, vint16m2_t vs1,
                                     vint16m2_t vs2, size_t vl) {
  return __riscv_vqmmacc_vv_i64m1_lm2(vd, vs1, vs2, vl);
}
