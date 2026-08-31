// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -target-feature +experimental-zvvi8i16mm -target-feature +experimental-zvvi16i32mm -target-feature +experimental-zvvi32i64mm \
// RUN:   -O2 -emit-llvm %s -o - | FileCheck %s

#pragma clang riscv intrinsic zvvm_vector

#include <riscv_vector.h>

// Widening (W=2) integer MAC: EEW(A/B) = SEW(C)/2. The type-suffix encodes
// the accumulator type and EMUL_C; `_lm{N}` (omitted for LMUL=1) encodes the
// A/B LMUL. Spot-check representative (SEW, EMUL_C, LMUL) cells.

// CHECK-LABEL: define dso_local <vscale x 2 x i32> @test_vwmmacc_vv_i32m1
// CHECK-SAME:    (<vscale x 2 x i32> [[VD:%.*]], <vscale x 4 x i16> [[VS1:%.*]], <vscale x 4 x i16> [[VS2:%.*]], i64 noundef [[VL:%.*]]) {{.*}} {
// CHECK:         [[TMP0:%.*]] = tail call <vscale x 2 x i32> @llvm.riscv.vwmmacc.nxv2i32.nxv4i16.i64(<vscale x 2 x i32> [[VD]], <vscale x 4 x i16> [[VS1]], <vscale x 4 x i16> [[VS2]], i64 [[VL]])
// CHECK-NEXT:    ret <vscale x 2 x i32> [[TMP0]]
//
vint32m1_t test_vwmmacc_vv_i32m1(vint32m1_t vd, vint16m1_t vs1,
                                 vint16m1_t vs2, size_t vl) {
  return __riscv_vwmmacc_vv_i32m1(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 4 x i32> @test_vwmmacc_vv_i32m2_lm4
// CHECK-SAME:    (<vscale x 4 x i32> [[VD:%.*]], <vscale x 16 x i16> [[VS1:%.*]], <vscale x 16 x i16> [[VS2:%.*]], i64 noundef [[VL:%.*]]) {{.*}} {
// CHECK:         [[TMP0:%.*]] = tail call <vscale x 4 x i32> @llvm.riscv.vwmmacc.nxv4i32.nxv16i16.i64(<vscale x 4 x i32> [[VD]], <vscale x 16 x i16> [[VS1]], <vscale x 16 x i16> [[VS2]], i64 [[VL]])
// CHECK-NEXT:    ret <vscale x 4 x i32> [[TMP0]]
//
vint32m2_t test_vwmmacc_vv_i32m2_lm4(vint32m2_t vd, vint16m4_t vs1,
                                     vint16m4_t vs2, size_t vl) {
  return __riscv_vwmmacc_vv_i32m2_lm4(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 16 x i32> @test_vwmmacc_vv_i32m8
// CHECK-SAME:    (<vscale x 16 x i32> [[VD:%.*]], <vscale x 4 x i16> [[VS1:%.*]], <vscale x 4 x i16> [[VS2:%.*]], i64 noundef [[VL:%.*]]) {{.*}} {
// CHECK:         [[TMP0:%.*]] = tail call <vscale x 16 x i32> @llvm.riscv.vwmmacc.nxv16i32.nxv4i16.i64(<vscale x 16 x i32> [[VD]], <vscale x 4 x i16> [[VS1]], <vscale x 4 x i16> [[VS2]], i64 [[VL]])
// CHECK-NEXT:    ret <vscale x 16 x i32> [[TMP0]]
//
vint32m8_t test_vwmmacc_vv_i32m8(vint32m8_t vd, vint16m1_t vs1,
                                 vint16m1_t vs2, size_t vl) {
  return __riscv_vwmmacc_vv_i32m8(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 16 x i16> @test_vwmmacc_vv_i16m4_lm2
// CHECK-SAME:    (<vscale x 16 x i16> [[VD:%.*]], <vscale x 16 x i8> [[VS1:%.*]], <vscale x 16 x i8> [[VS2:%.*]], i64 noundef [[VL:%.*]]) {{.*}} {
// CHECK:         [[TMP0:%.*]] = tail call <vscale x 16 x i16> @llvm.riscv.vwmmacc.nxv16i16.nxv16i8.i64(<vscale x 16 x i16> [[VD]], <vscale x 16 x i8> [[VS1]], <vscale x 16 x i8> [[VS2]], i64 [[VL]])
// CHECK-NEXT:    ret <vscale x 16 x i16> [[TMP0]]
//
vint16m4_t test_vwmmacc_vv_i16m4_lm2(vint16m4_t vd, vint8m2_t vs1,
                                     vint8m2_t vs2, size_t vl) {
  return __riscv_vwmmacc_vv_i16m4_lm2(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 1 x i64> @test_vwmmacc_vv_i64m1_lm8
// CHECK-SAME:    (<vscale x 1 x i64> [[VD:%.*]], <vscale x 16 x i32> [[VS1:%.*]], <vscale x 16 x i32> [[VS2:%.*]], i64 noundef [[VL:%.*]]) {{.*}} {
// CHECK:         [[TMP0:%.*]] = tail call <vscale x 1 x i64> @llvm.riscv.vwmmacc.nxv1i64.nxv16i32.i64(<vscale x 1 x i64> [[VD]], <vscale x 16 x i32> [[VS1]], <vscale x 16 x i32> [[VS2]], i64 [[VL]])
// CHECK-NEXT:    ret <vscale x 1 x i64> [[TMP0]]
//
vint64m1_t test_vwmmacc_vv_i64m1_lm8(vint64m1_t vd, vint32m8_t vs1,
                                     vint32m8_t vs2, size_t vl) {
  return __riscv_vwmmacc_vv_i64m1_lm8(vd, vs1, vs2, vl);
}
