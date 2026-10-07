// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -target-feature +experimental-zvvi8i64mm \
// RUN:   -O2 -emit-llvm %s -o - | FileCheck %s

#pragma clang riscv intrinsic zvvm_vector

#include <riscv_vector.h>

// 8x-widening (W=8) integer MAC: EEW(A/B) = SEW(C)/8, so the only covered
// combination is Int8 inputs into an Int64 accumulator (Int4 -> Int32 is
// deferred until 4-bit element types exist). Spot-check representative
// (EMUL_C, LMUL) cells.

// CHECK-LABEL: define dso_local <vscale x 1 x i64> @test_v8wmmacc_vv_i64m1
// CHECK-SAME:    (<vscale x 1 x i64> [[VD:%.*]], <vscale x 8 x i8> [[VS1:%.*]], <vscale x 8 x i8> [[VS2:%.*]], i64 noundef [[VL:%.*]]) {{.*}} {
// CHECK:         [[TMP0:%.*]] = tail call <vscale x 1 x i64> @llvm.riscv.v8wmmacc.nxv1i64.nxv8i8.i64(<vscale x 1 x i64> [[VD]], <vscale x 8 x i8> [[VS1]], <vscale x 8 x i8> [[VS2]], i64 [[VL]])
// CHECK-NEXT:    ret <vscale x 1 x i64> [[TMP0]]
//
vint64m1_t test_v8wmmacc_vv_i64m1(vint64m1_t vd, vint8m1_t vs1,
                                  vint8m1_t vs2, size_t vl) {
  return __riscv_v8wmmacc_vv_i64m1(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 8 x i64> @test_v8wmmacc_vv_i64m8_lm2
// CHECK-SAME:    (<vscale x 8 x i64> [[VD:%.*]], <vscale x 16 x i8> [[VS1:%.*]], <vscale x 16 x i8> [[VS2:%.*]], i64 noundef [[VL:%.*]]) {{.*}} {
// CHECK:         [[TMP0:%.*]] = tail call <vscale x 8 x i64> @llvm.riscv.v8wmmacc.nxv8i64.nxv16i8.i64(<vscale x 8 x i64> [[VD]], <vscale x 16 x i8> [[VS1]], <vscale x 16 x i8> [[VS2]], i64 [[VL]])
// CHECK-NEXT:    ret <vscale x 8 x i64> [[TMP0]]
//
vint64m8_t test_v8wmmacc_vv_i64m8_lm2(vint64m8_t vd, vint8m2_t vs1,
                                      vint8m2_t vs2, size_t vl) {
  return __riscv_v8wmmacc_vv_i64m8_lm2(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 2 x i64> @test_v8wmmacc_vv_i64m2_lm8
// CHECK-SAME:    (<vscale x 2 x i64> [[VD:%.*]], <vscale x 64 x i8> [[VS1:%.*]], <vscale x 64 x i8> [[VS2:%.*]], i64 noundef [[VL:%.*]]) {{.*}} {
// CHECK:         [[TMP0:%.*]] = tail call <vscale x 2 x i64> @llvm.riscv.v8wmmacc.nxv2i64.nxv64i8.i64(<vscale x 2 x i64> [[VD]], <vscale x 64 x i8> [[VS1]], <vscale x 64 x i8> [[VS2]], i64 [[VL]])
// CHECK-NEXT:    ret <vscale x 2 x i64> [[TMP0]]
//
vint64m2_t test_v8wmmacc_vv_i64m2_lm8(vint64m2_t vd, vint8m8_t vs1,
                                      vint8m8_t vs2, size_t vl) {
  return __riscv_v8wmmacc_vv_i64m2_lm8(vd, vs1, vs2, vl);
}
