// NOTE: Assertions were written by hand following the zvvm-vmmacc.c style.
// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -target-feature +experimental-zvvmm \
// RUN:   -O2 -emit-llvm %s -o - | FileCheck %s

#pragma clang riscv intrinsic zvvm_vector

#include <riscv_vector.h>

// Integer sign-combination variants (spec suffix order
// `{type-suffix}[_su|_us][_lm{N}]`). All four combos lower to the same
// signless IR intrinsic — vint and vuint share the LLVM vector type; the
// architectural signedness of A and B is selected by vtype.altfmt_A /
// altfmt_B, programmed separately via __riscv_vsetvl_matrix. These variants
// exist so user code type-checks against the data it actually has.

// CHECK-LABEL: define dso_local <vscale x 8 x i8> @test_vmmacc_vv_u8m1
// CHECK-SAME:    (<vscale x 8 x i8> [[VD:%.*]], <vscale x 8 x i8> [[VS1:%.*]], <vscale x 8 x i8> [[VS2:%.*]], i64 noundef [[VL:%.*]]) {{.*}} {
// CHECK:         [[TMP0:%.*]] = tail call <vscale x 8 x i8> @llvm.riscv.vmmacc.nxv8i8.nxv8i8.i64(<vscale x 8 x i8> [[VD]], <vscale x 8 x i8> [[VS1]], <vscale x 8 x i8> [[VS2]], i64 [[VL]])
// CHECK-NEXT:    ret <vscale x 8 x i8> [[TMP0]]
//
vuint8m1_t test_vmmacc_vv_u8m1(vuint8m1_t vd, vuint8m1_t vs1,
                               vuint8m1_t vs2, size_t vl) {
  return __riscv_vmmacc_vv_u8m1(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 8 x i32> @test_vmmacc_vv_u32m4_lm8
// CHECK-SAME:    (<vscale x 8 x i32> [[VD:%.*]], <vscale x 16 x i32> [[VS1:%.*]], <vscale x 16 x i32> [[VS2:%.*]], i64 noundef [[VL:%.*]]) {{.*}} {
// CHECK:         [[TMP0:%.*]] = tail call <vscale x 8 x i32> @llvm.riscv.vmmacc.nxv8i32.nxv16i32.i64(<vscale x 8 x i32> [[VD]], <vscale x 16 x i32> [[VS1]], <vscale x 16 x i32> [[VS2]], i64 [[VL]])
// CHECK-NEXT:    ret <vscale x 8 x i32> [[TMP0]]
//
vuint32m4_t test_vmmacc_vv_u32m4_lm8(vuint32m4_t vd, vuint32m8_t vs1,
                                     vuint32m8_t vs2, size_t vl) {
  return __riscv_vmmacc_vv_u32m4_lm8(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 4 x i32> @test_vmmacc_vv_i32m2_su
// CHECK-SAME:    (<vscale x 4 x i32> [[VD:%.*]], <vscale x 2 x i32> [[VS1:%.*]], <vscale x 2 x i32> [[VS2:%.*]], i64 noundef [[VL:%.*]]) {{.*}} {
// CHECK:         [[TMP0:%.*]] = tail call <vscale x 4 x i32> @llvm.riscv.vmmacc.nxv4i32.nxv2i32.i64(<vscale x 4 x i32> [[VD]], <vscale x 2 x i32> [[VS1]], <vscale x 2 x i32> [[VS2]], i64 [[VL]])
// CHECK-NEXT:    ret <vscale x 4 x i32> [[TMP0]]
//
vint32m2_t test_vmmacc_vv_i32m2_su(vint32m2_t vd, vint32m1_t vs1,
                                   vuint32m1_t vs2, size_t vl) {
  return __riscv_vmmacc_vv_i32m2_su(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 4 x i16> @test_vmmacc_vv_i16m1_us_lm4
// CHECK-SAME:    (<vscale x 4 x i16> [[VD:%.*]], <vscale x 16 x i16> [[VS1:%.*]], <vscale x 16 x i16> [[VS2:%.*]], i64 noundef [[VL:%.*]]) {{.*}} {
// CHECK:         [[TMP0:%.*]] = tail call <vscale x 4 x i16> @llvm.riscv.vmmacc.nxv4i16.nxv16i16.i64(<vscale x 4 x i16> [[VD]], <vscale x 16 x i16> [[VS1]], <vscale x 16 x i16> [[VS2]], i64 [[VL]])
// CHECK-NEXT:    ret <vscale x 4 x i16> [[TMP0]]
//
vint16m1_t test_vmmacc_vv_i16m1_us_lm4(vint16m1_t vd, vuint16m4_t vs1,
                                       vint16m4_t vs2, size_t vl) {
  return __riscv_vmmacc_vv_i16m1_us_lm4(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 4 x i32> @test_vwmmacc_vv_u32m2
// CHECK-SAME:    (<vscale x 4 x i32> [[VD:%.*]], <vscale x 4 x i16> [[VS1:%.*]], <vscale x 4 x i16> [[VS2:%.*]], i64 noundef [[VL:%.*]]) {{.*}} {
// CHECK:         [[TMP0:%.*]] = tail call <vscale x 4 x i32> @llvm.riscv.vwmmacc.nxv4i32.nxv4i16.i64(<vscale x 4 x i32> [[VD]], <vscale x 4 x i16> [[VS1]], <vscale x 4 x i16> [[VS2]], i64 [[VL]])
// CHECK-NEXT:    ret <vscale x 4 x i32> [[TMP0]]
//
vuint32m2_t test_vwmmacc_vv_u32m2(vuint32m2_t vd, vuint16m1_t vs1,
                                  vuint16m1_t vs2, size_t vl) {
  return __riscv_vwmmacc_vv_u32m2(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 4 x i32> @test_vwmmacc_vv_i32m2_su_lm4
// CHECK-SAME:    (<vscale x 4 x i32> [[VD:%.*]], <vscale x 16 x i16> [[VS1:%.*]], <vscale x 16 x i16> [[VS2:%.*]], i64 noundef [[VL:%.*]]) {{.*}} {
// CHECK:         [[TMP0:%.*]] = tail call <vscale x 4 x i32> @llvm.riscv.vwmmacc.nxv4i32.nxv16i16.i64(<vscale x 4 x i32> [[VD]], <vscale x 16 x i16> [[VS1]], <vscale x 16 x i16> [[VS2]], i64 [[VL]])
// CHECK-NEXT:    ret <vscale x 4 x i32> [[TMP0]]
//
vint32m2_t test_vwmmacc_vv_i32m2_su_lm4(vint32m2_t vd, vint16m4_t vs1,
                                        vuint16m4_t vs2, size_t vl) {
  return __riscv_vwmmacc_vv_i32m2_su_lm4(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 1 x i64> @test_vqmmacc_vv_u64m1_lm2
// CHECK-SAME:    (<vscale x 1 x i64> [[VD:%.*]], <vscale x 8 x i16> [[VS1:%.*]], <vscale x 8 x i16> [[VS2:%.*]], i64 noundef [[VL:%.*]]) {{.*}} {
// CHECK:         [[TMP0:%.*]] = tail call <vscale x 1 x i64> @llvm.riscv.vqmmacc.nxv1i64.nxv8i16.i64(<vscale x 1 x i64> [[VD]], <vscale x 8 x i16> [[VS1]], <vscale x 8 x i16> [[VS2]], i64 [[VL]])
// CHECK-NEXT:    ret <vscale x 1 x i64> [[TMP0]]
//
vuint64m1_t test_vqmmacc_vv_u64m1_lm2(vuint64m1_t vd, vuint16m2_t vs1,
                                      vuint16m2_t vs2, size_t vl) {
  return __riscv_vqmmacc_vv_u64m1_lm2(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 16 x i32> @test_vqmmacc_vv_i32m8_us
// CHECK-SAME:    (<vscale x 16 x i32> [[VD:%.*]], <vscale x 8 x i8> [[VS1:%.*]], <vscale x 8 x i8> [[VS2:%.*]], i64 noundef [[VL:%.*]]) {{.*}} {
// CHECK:         [[TMP0:%.*]] = tail call <vscale x 16 x i32> @llvm.riscv.vqmmacc.nxv16i32.nxv8i8.i64(<vscale x 16 x i32> [[VD]], <vscale x 8 x i8> [[VS1]], <vscale x 8 x i8> [[VS2]], i64 [[VL]])
// CHECK-NEXT:    ret <vscale x 16 x i32> [[TMP0]]
//
vint32m8_t test_vqmmacc_vv_i32m8_us(vint32m8_t vd, vuint8m1_t vs1,
                                    vint8m1_t vs2, size_t vl) {
  return __riscv_vqmmacc_vv_i32m8_us(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 4 x i64> @test_v8wmmacc_vv_u64m4_lm2
// CHECK-SAME:    (<vscale x 4 x i64> [[VD:%.*]], <vscale x 16 x i8> [[VS1:%.*]], <vscale x 16 x i8> [[VS2:%.*]], i64 noundef [[VL:%.*]]) {{.*}} {
// CHECK:         [[TMP0:%.*]] = tail call <vscale x 4 x i64> @llvm.riscv.v8wmmacc.nxv4i64.nxv16i8.i64(<vscale x 4 x i64> [[VD]], <vscale x 16 x i8> [[VS1]], <vscale x 16 x i8> [[VS2]], i64 [[VL]])
// CHECK-NEXT:    ret <vscale x 4 x i64> [[TMP0]]
//
vuint64m4_t test_v8wmmacc_vv_u64m4_lm2(vuint64m4_t vd, vuint8m2_t vs1,
                                       vuint8m2_t vs2, size_t vl) {
  return __riscv_v8wmmacc_vv_u64m4_lm2(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 1 x i64> @test_v8wmmacc_vv_i64m1_su
// CHECK-SAME:    (<vscale x 1 x i64> [[VD:%.*]], <vscale x 8 x i8> [[VS1:%.*]], <vscale x 8 x i8> [[VS2:%.*]], i64 noundef [[VL:%.*]]) {{.*}} {
// CHECK:         [[TMP0:%.*]] = tail call <vscale x 1 x i64> @llvm.riscv.v8wmmacc.nxv1i64.nxv8i8.i64(<vscale x 1 x i64> [[VD]], <vscale x 8 x i8> [[VS1]], <vscale x 8 x i8> [[VS2]], i64 [[VL]])
// CHECK-NEXT:    ret <vscale x 1 x i64> [[TMP0]]
//
vint64m1_t test_v8wmmacc_vv_i64m1_su(vint64m1_t vd, vint8m1_t vs1,
                                     vuint8m1_t vs2, size_t vl) {
  return __riscv_v8wmmacc_vv_i64m1_su(vd, vs1, vs2, vl);
}
