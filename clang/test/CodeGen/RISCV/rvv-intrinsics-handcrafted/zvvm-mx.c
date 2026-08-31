// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -target-feature +zvfhmin \
// RUN:   -target-feature +experimental-zvvxni8fp16mm \
// RUN:   -target-feature +experimental-zvvxni8fp32mm \
// RUN:   -target-feature +experimental-zvvxni8fp64mm \
// RUN:   -O2 -emit-llvm %s -o - | FileCheck %s

#pragma clang riscv intrinsic zvvm_vector

#include <riscv_vector.h>

// Integer-input microscaled MACs (MXINT8 inputs, FP accumulator). The name
// carries the input-type token `_i8m{LMUL}` (so no `_lm{N}` qualifier) and a
// trailing `_bs{32,16}` block-size record. The paired E8M0 block scales are
// passed as vuint16m1_t and lowered to the intrinsic's scale operand; the
// `_bs{N}` suffix lowers to the trailing bs immarg (0 = BS32, 1 = BS16),
// which is a semantic record only — vtype.bs carries the architectural
// block size. The Zvvxn* (BS=16) features imply their Zvvx* siblings, so
// the RUN line enables only the BS=16 flags.

// CHECK-LABEL: define dso_local <vscale x 4 x half> @test_vfwimmacc_vv_f16m1_i8m1_bs32
// CHECK-SAME:    (<vscale x 4 x half> [[VD:%.*]], <vscale x 8 x i8> [[VS1:%.*]], <vscale x 8 x i8> [[VS2:%.*]], <vscale x 4 x i16> [[V0:%.*]], i64 noundef [[VL:%.*]]) {{.*}} {
// CHECK:         [[TMP0:%.*]] = tail call <vscale x 4 x half> @llvm.riscv.vfwimmacc.nxv4f16.nxv8i8.i64(<vscale x 4 x half> [[VD]], <vscale x 8 x i8> [[VS1]], <vscale x 8 x i8> [[VS2]], <vscale x 4 x i16> [[V0]], i64 0, i64 [[VL]])
// CHECK-NEXT:    ret <vscale x 4 x half> [[TMP0]]
//
vfloat16m1_t test_vfwimmacc_vv_f16m1_i8m1_bs32(vfloat16m1_t vd, vint8m1_t vs1,
                                               vint8m1_t vs2, vuint16m1_t v0,
                                               size_t vl) {
  return __riscv_vfwimmacc_vv_f16m1_i8m1_bs32(vd, vs1, vs2, v0, vl);
}

// CHECK-LABEL: define dso_local <vscale x 4 x half> @test_vfwimmacc_vv_f16m1_i8m1_bs16
// CHECK:         call <vscale x 4 x half> @llvm.riscv.vfwimmacc.nxv4f16.nxv8i8.i64(<vscale x 4 x half> {{.*}}, <vscale x 8 x i8> {{.*}}, <vscale x 8 x i8> {{.*}}, <vscale x 4 x i16> {{.*}}, i64 1, i64 {{.*}})
//
vfloat16m1_t test_vfwimmacc_vv_f16m1_i8m1_bs16(vfloat16m1_t vd, vint8m1_t vs1,
                                               vint8m1_t vs2, vuint16m1_t v0,
                                               size_t vl) {
  return __riscv_vfwimmacc_vv_f16m1_i8m1_bs16(vd, vs1, vs2, v0, vl);
}

// EMUL_C = 8 with LMUL = 2 inputs: independent register-group axes.
// CHECK-LABEL: define dso_local <vscale x 32 x half> @test_vfwimmacc_vv_f16m8_i8m2_bs32
// CHECK:         call <vscale x 32 x half> @llvm.riscv.vfwimmacc.nxv32f16.nxv16i8.i64(<vscale x 32 x half> {{.*}}, <vscale x 16 x i8> {{.*}}, <vscale x 16 x i8> {{.*}}, <vscale x 4 x i16> {{.*}}, i64 0, i64 {{.*}})
//
vfloat16m8_t test_vfwimmacc_vv_f16m8_i8m2_bs32(vfloat16m8_t vd, vint8m2_t vs1,
                                               vint8m2_t vs2, vuint16m1_t v0,
                                               size_t vl) {
  return __riscv_vfwimmacc_vv_f16m8_i8m2_bs32(vd, vs1, vs2, v0, vl);
}

// CHECK-LABEL: define dso_local <vscale x 4 x float> @test_vfqimmacc_vv_f32m2_i8m4_bs16
// CHECK:         call <vscale x 4 x float> @llvm.riscv.vfqimmacc.nxv4f32.nxv32i8.i64(<vscale x 4 x float> {{.*}}, <vscale x 32 x i8> {{.*}}, <vscale x 32 x i8> {{.*}}, <vscale x 4 x i16> {{.*}}, i64 1, i64 {{.*}})
//
vfloat32m2_t test_vfqimmacc_vv_f32m2_i8m4_bs16(vfloat32m2_t vd, vint8m4_t vs1,
                                               vint8m4_t vs2, vuint16m1_t v0,
                                               size_t vl) {
  return __riscv_vfqimmacc_vv_f32m2_i8m4_bs16(vd, vs1, vs2, v0, vl);
}

// CHECK-LABEL: define dso_local <vscale x 8 x double> @test_vf8wimmacc_vv_f64m8_i8m8_bs32
// CHECK:         call <vscale x 8 x double> @llvm.riscv.vf8wimmacc.nxv8f64.nxv64i8.i64(<vscale x 8 x double> {{.*}}, <vscale x 64 x i8> {{.*}}, <vscale x 64 x i8> {{.*}}, <vscale x 4 x i16> {{.*}}, i64 0, i64 {{.*}})
//
vfloat64m8_t test_vf8wimmacc_vv_f64m8_i8m8_bs32(vfloat64m8_t vd, vint8m8_t vs1,
                                                vint8m8_t vs2, vuint16m1_t v0,
                                                size_t vl) {
  return __riscv_vf8wimmacc_vv_f64m8_i8m8_bs32(vd, vs1, vs2, v0, vl);
}

// CHECK-LABEL: define dso_local <vscale x 1 x double> @test_vf8wimmacc_vv_f64m1_i8m1_bs16
// CHECK:         call <vscale x 1 x double> @llvm.riscv.vf8wimmacc.nxv1f64.nxv8i8.i64(<vscale x 1 x double> {{.*}}, <vscale x 8 x i8> {{.*}}, <vscale x 8 x i8> {{.*}}, <vscale x 4 x i16> {{.*}}, i64 1, i64 {{.*}})
//
vfloat64m1_t test_vf8wimmacc_vv_f64m1_i8m1_bs16(vfloat64m1_t vd, vint8m1_t vs1,
                                                vint8m1_t vs2, vuint16m1_t v0,
                                                size_t vl) {
  return __riscv_vf8wimmacc_vv_f64m1_i8m1_bs16(vd, vs1, vs2, v0, vl);
}
