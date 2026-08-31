// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +v \
// RUN:   -target-feature +experimental-zvvi8mm -target-feature +experimental-zvvi32mm \
// RUN:   -target-feature +experimental-zvvi64mm -target-feature +experimental-zvvfp16mm \
// RUN:   -target-feature +experimental-zvvfp32mm -target-feature +experimental-zvvfp64mm \
// RUN:   -target-feature +zvfh \
// RUN:   -O2 -emit-llvm %s -o - | FileCheck %s

#pragma clang riscv intrinsic zvvm_vector

#include <riscv_vector.h>

// EMUL_C = 16 (m16 accumulator) surface. The IME-only v<elt>m16_t types are
// a 16-register-aligned pair of M8 groups, lowered to
// target("riscv.vector.tuple", <vscale x 64 x i8>, 2). The MAC intrinsics use
// the *_m16 IR family (the C SEW is carried by vtype, so the tuple type is
// byte-typed); pair/unpair are pure tuple insert/extract with no vl.

// CHECK-LABEL: define dso_local target("riscv.vector.tuple", <vscale x 64 x i8>, 2) @test_vmmacc_vv_i32m16
// CHECK:         [[TMP0:%.*]] = tail call target("riscv.vector.tuple", <vscale x 64 x i8>, 2) @llvm.riscv.vmmacc.m16.triscv.vector.tuple_nxv64i8_2t.nxv2i32.i64(target("riscv.vector.tuple", <vscale x 64 x i8>, 2) [[VD:%.*]], <vscale x 2 x i32> [[VS1:%.*]], <vscale x 2 x i32> [[VS2:%.*]], i64 [[VL:%.*]])
// CHECK-NEXT:    ret target("riscv.vector.tuple", <vscale x 64 x i8>, 2) [[TMP0]]
//
vint32m16_t test_vmmacc_vv_i32m16(vint32m16_t vd, vint32m1_t vs1,
                                  vint32m1_t vs2, size_t vl) {
  return __riscv_vmmacc_vv_i32m16(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local target("riscv.vector.tuple", <vscale x 64 x i8>, 2) @test_vmmacc_vv_i8m16_lm8
// CHECK:         call target("riscv.vector.tuple", <vscale x 64 x i8>, 2) @llvm.riscv.vmmacc.m16.triscv.vector.tuple_nxv64i8_2t.nxv64i8.i64(target("riscv.vector.tuple", <vscale x 64 x i8>, 2) {{%.*}}, <vscale x 64 x i8> {{%.*}}, <vscale x 64 x i8> {{%.*}}, i64 {{%.*}})
//
vint8m16_t test_vmmacc_vv_i8m16_lm8(vint8m16_t vd, vint8m8_t vs1,
                                    vint8m8_t vs2, size_t vl) {
  return __riscv_vmmacc_vv_i8m16_lm8(vd, vs1, vs2, vl);
}

// Unsigned accumulator variant (uu sign combo).
// CHECK-LABEL: define dso_local target("riscv.vector.tuple", <vscale x 64 x i8>, 2) @test_vmmacc_vv_u64m16_lm2
// CHECK:         call target("riscv.vector.tuple", <vscale x 64 x i8>, 2) @llvm.riscv.vmmacc.m16.triscv.vector.tuple_nxv64i8_2t.nxv2i64.i64(target("riscv.vector.tuple", <vscale x 64 x i8>, 2) {{%.*}}, <vscale x 2 x i64> {{%.*}}, <vscale x 2 x i64> {{%.*}}, i64 {{%.*}})
//
vuint64m16_t test_vmmacc_vv_u64m16_lm2(vuint64m16_t vd, vuint64m2_t vs1,
                                       vuint64m2_t vs2, size_t vl) {
  return __riscv_vmmacc_vv_u64m16_lm2(vd, vs1, vs2, vl);
}

// Mixed-sign variant (_su): signed accumulator, signed A x unsigned B.
// CHECK-LABEL: define dso_local target("riscv.vector.tuple", <vscale x 64 x i8>, 2) @test_vmmacc_vv_i32m16_su_lm4
// CHECK:         call target("riscv.vector.tuple", <vscale x 64 x i8>, 2) @llvm.riscv.vmmacc.m16.triscv.vector.tuple_nxv64i8_2t.nxv8i32.i64(target("riscv.vector.tuple", <vscale x 64 x i8>, 2) {{%.*}}, <vscale x 8 x i32> {{%.*}}, <vscale x 8 x i32> {{%.*}}, i64 {{%.*}})
//
vint32m16_t test_vmmacc_vv_i32m16_su_lm4(vint32m16_t vd, vint32m4_t vs1,
                                         vuint32m4_t vs2, size_t vl) {
  return __riscv_vmmacc_vv_i32m16_su_lm4(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local target("riscv.vector.tuple", <vscale x 64 x i8>, 2) @test_vfmmacc_vv_f32m16
// CHECK:         call target("riscv.vector.tuple", <vscale x 64 x i8>, 2) @llvm.riscv.vfmmacc.m16.triscv.vector.tuple_nxv64i8_2t.nxv2f32.i64(target("riscv.vector.tuple", <vscale x 64 x i8>, 2) {{%.*}}, <vscale x 2 x float> {{%.*}}, <vscale x 2 x float> {{%.*}}, i64 {{%.*}})
//
vfloat32m16_t test_vfmmacc_vv_f32m16(vfloat32m16_t vd, vfloat32m1_t vs1,
                                     vfloat32m1_t vs2, size_t vl) {
  return __riscv_vfmmacc_vv_f32m16(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local target("riscv.vector.tuple", <vscale x 64 x i8>, 2) @test_vfmmacc_vv_f16m16_lm8
// CHECK:         call target("riscv.vector.tuple", <vscale x 64 x i8>, 2) @llvm.riscv.vfmmacc.m16.triscv.vector.tuple_nxv64i8_2t.nxv32f16.i64(target("riscv.vector.tuple", <vscale x 64 x i8>, 2) {{%.*}}, <vscale x 32 x half> {{%.*}}, <vscale x 32 x half> {{%.*}}, i64 {{%.*}})
//
vfloat16m16_t test_vfmmacc_vv_f16m16_lm8(vfloat16m16_t vd, vfloat16m8_t vs1,
                                         vfloat16m8_t vs2, size_t vl) {
  return __riscv_vfmmacc_vv_f16m16_lm8(vd, vs1, vs2, vl);
}

// Pair: (lo, hi) -> m16 value via two tuple inserts on poison. No vl operand.
// CHECK-LABEL: define dso_local target("riscv.vector.tuple", <vscale x 64 x i8>, 2) @test_ime_vpair_i32m16
// CHECK:         [[P0:%.*]] = tail call target("riscv.vector.tuple", <vscale x 64 x i8>, 2) @llvm.riscv.tuple.insert.triscv.vector.tuple_nxv64i8_2t.nxv16i32(target("riscv.vector.tuple", <vscale x 64 x i8>, 2) poison, <vscale x 16 x i32> [[LO:%.*]], i32 0)
// CHECK-NEXT:    [[P1:%.*]] = tail call target("riscv.vector.tuple", <vscale x 64 x i8>, 2) @llvm.riscv.tuple.insert.triscv.vector.tuple_nxv64i8_2t.nxv16i32(target("riscv.vector.tuple", <vscale x 64 x i8>, 2) [[P0]], <vscale x 16 x i32> [[HI:%.*]], i32 1)
// CHECK-NEXT:    ret target("riscv.vector.tuple", <vscale x 64 x i8>, 2) [[P1]]
//
vint32m16_t test_ime_vpair_i32m16(vint32m8_t lo, vint32m8_t hi) {
  return __riscv_ime_vpair_i32m16(lo, hi);
}

// CHECK-LABEL: define dso_local <vscale x 16 x float> @test_ime_vunpairlo_f32m16
// CHECK:         call <vscale x 16 x float> @llvm.riscv.tuple.extract.nxv16f32.triscv.vector.tuple_nxv64i8_2t(target("riscv.vector.tuple", <vscale x 64 x i8>, 2) {{%.*}}, i32 0)
//
vfloat32m8_t test_ime_vunpairlo_f32m16(vfloat32m16_t src) {
  return __riscv_ime_vunpairlo_f32m16(src);
}

// CHECK-LABEL: define dso_local <vscale x 8 x i64> @test_ime_vunpairhi_u64m16
// CHECK:         call <vscale x 8 x i64> @llvm.riscv.tuple.extract.nxv8i64.triscv.vector.tuple_nxv64i8_2t(target("riscv.vector.tuple", <vscale x 64 x i8>, 2) {{%.*}}, i32 1)
//
vuint64m8_t test_ime_vunpairhi_u64m16(vuint64m16_t src) {
  return __riscv_ime_vunpairhi_u64m16(src);
}

// Composed pair -> MAC -> unpair, the spec's load-pair-compute-unpair-store
// shape.
// CHECK-LABEL: define dso_local <vscale x 16 x i32> @test_pair_mac_unpair
// CHECK:         call target("riscv.vector.tuple", <vscale x 64 x i8>, 2) @llvm.riscv.tuple.insert.triscv.vector.tuple_nxv64i8_2t.nxv16i32(target("riscv.vector.tuple", <vscale x 64 x i8>, 2) poison
// CHECK:         call target("riscv.vector.tuple", <vscale x 64 x i8>, 2) @llvm.riscv.vmmacc.m16.triscv.vector.tuple_nxv64i8_2t.nxv2i32.i64
// CHECK:         call <vscale x 16 x i32> @llvm.riscv.tuple.extract.nxv16i32.triscv.vector.tuple_nxv64i8_2t(target("riscv.vector.tuple", <vscale x 64 x i8>, 2) {{%.*}}, i32 0)
//
vint32m8_t test_pair_mac_unpair(vint32m8_t lo, vint32m8_t hi, vint32m1_t vs1,
                                vint32m1_t vs2, size_t vl) {
  vint32m16_t acc = __riscv_ime_vpair_i32m16(lo, hi);
  acc = __riscv_vmmacc_vv_i32m16(acc, vs1, vs2, vl);
  return __riscv_ime_vunpairlo_i32m16(acc);
}
