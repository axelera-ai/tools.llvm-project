// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -target-feature +zvfhmin \
// RUN:   -target-feature +experimental-zvvi32mm \
// RUN:   -target-feature +experimental-zvvi16i32mm \
// RUN:   -target-feature +experimental-zvvfp32mm \
// RUN:   -target-feature +experimental-zvvfp16fp64mm \
// RUN:   -target-feature +experimental-zvvxni8fp16mm \
// RUN:   -target-feature +experimental-zvvmtls \
// RUN:   -target-feature +experimental-zvvmttls \
// RUN:   -O2 -emit-llvm %s -o - | FileCheck %s

#pragma clang riscv intrinsic zvvm_vector

#include <riscv_vector.h>

// Overloaded short forms (spec §"Overloaded short forms"): the variant is
// selected solely from the argument types — no `_lm{N}`, input-type,
// `_su`/`_us`, or `_L{N}` suffix appears in user code. Each function below
// must resolve to exactly the IR intrinsic its long-form sibling produces.
//
// Exceptions kept by design: the integer-input MX MACs retain the `_bs{N}`
// qualifier in the short form (two block sizes share one C signature), and
// unmasked tile loads have no usable short form (return-type-only variance;
// see the Sema test).

// MAC with non-default EMUL_C and LMUL: (vd, vs1, vs2) types select the
// i32m2_lm4 cell.
// CHECK-LABEL: define dso_local <vscale x 4 x i32> @test_mac_short
// CHECK:         call <vscale x 4 x i32> @llvm.riscv.vmmacc.nxv4i32.nxv8i32.i64(<vscale x 4 x i32> %{{.*}}, <vscale x 8 x i32> %{{.*}}, <vscale x 8 x i32> %{{.*}}, i64 %{{.*}})
vint32m2_t test_mac_short(vint32m2_t vd, vint32m4_t vs1, vint32m4_t vs2,
                          size_t vl) {
  return __riscv_vmmacc_vv(vd, vs1, vs2, vl);
}

// Mixed-sign cell: (vint, vint, vuint) selects the `_su` variant of the
// same signless IR intrinsic.
// CHECK-LABEL: define dso_local <vscale x 2 x i32> @test_mac_su_short
// CHECK:         call <vscale x 2 x i32> @llvm.riscv.vmmacc.nxv2i32.nxv2i32.i64(<vscale x 2 x i32> %{{.*}}, <vscale x 2 x i32> %{{.*}}, <vscale x 2 x i32> %{{.*}}, i64 %{{.*}})
vint32m1_t test_mac_su_short(vint32m1_t vd, vint32m1_t vs1, vuint32m1_t vs2,
                             size_t vl) {
  return __riscv_vmmacc_vv(vd, vs1, vs2, vl);
}

// Unsigned-unsigned cell.
// CHECK-LABEL: define dso_local <vscale x 2 x i32> @test_mac_uu_short
// CHECK:         call <vscale x 2 x i32> @llvm.riscv.vmmacc.nxv2i32.nxv2i32.i64
vuint32m1_t test_mac_uu_short(vuint32m1_t vd, vuint32m1_t vs1,
                              vuint32m1_t vs2, size_t vl) {
  return __riscv_vmmacc_vv(vd, vs1, vs2, vl);
}

// m16 accumulator cell: the tuple vd type selects the *_m16 IR intrinsic.
// CHECK-LABEL: define dso_local target("riscv.vector.tuple", <vscale x 64 x i8>, 2) @test_mac_m16_short
// CHECK:         call target("riscv.vector.tuple", <vscale x 64 x i8>, 2) @llvm.riscv.vmmacc.m16.triscv.vector.tuple_nxv64i8_2t.nxv16i32.i64(target("riscv.vector.tuple", <vscale x 64 x i8>, 2) %{{.*}}, <vscale x 16 x i32> %{{.*}}, <vscale x 16 x i32> %{{.*}}, i64 %{{.*}})
vint32m16_t test_mac_m16_short(vint32m16_t vd, vint32m8_t vs1,
                               vint32m8_t vs2, size_t vl) {
  return __riscv_vmmacc_vv(vd, vs1, vs2, vl);
}

// Widening MAC: input SEW half of accumulator SEW selects vwmmacc's
// i32m1_lm2 cell.
// CHECK-LABEL: define dso_local <vscale x 2 x i32> @test_wmac_short
// CHECK:         call <vscale x 2 x i32> @llvm.riscv.vwmmacc.nxv2i32.nxv8i16.i64(<vscale x 2 x i32> %{{.*}}, <vscale x 8 x i16> %{{.*}}, <vscale x 8 x i16> %{{.*}}, i64 %{{.*}})
vint32m1_t test_wmac_short(vint32m1_t vd, vint16m2_t vs1, vint16m2_t vs2,
                           size_t vl) {
  return __riscv_vwmmacc_vv(vd, vs1, vs2, vl);
}

// FP MAC and FP quad-widening MAC.
// CHECK-LABEL: define dso_local <vscale x 2 x float> @test_fmac_short
// CHECK:         call <vscale x 2 x float> @llvm.riscv.vfmmacc.nxv2f32.nxv2f32.i64
vfloat32m1_t test_fmac_short(vfloat32m1_t vd, vfloat32m1_t vs1,
                             vfloat32m1_t vs2, size_t vl) {
  return __riscv_vfmmacc_vv(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 2 x double> @test_fqmac_short
// CHECK:         call <vscale x 2 x double> @llvm.riscv.vfqmmacc.nxv2f64.nxv4f16.i64
vfloat64m2_t test_fqmac_short(vfloat64m2_t vd, vfloat16m1_t vs1,
                              vfloat16m1_t vs2, size_t vl) {
  return __riscv_vfqmmacc_vv(vd, vs1, vs2, vl);
}

// Integer-input MX MAC: `_bs{N}` stays in the short form; the bs immarg
// must follow the suffix, not the (identical) argument types.
// CHECK-LABEL: define dso_local <vscale x 4 x half> @test_mx_bs32_short
// CHECK:         call <vscale x 4 x half> @llvm.riscv.vfwimmacc.nxv4f16.nxv8i8.i64(<vscale x 4 x half> %{{.*}}, <vscale x 8 x i8> %{{.*}}, <vscale x 8 x i8> %{{.*}}, <vscale x 4 x i16> %{{.*}}, i64 0, i64 %{{.*}})
vfloat16m1_t test_mx_bs32_short(vfloat16m1_t vd, vint8m1_t vs1, vint8m1_t vs2,
                                vuint16m1_t v0, size_t vl) {
  return __riscv_vfwimmacc_vv_bs32(vd, vs1, vs2, v0, vl);
}

// CHECK-LABEL: define dso_local <vscale x 4 x half> @test_mx_bs16_short
// CHECK:         call <vscale x 4 x half> @llvm.riscv.vfwimmacc.nxv4f16.nxv8i8.i64(<vscale x 4 x half> %{{.*}}, <vscale x 8 x i8> %{{.*}}, <vscale x 8 x i8> %{{.*}}, <vscale x 4 x i16> %{{.*}}, i64 1, i64 %{{.*}})
vfloat16m1_t test_mx_bs16_short(vfloat16m1_t vd, vint8m1_t vs1, vint8m1_t vs2,
                                vuint16m1_t v0, size_t vl) {
  return __riscv_vfwimmacc_vv_bs16(vd, vs1, vs2, v0, vl);
}

// Pair/unpair short forms (explicitly spec'd).
// CHECK-LABEL: define dso_local target("riscv.vector.tuple", <vscale x 64 x i8>, 2) @test_pair_short
// CHECK:         call target("riscv.vector.tuple", <vscale x 64 x i8>, 2) @llvm.riscv.tuple.insert.triscv.vector.tuple_nxv64i8_2t.nxv16i32(target("riscv.vector.tuple", <vscale x 64 x i8>, 2) poison, <vscale x 16 x i32> %{{.*}}, i32 0)
// CHECK:         call target("riscv.vector.tuple", <vscale x 64 x i8>, 2) @llvm.riscv.tuple.insert.triscv.vector.tuple_nxv64i8_2t.nxv16i32(target("riscv.vector.tuple", <vscale x 64 x i8>, 2) %{{.*}}, <vscale x 16 x i32> %{{.*}}, i32 1)
vint32m16_t test_pair_short(vint32m8_t lo, vint32m8_t hi) {
  return __riscv_ime_vpair(lo, hi);
}

// CHECK-LABEL: define dso_local <vscale x 16 x i32> @test_unpairhi_short
// CHECK:         call <vscale x 16 x i32> @llvm.riscv.tuple.extract.nxv16i32.triscv.vector.tuple_nxv64i8_2t(target("riscv.vector.tuple", <vscale x 64 x i8>, 2) %{{.*}}, i32 1)
vint32m8_t test_unpairhi_short(vint32m16_t src) {
  return __riscv_ime_vunpairhi(src);
}

// Tile store: value type + pointer select the cell; resolves to the
// dynamic-lambda (default) variant.
// CHECK-LABEL: define dso_local void @test_store_short
// CHECK:         call void @llvm.riscv.vmts.nxv2i32.p0.i64(<vscale x 2 x i32> %{{.*}}, ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
void test_store_short(int32_t *base, size_t ld, vint32m1_t v, size_t vl) {
  __riscv_vmts_v(base, ld, v, vl);
}

// Masked transposing tile store.
// CHECK-LABEL: define dso_local void @test_masked_tstore_short
// CHECK:         call void @llvm.riscv.vmtts.mask.nxv2i32.p0.i64(<vscale x 2 x i32> %{{.*}}, ptr %{{.*}}, i64 %{{.*}}, <vscale x 2 x i1> %{{.*}}, i64 %{{.*}})
void test_masked_tstore_short(vbool32_t mask, int32_t *base, size_t ld,
                              vint32m1_t v, size_t vl) {
  __riscv_vmtts_v(mask, base, ld, v, vl);
}

// `_as_e{S}` store: the (uint32_t*, vfloat16m1_t) pair uniquely selects the
// f16m1_as_e32 cell — the value is reinterpreted to the e32 storage
// container.
// CHECK-LABEL: define dso_local void @test_as_e_store_short
// CHECK:         bitcast <vscale x 4 x half> %{{.*}} to <vscale x 2 x i32>
// CHECK:         call void @llvm.riscv.vmts.nxv2i32.p0.i64(<vscale x 2 x i32> %{{.*}}, ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
void test_as_e_store_short(uint32_t *base, size_t ld, vfloat16m1_t v,
                           size_t vl) {
  __riscv_vmts_v(base, ld, v, vl);
}

// Masked tile load: (mask, pointer) determine element type and LMUL.
// CHECK-LABEL: define dso_local <vscale x 4 x i32> @test_masked_load_short
// CHECK:         call <vscale x 4 x i32> @llvm.riscv.vmtl.mask.nxv4i32.p0.i64(<vscale x 4 x i32> poison, ptr %{{.*}}, i64 %{{.*}}, <vscale x 4 x i1> %{{.*}}, i64 %{{.*}}, i64 3)
vint32m2_t test_masked_load_short(vbool16_t mask, const int32_t *base,
                                  size_t ld, size_t vl) {
  return __riscv_vmtl_v(mask, base, ld, vl);
}

// `_tu` tile load: the passthru argument makes the unmasked load
// overloadable under the `{mnemonic}_tu` short name (upstream convention).
// CHECK-LABEL: define dso_local <vscale x 2 x i32> @test_tu_load_short
// CHECK:         call <vscale x 2 x i32> @llvm.riscv.vmttl.nxv2i32.p0.i64(<vscale x 2 x i32> %{{.*}}, ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
vint32m1_t test_tu_load_short(vint32m1_t passthru, const int32_t *base,
                              size_t ld, size_t vl) {
  return __riscv_vmttl_v_tu(passthru, base, ld, vl);
}
