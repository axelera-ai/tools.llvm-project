// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +v \
// RUN:   -target-feature +experimental-zvvmtls \
// RUN:   -target-feature +experimental-zvvmttls \
// RUN:   -O2 -emit-llvm %s -o - | FileCheck %s

// OFP8 (E4M3 / E5M2) tile load/store builtins. The natural storage width of
// OFP8 is 8 bits, so the unqualified forms run at SEW=8 on a uint8_t base
// (spec §"Alternate-format and storage-width-qualified tile load/store
// intrinsics"). In IR the OFP8 vector types are i8 containers, so every
// form calls the i8 tile intrinsic directly; the `_as_e{S}` forms call it at
// the S-bit storage container and reinterpret with a bit-preserving bitcast.

#pragma clang riscv intrinsic zvvm_vector

#include <riscv_vector.h>

// Natural-storage default-lambda forms.
//
// CHECK-LABEL: define dso_local <vscale x 8 x i8> @test_vmtl_f8e4m3m1
// CHECK:         tail call <vscale x 8 x i8> @llvm.riscv.vmtl.nxv8i8.p0.i64(<vscale x 8 x i8> poison, ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
vfloat8e4m3m1_t test_vmtl_f8e4m3m1(const uint8_t *base, size_t ld,
                                   size_t vl) {
  return __riscv_vmtl_v_f8e4m3m1(base, ld, vl);
}

// CHECK-LABEL: define dso_local <vscale x 16 x i8> @test_vmtl_f8e5m2m2
// CHECK:         tail call <vscale x 16 x i8> @llvm.riscv.vmtl.nxv16i8.p0.i64(<vscale x 16 x i8> poison, ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
vfloat8e5m2m2_t test_vmtl_f8e5m2m2(const uint8_t *base, size_t ld,
                                   size_t vl) {
  return __riscv_vmtl_v_f8e5m2m2(base, ld, vl);
}

// CHECK-LABEL: define dso_local <vscale x 64 x i8> @test_vmttl_f8e4m3m8
// CHECK:         tail call <vscale x 64 x i8> @llvm.riscv.vmttl.nxv64i8.p0.i64(<vscale x 64 x i8> poison, ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
vfloat8e4m3m8_t test_vmttl_f8e4m3m8(const uint8_t *base, size_t ld,
                                    size_t vl) {
  return __riscv_vmttl_v_f8e4m3m8(base, ld, vl);
}

// CHECK-LABEL: define dso_local void @test_vmts_f8e4m3m1
// CHECK:         tail call void @llvm.riscv.vmts.nxv8i8.p0.i64(<vscale x 8 x i8> %{{.*}}, ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
void test_vmts_f8e4m3m1(uint8_t *base, size_t ld, vfloat8e4m3m1_t v,
                        size_t vl) {
  __riscv_vmts_v_f8e4m3m1(base, ld, v, vl);
}

// CHECK-LABEL: define dso_local void @test_vmtts_f8e5m2m4
// CHECK:         tail call void @llvm.riscv.vmtts.nxv32i8.p0.i64(<vscale x 32 x i8> %{{.*}}, ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
void test_vmtts_f8e5m2m4(uint8_t *base, size_t ld, vfloat8e5m2m4_t v,
                         size_t vl) {
  __riscv_vmtts_v_f8e5m2m4(base, ld, v, vl);
}

// `_tu` (non-spec, mirrors the other element types): explicit passthru.
//
// CHECK-LABEL: define dso_local <vscale x 8 x i8> @test_vmtl_f8e4m3m1_tu
// CHECK:         tail call <vscale x 8 x i8> @llvm.riscv.vmtl.nxv8i8.p0.i64(<vscale x 8 x i8> %pt, ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
vfloat8e4m3m1_t test_vmtl_f8e4m3m1_tu(vfloat8e4m3m1_t pt, const uint8_t *base,
                                      size_t ld, size_t vl) {
  return __riscv_vmtl_v_f8e4m3m1_tu(pt, base, ld, vl);
}

// Lambda overrides.
//
// CHECK-LABEL: define dso_local <vscale x 8 x i8> @test_vmtl_f8e4m3m1_L4
// CHECK:         tail call <vscale x 8 x i8> @llvm.riscv.vmtl.l4.nxv8i8.p0.i64(<vscale x 8 x i8> poison, ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
vfloat8e4m3m1_t test_vmtl_f8e4m3m1_L4(const uint8_t *base, size_t ld,
                                      size_t vl) {
  return __riscv_vmtl_v_f8e4m3m1_L4(base, ld, vl);
}

// CHECK-LABEL: define dso_local <vscale x 32 x i8> @test_vmttl_f8e5m2m4_L64
// CHECK:         tail call <vscale x 32 x i8> @llvm.riscv.vmttl.l64.nxv32i8.p0.i64(<vscale x 32 x i8> poison, ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
vfloat8e5m2m4_t test_vmttl_f8e5m2m4_L64(const uint8_t *base, size_t ld,
                                        size_t vl) {
  return __riscv_vmttl_v_f8e5m2m4_L64(base, ld, vl);
}

// CHECK-LABEL: define dso_local void @test_vmts_f8e5m2m2_L16
// CHECK:         tail call void @llvm.riscv.vmts.l16.nxv16i8.p0.i64(<vscale x 16 x i8> %{{.*}}, ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
void test_vmts_f8e5m2m2_L16(uint8_t *base, size_t ld, vfloat8e5m2m2_t v,
                            size_t vl) {
  __riscv_vmts_v_f8e5m2m2_L16(base, ld, v, vl);
}

// CHECK-LABEL: define dso_local void @test_vmtts_f8e4m3m8_L1
// CHECK:         tail call void @llvm.riscv.vmtts.l1.nxv64i8.p0.i64(<vscale x 64 x i8> %{{.*}}, ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
void test_vmtts_f8e4m3m8_L1(uint8_t *base, size_t ld, vfloat8e4m3m8_t v,
                            size_t vl) {
  __riscv_vmtts_v_f8e4m3m8_L1(base, ld, v, vl);
}

// Masked forms: the mask ratio is SEW/LMUL = 8/LMUL.
//
// CHECK-LABEL: define dso_local <vscale x 8 x i8> @test_vmtl_f8e4m3m1_m
// CHECK:         tail call <vscale x 8 x i8> @llvm.riscv.vmtl.mask.nxv8i8.p0.i64(<vscale x 8 x i8> poison, ptr %{{.*}}, i64 %{{.*}}, <vscale x 8 x i1> %mask, i64 %{{.*}}, i64 3)
vfloat8e4m3m1_t test_vmtl_f8e4m3m1_m(vbool8_t mask, const uint8_t *base,
                                     size_t ld, size_t vl) {
  return __riscv_vmtl_v_f8e4m3m1_m(mask, base, ld, vl);
}

// CHECK-LABEL: define dso_local <vscale x 16 x i8> @test_vmttl_f8e5m2m2_L8_m
// CHECK:         tail call <vscale x 16 x i8> @llvm.riscv.vmttl.l8.mask.nxv16i8.p0.i64(<vscale x 16 x i8> poison, ptr %{{.*}}, i64 %{{.*}}, <vscale x 16 x i1> %mask, i64 %{{.*}}, i64 3)
vfloat8e5m2m2_t test_vmttl_f8e5m2m2_L8_m(vbool4_t mask, const uint8_t *base,
                                         size_t ld, size_t vl) {
  return __riscv_vmttl_v_f8e5m2m2_L8_m(mask, base, ld, vl);
}

// CHECK-LABEL: define dso_local void @test_vmts_f8e5m2m8_m
// CHECK:         tail call void @llvm.riscv.vmts.mask.nxv64i8.p0.i64(<vscale x 64 x i8> %{{.*}}, ptr %{{.*}}, i64 %{{.*}}, <vscale x 64 x i1> %mask, i64 %{{.*}})
void test_vmts_f8e5m2m8_m(vbool1_t mask, uint8_t *base, size_t ld,
                          vfloat8e5m2m8_t v, size_t vl) {
  __riscv_vmts_v_f8e5m2m8_m(mask, base, ld, v, vl);
}

// CHECK-LABEL: define dso_local void @test_vmtts_f8e4m3m1_L2_m
// CHECK:         tail call void @llvm.riscv.vmtts.l2.mask.nxv8i8.p0.i64(<vscale x 8 x i8> %{{.*}}, ptr %{{.*}}, i64 %{{.*}}, <vscale x 8 x i1> %mask, i64 %{{.*}})
void test_vmtts_f8e4m3m1_L2_m(vbool8_t mask, uint8_t *base, size_t ld,
                              vfloat8e4m3m1_t v, size_t vl) {
  __riscv_vmtts_v_f8e4m3m1_L2_m(mask, base, ld, v, vl);
}

// `_as_e8` is the natural width: an alias of the unqualified form (same
// container, no reinterpret).
//
// CHECK-LABEL: define dso_local <vscale x 8 x i8> @test_vmtl_f8e4m3m1_as_e8
// CHECK:         tail call <vscale x 8 x i8> @llvm.riscv.vmtl.nxv8i8.p0.i64(<vscale x 8 x i8> poison, ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
// CHECK-NOT:     bitcast
vfloat8e4m3m1_t test_vmtl_f8e4m3m1_as_e8(const uint8_t *base, size_t ld,
                                         size_t vl) {
  return __riscv_vmtl_v_f8e4m3m1_as_e8(base, ld, vl);
}

// Packed OFP8 in wider storage elements (the flash-attention kernel's K, Q
// and P tiles).
//
// CHECK-LABEL: define dso_local <vscale x 32 x i8> @test_vmtl_f8e4m3m4_as_e16
// CHECK:         [[L:%.*]] = tail call <vscale x 16 x i16> @llvm.riscv.vmtl.nxv16i16.p0.i64(<vscale x 16 x i16> poison, ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
// CHECK:         bitcast <vscale x 16 x i16> [[L]] to <vscale x 32 x i8>
vfloat8e4m3m4_t test_vmtl_f8e4m3m4_as_e16(const uint16_t *base, size_t ld,
                                          size_t vl) {
  return __riscv_vmtl_v_f8e4m3m4_as_e16(base, ld, vl);
}

// CHECK-LABEL: define dso_local <vscale x 32 x i8> @test_vmtl_f8e4m3m4_as_e32
// CHECK:         [[L:%.*]] = tail call <vscale x 8 x i32> @llvm.riscv.vmtl.nxv8i32.p0.i64(<vscale x 8 x i32> poison, ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
// CHECK:         bitcast <vscale x 8 x i32> [[L]] to <vscale x 32 x i8>
vfloat8e4m3m4_t test_vmtl_f8e4m3m4_as_e32(const uint32_t *base, size_t ld,
                                          size_t vl) {
  return __riscv_vmtl_v_f8e4m3m4_as_e32(base, ld, vl);
}

// CHECK-LABEL: define dso_local <vscale x 8 x i8> @test_vmtl_f8e5m2m1_as_e64_L2_m
// CHECK:         [[L:%.*]] = tail call <vscale x 1 x i64> @llvm.riscv.vmtl.l2.mask.nxv1i64.p0.i64(<vscale x 1 x i64> poison, ptr %{{.*}}, i64 %{{.*}}, <vscale x 1 x i1> %mask, i64 %{{.*}}, i64 3)
// CHECK:         bitcast <vscale x 1 x i64> [[L]] to <vscale x 8 x i8>
vfloat8e5m2m1_t test_vmtl_f8e5m2m1_as_e64_L2_m(vbool64_t mask,
                                               const uint64_t *base, size_t ld,
                                               size_t vl) {
  return __riscv_vmtl_v_f8e5m2m1_as_e64_L2_m(mask, base, ld, vl);
}

// CHECK-LABEL: define dso_local void @test_vmts_f8e4m3m1_as_e32_L4
// CHECK:         [[B:%.*]] = bitcast <vscale x 8 x i8> %v to <vscale x 2 x i32>
// CHECK:         tail call void @llvm.riscv.vmts.l4.nxv2i32.p0.i64(<vscale x 2 x i32> [[B]], ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
void test_vmts_f8e4m3m1_as_e32_L4(uint32_t *base, size_t ld, vfloat8e4m3m1_t v,
                                  size_t vl) {
  __riscv_vmts_v_f8e4m3m1_as_e32_L4(base, ld, v, vl);
}

// CHECK-LABEL: define dso_local void @test_vmts_f8e5m2m2_as_e16_m
// CHECK:         [[B:%.*]] = bitcast <vscale x 16 x i8> %v to <vscale x 8 x i16>
// CHECK:         tail call void @llvm.riscv.vmts.mask.nxv8i16.p0.i64(<vscale x 8 x i16> [[B]], ptr %{{.*}}, i64 %{{.*}}, <vscale x 8 x i1> %mask, i64 %{{.*}})
void test_vmts_f8e5m2m2_as_e16_m(vbool8_t mask, uint16_t *base, size_t ld,
                                 vfloat8e5m2m2_t v, size_t vl) {
  __riscv_vmts_v_f8e5m2m2_as_e16_m(mask, base, ld, v, vl);
}

// Overloaded short forms: stores (value type selects the cell, including
// the OFP8 format and the `_as_e{S}` storage width via the pointer type).
//
// CHECK-LABEL: define dso_local void @test_vmts_short_f8e5m2m1
// CHECK:         tail call void @llvm.riscv.vmts.nxv8i8.p0.i64(<vscale x 8 x i8> %{{.*}}, ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
void test_vmts_short_f8e5m2m1(uint8_t *base, size_t ld, vfloat8e5m2m1_t v,
                              size_t vl) {
  __riscv_vmts_v(base, ld, v, vl);
}

// CHECK-LABEL: define dso_local void @test_vmts_short_f8e4m3m1_as_e32
// CHECK:         [[B:%.*]] = bitcast <vscale x 8 x i8> %v to <vscale x 2 x i32>
// CHECK:         tail call void @llvm.riscv.vmts.nxv2i32.p0.i64(<vscale x 2 x i32> [[B]], ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
void test_vmts_short_f8e4m3m1_as_e32(uint32_t *base, size_t ld,
                                     vfloat8e4m3m1_t v, size_t vl) {
  __riscv_vmts_v(base, ld, v, vl);
}

// CHECK-LABEL: define dso_local void @test_vmtts_short_f8e4m3m2_m
// CHECK:         tail call void @llvm.riscv.vmtts.mask.nxv16i8.p0.i64(<vscale x 16 x i8> %{{.*}}, ptr %{{.*}}, i64 %{{.*}}, <vscale x 16 x i1> %mask, i64 %{{.*}})
void test_vmtts_short_f8e4m3m2_m(vbool4_t mask, uint8_t *base, size_t ld,
                                 vfloat8e4m3m2_t v, size_t vl) {
  __riscv_vmtts_v(mask, base, ld, v, vl);
}
