// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 \
// RUN:   -target-feature +v -target-feature +zvfh \
// RUN:   -target-feature +experimental-zvvmm \
// RUN:   -target-feature +experimental-zvvmtls \
// RUN:   -O2 -emit-llvm %s -o - | FileCheck %s

// Storage-width-qualified tile load/store builtins (`_as_e{S}`, spec
// §"Alternate-format and storage-width-qualified tile load/store
// intrinsics"; order-preserving pair only). The instruction moves S-bit
// storage elements: the IR intrinsic is called at the S-bit storage
// container type and the result/value is reinterpreted (bit-preserving
// bitcast) to/from the logical vector type. `ld` and `vl` count S-bit
// storage elements, the base pointer is uint{S}_t regardless of the logical
// type, and a mask bit controls one whole storage element (mask ratio
// S/LMUL).

#pragma clang riscv intrinsic zvvm_vector

#include <riscv_vector.h>

// Two packed f16 values per 32-bit storage element: the load runs at the
// i32 container and the result is reinterpreted.
//
// CHECK-LABEL: define dso_local <vscale x 4 x half> @test_vmtl_f16m1_as_e32
// CHECK:         [[L:%.*]] = tail call <vscale x 2 x i32> @llvm.riscv.vmtl.nxv2i32.p0.i64(<vscale x 2 x i32> poison, ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
// CHECK:         bitcast <vscale x 2 x i32> [[L]] to <vscale x 4 x half>
vfloat16m1_t test_vmtl_f16m1_as_e32(const uint32_t *base, size_t ld,
                                    size_t vl) {
  return __riscv_vmtl_v_f16m1_as_e32(base, ld, vl);
}

// Eight packed i8 values per 64-bit storage element.
//
// CHECK-LABEL: define dso_local <vscale x 8 x i8> @test_vmtl_i8m1_as_e64
// CHECK:         [[L:%.*]] = tail call <vscale x 1 x i64> @llvm.riscv.vmtl.nxv1i64.p0.i64(<vscale x 1 x i64> poison, ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
// CHECK:         bitcast <vscale x 1 x i64> [[L]] to <vscale x 8 x i8>
vint8m1_t test_vmtl_i8m1_as_e64(const uint64_t *base, size_t ld, size_t vl) {
  return __riscv_vmtl_v_i8m1_as_e64(base, ld, vl);
}

// Natural-width alias: `_as_e32` on an i32 logical type is equivalent to
// the unqualified form — same container, no reinterpret.
//
// CHECK-LABEL: define dso_local <vscale x 2 x i32> @test_vmtl_i32m1_as_e32
// CHECK:         tail call <vscale x 2 x i32> @llvm.riscv.vmtl.nxv2i32.p0.i64(<vscale x 2 x i32> poison, ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
// CHECK-NOT:     bitcast
vint32m1_t test_vmtl_i32m1_as_e32(const uint32_t *base, size_t ld,
                                  size_t vl) {
  return __riscv_vmtl_v_i32m1_as_e32(base, ld, vl);
}

// LMUL > 1: the m{N} in the type suffix is the register-group size, shared
// by the logical view and the storage container (f16m4 <-> i32m4).
//
// CHECK-LABEL: define dso_local <vscale x 16 x half> @test_vmtl_f16m4_as_e32
// CHECK:         [[L:%.*]] = tail call <vscale x 8 x i32> @llvm.riscv.vmtl.nxv8i32.p0.i64(<vscale x 8 x i32> poison, ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
// CHECK:         bitcast <vscale x 8 x i32> [[L]] to <vscale x 16 x half>
vfloat16m4_t test_vmtl_f16m4_as_e32(const uint32_t *base, size_t ld,
                                    size_t vl) {
  return __riscv_vmtl_v_f16m4_as_e32(base, ld, vl);
}

// `_L{N}` composes before `_as_e{S}`'s masked/unmasked axis exactly like
// the plain tile forms; unsigned logical types share the signless IR
// intrinsic.
//
// CHECK-LABEL: define dso_local <vscale x 8 x i16> @test_vmtl_u16m2_as_e32_L4
// CHECK:         [[L:%.*]] = tail call <vscale x 4 x i32> @llvm.riscv.vmtl.l4.nxv4i32.p0.i64(<vscale x 4 x i32> poison, ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
// CHECK:         bitcast <vscale x 4 x i32> [[L]] to <vscale x 8 x i16>
vuint16m2_t test_vmtl_u16m2_as_e32_L4(const uint32_t *base, size_t ld,
                                      size_t vl) {
  return __riscv_vmtl_v_u16m2_as_e32_L4(base, ld, vl);
}

// Masked: the mask ratio is S/LMUL — vbool32_t for e32 at LMUL=1 (one mask
// bit per 32-bit storage element), regardless of the 16-bit logical width.
//
// CHECK-LABEL: define dso_local <vscale x 4 x half> @test_vmtl_f16m1_as_e32_m
// CHECK:         [[L:%.*]] = tail call <vscale x 2 x i32> @llvm.riscv.vmtl.mask.nxv2i32.p0.i64(<vscale x 2 x i32> poison, ptr %{{.*}}, i64 %{{.*}}, <vscale x 2 x i1> %{{.*}}, i64 %{{.*}}, i64 3)
// CHECK:         bitcast <vscale x 2 x i32> [[L]] to <vscale x 4 x half>
vfloat16m1_t test_vmtl_f16m1_as_e32_m(vbool32_t mask, const uint32_t *base,
                                      size_t ld, size_t vl) {
  return __riscv_vmtl_v_f16m1_as_e32_m(mask, base, ld, vl);
}

// Store: the logical value is reinterpreted to the storage container before
// the store intrinsic (value, ptr, ld, vl).
//
// CHECK-LABEL: define dso_local void @test_vmts_f16m1_as_e32
// CHECK:         [[V:%.*]] = bitcast <vscale x 4 x half> %{{.*}} to <vscale x 2 x i32>
// CHECK:         tail call void @llvm.riscv.vmts.nxv2i32.p0.i64(<vscale x 2 x i32> [[V]], ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
void test_vmts_f16m1_as_e32(uint32_t *base, size_t ld, vfloat16m1_t value,
                            size_t vl) {
  __riscv_vmts_v_f16m1_as_e32(base, ld, value, vl);
}

// Masked store with a sub-width logical type: u8 elements in 16-bit storage,
// mask ratio 16/1.
//
// CHECK-LABEL: define dso_local void @test_vmts_u8m1_as_e16_m
// CHECK:         [[V:%.*]] = bitcast <vscale x 8 x i8> %{{.*}} to <vscale x 4 x i16>
// CHECK:         tail call void @llvm.riscv.vmts.mask.nxv4i16.p0.i64(<vscale x 4 x i16> [[V]], ptr %{{.*}}, i64 %{{.*}}, <vscale x 4 x i1> %{{.*}}, i64 %{{.*}})
void test_vmts_u8m1_as_e16_m(vbool16_t mask, uint16_t *base, size_t ld,
                             vuint8m1_t value, size_t vl) {
  __riscv_vmts_v_u8m1_as_e16_m(mask, base, ld, value, vl);
}

// Natural-width store alias with a lambda override.
//
// CHECK-LABEL: define dso_local void @test_vmts_i64m1_as_e64_L8
// CHECK:         tail call void @llvm.riscv.vmts.l8.nxv1i64.p0.i64(<vscale x 1 x i64> %{{.*}}, ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
// CHECK-NOT:     bitcast
void test_vmts_i64m1_as_e64_L8(uint64_t *base, size_t ld, vint64m1_t value,
                               size_t vl) {
  __riscv_vmts_v_i64m1_as_e64_L8(base, ld, value, vl);
}
