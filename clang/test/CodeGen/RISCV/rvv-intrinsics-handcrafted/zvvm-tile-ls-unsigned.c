// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 \
// RUN:   -target-feature +v -target-feature +experimental-zvvmm \
// RUN:   -target-feature +experimental-zvvmtls \
// RUN:   -target-feature +experimental-zvvmttls \
// RUN:   -O2 -emit-llvm %s -o - | FileCheck %s

// Unsigned-element tile load/store builtins (u8/u16/u32/u64), mirroring the
// signed coverage across all four mnemonics, `_L{N}` overrides and masked
// forms. The IR intrinsics are signless — the unsigned variants exist so
// user code type-checks against vuint tiles.

#pragma clang riscv intrinsic zvvm_vector

#include <riscv_vector.h>

// CHECK-LABEL: define dso_local <vscale x 2 x i32> @test_vmtl_u32m1
// CHECK:         tail call <vscale x 2 x i32> @llvm.riscv.vmtl.nxv2i32.p0.i64(<vscale x 2 x i32> poison, ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
vuint32m1_t test_vmtl_u32m1(const uint32_t *base, size_t ld, size_t vl) {
  return __riscv_vmtl_v_u32m1(base, ld, vl);
}

// CHECK-LABEL: define dso_local <vscale x 16 x i8> @test_vmtl_u8m2_L4
// CHECK:         tail call <vscale x 16 x i8> @llvm.riscv.vmtl.l4.nxv16i8.p0.i64(<vscale x 16 x i8> poison, ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
vuint8m2_t test_vmtl_u8m2_L4(const uint8_t *base, size_t ld, size_t vl) {
  return __riscv_vmtl_v_u8m2_L4(base, ld, vl);
}

// CHECK-LABEL: define dso_local void @test_vmts_u16m1
// CHECK:         tail call void @llvm.riscv.vmts.nxv4i16.p0.i64(<vscale x 4 x i16> %{{.*}}, ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
void test_vmts_u16m1(uint16_t *base, size_t ld, vuint16m1_t value,
                     size_t vl) {
  __riscv_vmts_v_u16m1(base, ld, value, vl);
}

// Masked transposing load: mask ratio SEW/LMUL = 32.
//
// CHECK-LABEL: define dso_local <vscale x 2 x i32> @test_vmttl_u32m1_m
// CHECK:         tail call <vscale x 2 x i32> @llvm.riscv.vmttl.mask.nxv2i32.p0.i64(<vscale x 2 x i32> poison, ptr %{{.*}}, i64 %{{.*}}, <vscale x 2 x i1> %{{.*}}, i64 %{{.*}}, i64 3)
vuint32m1_t test_vmttl_u32m1_m(vbool32_t mask, const uint32_t *base,
                               size_t ld, size_t vl) {
  return __riscv_vmttl_v_u32m1_m(mask, base, ld, vl);
}

// CHECK-LABEL: define dso_local void @test_vmtts_u64m2
// CHECK:         tail call void @llvm.riscv.vmtts.nxv2i64.p0.i64(<vscale x 2 x i64> %{{.*}}, ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
void test_vmtts_u64m2(uint64_t *base, size_t ld, vuint64m2_t value,
                      size_t vl) {
  __riscv_vmtts_v_u64m2(base, ld, value, vl);
}

// `_L{N}` composed with `_m` on an unsigned store.
//
// CHECK-LABEL: define dso_local void @test_vmts_u8m1_L16_m
// CHECK:         tail call void @llvm.riscv.vmts.l16.mask.nxv8i8.p0.i64(<vscale x 8 x i8> %{{.*}}, ptr %{{.*}}, i64 %{{.*}}, <vscale x 8 x i1> %{{.*}}, i64 %{{.*}})
void test_vmts_u8m1_L16_m(vbool8_t mask, uint8_t *base, size_t ld,
                          vuint8m1_t value, size_t vl) {
  __riscv_vmts_v_u8m1_L16_m(mask, base, ld, value, vl);
}
