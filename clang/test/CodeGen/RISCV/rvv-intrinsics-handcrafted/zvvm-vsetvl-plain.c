// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -target-feature +experimental-zvvmm \
// RUN:   -O2 -emit-llvm %s -o - | FileCheck %s --check-prefix=ZVVM
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -target-feature +experimental-zvvmtls \
// RUN:   -O2 -emit-llvm %s -o - | FileCheck %s --check-prefix=ZVVM
// RUN: %clang_cc1 -triple riscv64 -target-feature +v \
// RUN:   -O2 -emit-llvm %s -o - | FileCheck %s --check-prefix=PLAIN

// The plain vsetvl builtins under a Zvvm family extension lower to the matrix
// state-writing vsetvli intrinsics, so `__riscv_vsetvl_e16m4(vl);
// __riscv_vsetlambda(16);` keeps the vsetvli, unused result and all, in front
// of the lambda write. Without Zvvm they stay the memory-free intrinsics.

#include <riscv_vector.h>

// ZVVM-LABEL: define dso_local{{.*}} void @test_vsetvl_unused
// ZVVM:         tail call i64 @llvm.riscv.vsetvli.zvvm.i64(i64 1024, i64 1, i64 2)
// ZVVM-NEXT:    ret void
//
// PLAIN-LABEL: define dso_local{{.*}} void @test_vsetvl_unused
// PLAIN-NEXT:  entry:
// PLAIN-NEXT:    ret void
//
void test_vsetvl_unused(void) {
  __riscv_vsetvl_e16m4(1024);
}

// ZVVM-LABEL: define dso_local{{.*}} i64 @test_vsetvl_used
// ZVVM:         [[VL:%.*]] = tail call i64 @llvm.riscv.vsetvli.zvvm.i64(i64 %avl, i64 2, i64 0)
// ZVVM-NEXT:    ret i64 [[VL]]
//
// PLAIN-LABEL: define dso_local{{.*}} i64 @test_vsetvl_used
// PLAIN:         [[VL:%.*]] = tail call i64 @llvm.riscv.vsetvli.i64(i64 %avl, i64 2, i64 0)
// PLAIN-NEXT:    ret i64 [[VL]]
//
size_t test_vsetvl_used(size_t avl) {
  return __riscv_vsetvl_e32m1(avl);
}

// ZVVM-LABEL: define dso_local{{.*}} i64 @test_vsetvlmax
// ZVVM:         [[VL:%.*]] = tail call i64 @llvm.riscv.vsetvlimax.zvvm.i64(i64 0, i64 3)
// ZVVM-NEXT:    ret i64 [[VL]]
//
// PLAIN-LABEL: define dso_local{{.*}} i64 @test_vsetvlmax
// PLAIN:         [[VL:%.*]] = tail call i64 @llvm.riscv.vsetvlimax.i64(i64 0, i64 3)
// PLAIN-NEXT:    ret i64 [[VL]]
//
size_t test_vsetvlmax(void) {
  return __riscv_vsetvlmax_e8m8();
}
