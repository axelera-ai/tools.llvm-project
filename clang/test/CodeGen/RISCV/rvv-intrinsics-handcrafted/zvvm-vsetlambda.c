// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -target-feature +experimental-zvvmm \
// RUN:   -O2 -emit-llvm %s -o - | FileCheck %s

// Lambda surface: __riscv_vsetlambda(L) wraps the encoding-form IR intrinsic
// int_riscv_vsetlambda for every request, including L == 0, which per the IME
// spec is a preserve-or-initialize WRITE (encoding 0), not a read-only query.
// The C signature accepts the spec's power-of-two form; Clang converts
// between the user value (0 / 2^k) and the 3-bit vtype.lambda encoding
// (0 / k+1) at IR-gen time — folded for constants, cttz/select for runtime
// requests. The pure-read counterpart is __riscv_ime_lambda().

#pragma clang riscv intrinsic zvvm_vector

#include <riscv_vector.h>

// L == 0: preserve-or-initialize write. Calls int_riscv_vsetlambda with
// encoding 0 (NOT int_riscv_query_lambda); the result is converted from
// the established encoding via select(eq, 0, 1 << (enc - 1)).
//
// CHECK-LABEL: define dso_local{{.*}} i64 @test_vsetlambda_0
// CHECK:         [[ENC:%.*]] = tail call i64 @llvm.riscv.vsetlambda.i64(i64 0)
// CHECK-NEXT:    [[ISZERO:%.*]] = icmp eq i64 [[ENC]], 0
// CHECK-NEXT:    [[ENCM1:%.*]] = add i64 [[ENC]], -1
// CHECK-NEXT:    [[POT:%.*]] = shl {{.*}}i64 1, [[ENCM1]]
// CHECK-NEXT:    [[RES:%.*]] = select i1 [[ISZERO]], i64 0, i64 [[POT]]
// CHECK-NEXT:    ret i64 [[RES]]
//
size_t test_vsetlambda_0(void) {
  return __riscv_vsetlambda(0);
}

// L == 1: encoding = 1. Sets lambda to L1.
//
// CHECK-LABEL: define dso_local{{.*}} i64 @test_vsetlambda_1
// CHECK:         tail call i64 @llvm.riscv.vsetlambda.i64(i64 1)
//
size_t test_vsetlambda_1(void) {
  return __riscv_vsetlambda(1);
}

// L == 4: encoding = log2(4) + 1 = 3.
//
// CHECK-LABEL: define dso_local{{.*}} i64 @test_vsetlambda_4
// CHECK:         tail call i64 @llvm.riscv.vsetlambda.i64(i64 3)
//
size_t test_vsetlambda_4(void) {
  return __riscv_vsetlambda(4);
}

// L == 64: encoding = log2(64) + 1 = 7. Largest valid lambda.
//
// CHECK-LABEL: define dso_local{{.*}} i64 @test_vsetlambda_64
// CHECK:         tail call i64 @llvm.riscv.vsetlambda.i64(i64 7)
//
size_t test_vsetlambda_64(void) {
  return __riscv_vsetlambda(64);
}

// Runtime request: encoding computed as (L == 0) ? 0 : cttz(L) + 1, then a
// single call to int_riscv_vsetlambda. Out-of-domain runtime values are UB.
//
// CHECK-LABEL: define dso_local{{.*}} i64 @test_vsetlambda_runtime
// CHECK:         call {{.*}}i64 @llvm.cttz.i64(i64 %{{.*}}, i1 true)
// CHECK:         call i64 @llvm.riscv.vsetlambda.i64(i64 %{{.*}})
//
size_t test_vsetlambda_runtime(size_t L) {
  return __riscv_vsetlambda(L);
}

// __riscv_ime_lambda(): pure read of vtype.lambda via int_riscv_query_lambda,
// converted from the encoding to the power-of-two form. No vtype write.
//
// CHECK-LABEL: define dso_local{{.*}} i64 @test_ime_lambda
// CHECK:         [[QENC:%.*]] = tail call i64 @llvm.riscv.query.lambda.i64()
// CHECK-NEXT:    [[QISZERO:%.*]] = icmp eq i64 [[QENC]], 0
// CHECK-NEXT:    [[QENCM1:%.*]] = add i64 [[QENC]], -1
// CHECK-NEXT:    [[QPOT:%.*]] = shl {{.*}}i64 1, [[QENCM1]]
// CHECK-NEXT:    [[QRES:%.*]] = select i1 [[QISZERO]], i64 0, i64 [[QPOT]]
// CHECK-NEXT:    ret i64 [[QRES]]
// CHECK-NOT:     @llvm.riscv.vsetlambda
//
size_t test_ime_lambda(void) {
  return __riscv_ime_lambda();
}

// __riscv_ime_vlen(): read_register(vlenb) scaled from bytes to bits.
//
// CHECK-LABEL: define dso_local{{.*}} i64 @test_ime_vlen
// CHECK:         [[VLENB:%.*]] = {{.*}}call i64 @llvm.read_register.i64(metadata [[MD:![0-9]+]])
// CHECK:         [[VLEN:%.*]] = shl i64 [[VLENB]], 3
// CHECK:         ret i64 [[VLEN]]
//
// CHECK: [[MD]] = !{!"vlenb"}
//
size_t test_ime_vlen(void) {
  return __riscv_ime_vlen();
}
