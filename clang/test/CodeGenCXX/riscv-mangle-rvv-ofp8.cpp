// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64-none-linux-gnu -target-feature +v \
// RUN:   -target-feature +experimental-zvvfmm -emit-llvm -o - %s | FileCheck %s

// The IME (Zvvm) OFP8 vector types mangle as vendor types after their
// __rvv_* builtin names, like every other RVV type, and are distinct
// overloads from the vint8/vuint8 types that share their i8 container.

#include <riscv_vector.h>

// CHECK: define{{.*}} void @_Z1fu20__rvv_float8e4m3m1_t(
void f(vfloat8e4m3m1_t) {}
// CHECK: define{{.*}} void @_Z1fu20__rvv_float8e5m2m1_t(
void f(vfloat8e5m2m1_t) {}
// CHECK: define{{.*}} void @_Z1fu15__rvv_uint8m1_t(
void f(vuint8m1_t) {}
// CHECK: define{{.*}} void @_Z1fu14__rvv_int8m1_t(
void f(vint8m1_t) {}
// CHECK: define{{.*}} void @_Z1gu21__rvv_float8e4m3mf8_t(
void g(vfloat8e4m3mf8_t) {}
// CHECK: define{{.*}} void @_Z1gu20__rvv_float8e5m2m8_t(
void g(vfloat8e5m2m8_t) {}
// CHECK: define{{.*}} void @_Z1hu21__rvv_float8e4m3m16_t(
void h(vfloat8e4m3m16_t) {}
// CHECK: define{{.*}} void @_Z1hu21__rvv_float8e5m2m16_t(
void h(vfloat8e5m2m16_t) {}
