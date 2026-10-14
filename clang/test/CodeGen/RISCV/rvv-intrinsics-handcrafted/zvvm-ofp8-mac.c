// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +v -target-feature +zvfh \
// RUN:   -target-feature +zvfbfmin -target-feature +experimental-zvvofp8mm \
// RUN:   -target-feature +experimental-zvvofp8fp16mm \
// RUN:   -target-feature +experimental-zvvofp8bf16mm \
// RUN:   -target-feature +experimental-zvvofp8fp32mm \
// RUN:   -target-feature +experimental-zvvofp8fp64mm \
// RUN:   -O2 -emit-llvm %s -o - | FileCheck %s

#pragma clang riscv intrinsic zvvm_vector

#include <riscv_vector.h>

// OFP8-input FP MACs. Names follow the spec suffix order
// {type-suffix}[_{inputA}[_{inputB}]]: the OFP8 input token carries the A/B
// LMUL (no _lm{N}), a mixed-format cell names A then B, and vfmmacc names
// its inputs only when they differ from the OFP8 accumulator format. E4M3
// vs E5M2 (and FP16 vs BF16 for the accumulator) is vtype.altfmt_A/B /
// altfmt state, so every format combination lowers to the same signless
// intrinsic with i8 input containers; a BF16 accumulator keeps its bfloat
// IR type. Each mnemonic x format combo is covered over representative
// (EMUL_C, LMUL) cells, followed by overloaded short forms.

// CHECK-LABEL: define dso_local <vscale x 16 x half> @test_vfwmmacc_vv_f16m4_f8e4m3m1(
// CHECK:         tail call <vscale x 16 x half> @llvm.riscv.vfwmmacc.nxv16f16.nxv8i8.i64(<vscale x 16 x half> {{%.*}}, <vscale x 8 x i8> {{%.*}}, <vscale x 8 x i8> {{%.*}}, i64 {{%.*}})
vfloat16m4_t test_vfwmmacc_vv_f16m4_f8e4m3m1(vfloat16m4_t vd, vfloat8e4m3m1_t vs1, vfloat8e4m3m1_t vs2, size_t vl) {
  return __riscv_vfwmmacc_vv_f16m4_f8e4m3m1(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 16 x half> @test_vfwmmacc_vv_f16m4_f8e5m2m1(
// CHECK:         tail call <vscale x 16 x half> @llvm.riscv.vfwmmacc.nxv16f16.nxv8i8.i64(<vscale x 16 x half> {{%.*}}, <vscale x 8 x i8> {{%.*}}, <vscale x 8 x i8> {{%.*}}, i64 {{%.*}})
vfloat16m4_t test_vfwmmacc_vv_f16m4_f8e5m2m1(vfloat16m4_t vd, vfloat8e5m2m1_t vs1, vfloat8e5m2m1_t vs2, size_t vl) {
  return __riscv_vfwmmacc_vv_f16m4_f8e5m2m1(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 16 x half> @test_vfwmmacc_vv_f16m4_f8e4m3m1_f8e5m2m1(
// CHECK:         tail call <vscale x 16 x half> @llvm.riscv.vfwmmacc.nxv16f16.nxv8i8.i64(<vscale x 16 x half> {{%.*}}, <vscale x 8 x i8> {{%.*}}, <vscale x 8 x i8> {{%.*}}, i64 {{%.*}})
vfloat16m4_t test_vfwmmacc_vv_f16m4_f8e4m3m1_f8e5m2m1(vfloat16m4_t vd, vfloat8e4m3m1_t vs1, vfloat8e5m2m1_t vs2, size_t vl) {
  return __riscv_vfwmmacc_vv_f16m4_f8e4m3m1_f8e5m2m1(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 8 x half> @test_vfwmmacc_vv_f16m2_f8e5m2m8_f8e4m3m8(
// CHECK:         tail call <vscale x 8 x half> @llvm.riscv.vfwmmacc.nxv8f16.nxv64i8.i64(<vscale x 8 x half> {{%.*}}, <vscale x 64 x i8> {{%.*}}, <vscale x 64 x i8> {{%.*}}, i64 {{%.*}})
vfloat16m2_t test_vfwmmacc_vv_f16m2_f8e5m2m8_f8e4m3m8(vfloat16m2_t vd, vfloat8e5m2m8_t vs1, vfloat8e4m3m8_t vs2, size_t vl) {
  return __riscv_vfwmmacc_vv_f16m2_f8e5m2m8_f8e4m3m8(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 4 x half> @test_vfwmmacc_vv_f16m1_f8e4m3m4(
// CHECK:         tail call <vscale x 4 x half> @llvm.riscv.vfwmmacc.nxv4f16.nxv32i8.i64(<vscale x 4 x half> {{%.*}}, <vscale x 32 x i8> {{%.*}}, <vscale x 32 x i8> {{%.*}}, i64 {{%.*}})
vfloat16m1_t test_vfwmmacc_vv_f16m1_f8e4m3m4(vfloat16m1_t vd, vfloat8e4m3m4_t vs1, vfloat8e4m3m4_t vs2, size_t vl) {
  return __riscv_vfwmmacc_vv_f16m1_f8e4m3m4(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 16 x bfloat> @test_vfwmmacc_vv_bf16m4_f8e4m3m1(
// CHECK:         tail call <vscale x 16 x bfloat> @llvm.riscv.vfwmmacc.nxv16bf16.nxv8i8.i64(<vscale x 16 x bfloat> {{%.*}}, <vscale x 8 x i8> {{%.*}}, <vscale x 8 x i8> {{%.*}}, i64 {{%.*}})
vbfloat16m4_t test_vfwmmacc_vv_bf16m4_f8e4m3m1(vbfloat16m4_t vd, vfloat8e4m3m1_t vs1, vfloat8e4m3m1_t vs2, size_t vl) {
  return __riscv_vfwmmacc_vv_bf16m4_f8e4m3m1(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 4 x bfloat> @test_vfwmmacc_vv_bf16m1_f8e4m3m4(
// CHECK:         tail call <vscale x 4 x bfloat> @llvm.riscv.vfwmmacc.nxv4bf16.nxv32i8.i64(<vscale x 4 x bfloat> {{%.*}}, <vscale x 32 x i8> {{%.*}}, <vscale x 32 x i8> {{%.*}}, i64 {{%.*}})
vbfloat16m1_t test_vfwmmacc_vv_bf16m1_f8e4m3m4(vbfloat16m1_t vd, vfloat8e4m3m4_t vs1, vfloat8e4m3m4_t vs2, size_t vl) {
  return __riscv_vfwmmacc_vv_bf16m1_f8e4m3m4(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 32 x bfloat> @test_vfwmmacc_vv_bf16m8_f8e5m2m2(
// CHECK:         tail call <vscale x 32 x bfloat> @llvm.riscv.vfwmmacc.nxv32bf16.nxv16i8.i64(<vscale x 32 x bfloat> {{%.*}}, <vscale x 16 x i8> {{%.*}}, <vscale x 16 x i8> {{%.*}}, i64 {{%.*}})
vbfloat16m8_t test_vfwmmacc_vv_bf16m8_f8e5m2m2(vbfloat16m8_t vd, vfloat8e5m2m2_t vs1, vfloat8e5m2m2_t vs2, size_t vl) {
  return __riscv_vfwmmacc_vv_bf16m8_f8e5m2m2(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 16 x bfloat> @test_vfwmmacc_vv_bf16m4_f8e4m3m1_f8e5m2m1(
// CHECK:         tail call <vscale x 16 x bfloat> @llvm.riscv.vfwmmacc.nxv16bf16.nxv8i8.i64(<vscale x 16 x bfloat> {{%.*}}, <vscale x 8 x i8> {{%.*}}, <vscale x 8 x i8> {{%.*}}, i64 {{%.*}})
vbfloat16m4_t test_vfwmmacc_vv_bf16m4_f8e4m3m1_f8e5m2m1(vbfloat16m4_t vd, vfloat8e4m3m1_t vs1, vfloat8e5m2m1_t vs2, size_t vl) {
  return __riscv_vfwmmacc_vv_bf16m4_f8e4m3m1_f8e5m2m1(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 8 x bfloat> @test_vfwmmacc_vv_bf16m2_f8e5m2m1_f8e4m3m1(
// CHECK:         tail call <vscale x 8 x bfloat> @llvm.riscv.vfwmmacc.nxv8bf16.nxv8i8.i64(<vscale x 8 x bfloat> {{%.*}}, <vscale x 8 x i8> {{%.*}}, <vscale x 8 x i8> {{%.*}}, i64 {{%.*}})
vbfloat16m2_t test_vfwmmacc_vv_bf16m2_f8e5m2m1_f8e4m3m1(vbfloat16m2_t vd, vfloat8e5m2m1_t vs1, vfloat8e4m3m1_t vs2, size_t vl) {
  return __riscv_vfwmmacc_vv_bf16m2_f8e5m2m1_f8e4m3m1(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 4 x float> @test_vfqmmacc_vv_f32m2_f8e4m3m1(
// CHECK:         tail call <vscale x 4 x float> @llvm.riscv.vfqmmacc.nxv4f32.nxv8i8.i64(<vscale x 4 x float> {{%.*}}, <vscale x 8 x i8> {{%.*}}, <vscale x 8 x i8> {{%.*}}, i64 {{%.*}})
vfloat32m2_t test_vfqmmacc_vv_f32m2_f8e4m3m1(vfloat32m2_t vd, vfloat8e4m3m1_t vs1, vfloat8e4m3m1_t vs2, size_t vl) {
  return __riscv_vfqmmacc_vv_f32m2_f8e4m3m1(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 4 x float> @test_vfqmmacc_vv_f32m2_f8e4m3m4(
// CHECK:         tail call <vscale x 4 x float> @llvm.riscv.vfqmmacc.nxv4f32.nxv32i8.i64(<vscale x 4 x float> {{%.*}}, <vscale x 32 x i8> {{%.*}}, <vscale x 32 x i8> {{%.*}}, i64 {{%.*}})
vfloat32m2_t test_vfqmmacc_vv_f32m2_f8e4m3m4(vfloat32m2_t vd, vfloat8e4m3m4_t vs1, vfloat8e4m3m4_t vs2, size_t vl) {
  return __riscv_vfqmmacc_vv_f32m2_f8e4m3m4(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 4 x float> @test_vfqmmacc_vv_f32m2_f8e5m2m1(
// CHECK:         tail call <vscale x 4 x float> @llvm.riscv.vfqmmacc.nxv4f32.nxv8i8.i64(<vscale x 4 x float> {{%.*}}, <vscale x 8 x i8> {{%.*}}, <vscale x 8 x i8> {{%.*}}, i64 {{%.*}})
vfloat32m2_t test_vfqmmacc_vv_f32m2_f8e5m2m1(vfloat32m2_t vd, vfloat8e5m2m1_t vs1, vfloat8e5m2m1_t vs2, size_t vl) {
  return __riscv_vfqmmacc_vv_f32m2_f8e5m2m1(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 16 x float> @test_vfqmmacc_vv_f32m8_f8e4m3m2_f8e5m2m2(
// CHECK:         tail call <vscale x 16 x float> @llvm.riscv.vfqmmacc.nxv16f32.nxv16i8.i64(<vscale x 16 x float> {{%.*}}, <vscale x 16 x i8> {{%.*}}, <vscale x 16 x i8> {{%.*}}, i64 {{%.*}})
vfloat32m8_t test_vfqmmacc_vv_f32m8_f8e4m3m2_f8e5m2m2(vfloat32m8_t vd, vfloat8e4m3m2_t vs1, vfloat8e5m2m2_t vs2, size_t vl) {
  return __riscv_vfqmmacc_vv_f32m8_f8e4m3m2_f8e5m2m2(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 2 x float> @test_vfqmmacc_vv_f32m1_f8e5m2m8_f8e4m3m8(
// CHECK:         tail call <vscale x 2 x float> @llvm.riscv.vfqmmacc.nxv2f32.nxv64i8.i64(<vscale x 2 x float> {{%.*}}, <vscale x 64 x i8> {{%.*}}, <vscale x 64 x i8> {{%.*}}, i64 {{%.*}})
vfloat32m1_t test_vfqmmacc_vv_f32m1_f8e5m2m8_f8e4m3m8(vfloat32m1_t vd, vfloat8e5m2m8_t vs1, vfloat8e4m3m8_t vs2, size_t vl) {
  return __riscv_vfqmmacc_vv_f32m1_f8e5m2m8_f8e4m3m8(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 1 x double> @test_vf8wmmacc_vv_f64m1_f8e4m3m1(
// CHECK:         tail call <vscale x 1 x double> @llvm.riscv.vf8wmmacc.nxv1f64.nxv8i8.i64(<vscale x 1 x double> {{%.*}}, <vscale x 8 x i8> {{%.*}}, <vscale x 8 x i8> {{%.*}}, i64 {{%.*}})
vfloat64m1_t test_vf8wmmacc_vv_f64m1_f8e4m3m1(vfloat64m1_t vd, vfloat8e4m3m1_t vs1, vfloat8e4m3m1_t vs2, size_t vl) {
  return __riscv_vf8wmmacc_vv_f64m1_f8e4m3m1(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 8 x double> @test_vf8wmmacc_vv_f64m8_f8e5m2m2(
// CHECK:         tail call <vscale x 8 x double> @llvm.riscv.vf8wmmacc.nxv8f64.nxv16i8.i64(<vscale x 8 x double> {{%.*}}, <vscale x 16 x i8> {{%.*}}, <vscale x 16 x i8> {{%.*}}, i64 {{%.*}})
vfloat64m8_t test_vf8wmmacc_vv_f64m8_f8e5m2m2(vfloat64m8_t vd, vfloat8e5m2m2_t vs1, vfloat8e5m2m2_t vs2, size_t vl) {
  return __riscv_vf8wmmacc_vv_f64m8_f8e5m2m2(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 4 x double> @test_vf8wmmacc_vv_f64m4_f8e4m3m4_f8e5m2m4(
// CHECK:         tail call <vscale x 4 x double> @llvm.riscv.vf8wmmacc.nxv4f64.nxv32i8.i64(<vscale x 4 x double> {{%.*}}, <vscale x 32 x i8> {{%.*}}, <vscale x 32 x i8> {{%.*}}, i64 {{%.*}})
vfloat64m4_t test_vf8wmmacc_vv_f64m4_f8e4m3m4_f8e5m2m4(vfloat64m4_t vd, vfloat8e4m3m4_t vs1, vfloat8e5m2m4_t vs2, size_t vl) {
  return __riscv_vf8wmmacc_vv_f64m4_f8e4m3m4_f8e5m2m4(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 2 x double> @test_vf8wmmacc_vv_f64m2_f8e5m2m8_f8e4m3m8(
// CHECK:         tail call <vscale x 2 x double> @llvm.riscv.vf8wmmacc.nxv2f64.nxv64i8.i64(<vscale x 2 x double> {{%.*}}, <vscale x 64 x i8> {{%.*}}, <vscale x 64 x i8> {{%.*}}, i64 {{%.*}})
vfloat64m2_t test_vf8wmmacc_vv_f64m2_f8e5m2m8_f8e4m3m8(vfloat64m2_t vd, vfloat8e5m2m8_t vs1, vfloat8e4m3m8_t vs2, size_t vl) {
  return __riscv_vf8wmmacc_vv_f64m2_f8e5m2m8_f8e4m3m8(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 64 x i8> @test_vfmmacc_vv_f8e4m3m8(
// CHECK:         tail call <vscale x 64 x i8> @llvm.riscv.vfmmacc.nxv64i8.nxv8i8.i64(<vscale x 64 x i8> {{%.*}}, <vscale x 8 x i8> {{%.*}}, <vscale x 8 x i8> {{%.*}}, i64 {{%.*}})
vfloat8e4m3m8_t test_vfmmacc_vv_f8e4m3m8(vfloat8e4m3m8_t vd, vfloat8e4m3m1_t vs1, vfloat8e4m3m1_t vs2, size_t vl) {
  return __riscv_vfmmacc_vv_f8e4m3m8(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 32 x i8> @test_vfmmacc_vv_f8e5m2m4_lm2(
// CHECK:         tail call <vscale x 32 x i8> @llvm.riscv.vfmmacc.nxv32i8.nxv16i8.i64(<vscale x 32 x i8> {{%.*}}, <vscale x 16 x i8> {{%.*}}, <vscale x 16 x i8> {{%.*}}, i64 {{%.*}})
vfloat8e5m2m4_t test_vfmmacc_vv_f8e5m2m4_lm2(vfloat8e5m2m4_t vd, vfloat8e5m2m2_t vs1, vfloat8e5m2m2_t vs2, size_t vl) {
  return __riscv_vfmmacc_vv_f8e5m2m4_lm2(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 8 x i8> @test_vfmmacc_vv_f8e4m3m1_f8e5m2m4(
// CHECK:         tail call <vscale x 8 x i8> @llvm.riscv.vfmmacc.nxv8i8.nxv32i8.i64(<vscale x 8 x i8> {{%.*}}, <vscale x 32 x i8> {{%.*}}, <vscale x 32 x i8> {{%.*}}, i64 {{%.*}})
vfloat8e4m3m1_t test_vfmmacc_vv_f8e4m3m1_f8e5m2m4(vfloat8e4m3m1_t vd, vfloat8e5m2m4_t vs1, vfloat8e5m2m4_t vs2, size_t vl) {
  return __riscv_vfmmacc_vv_f8e4m3m1_f8e5m2m4(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 16 x i8> @test_vfmmacc_vv_f8e5m2m2_f8e4m3m1(
// CHECK:         tail call <vscale x 16 x i8> @llvm.riscv.vfmmacc.nxv16i8.nxv8i8.i64(<vscale x 16 x i8> {{%.*}}, <vscale x 8 x i8> {{%.*}}, <vscale x 8 x i8> {{%.*}}, i64 {{%.*}})
vfloat8e5m2m2_t test_vfmmacc_vv_f8e5m2m2_f8e4m3m1(vfloat8e5m2m2_t vd, vfloat8e4m3m1_t vs1, vfloat8e4m3m1_t vs2, size_t vl) {
  return __riscv_vfmmacc_vv_f8e5m2m2_f8e4m3m1(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 8 x i8> @test_vfmmacc_vv_f8e4m3m1_f8e4m3m1_f8e5m2m1(
// CHECK:         tail call <vscale x 8 x i8> @llvm.riscv.vfmmacc.nxv8i8.nxv8i8.i64(<vscale x 8 x i8> {{%.*}}, <vscale x 8 x i8> {{%.*}}, <vscale x 8 x i8> {{%.*}}, i64 {{%.*}})
vfloat8e4m3m1_t test_vfmmacc_vv_f8e4m3m1_f8e4m3m1_f8e5m2m1(vfloat8e4m3m1_t vd, vfloat8e4m3m1_t vs1, vfloat8e5m2m1_t vs2, size_t vl) {
  return __riscv_vfmmacc_vv_f8e4m3m1_f8e4m3m1_f8e5m2m1(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 64 x i8> @test_vfmmacc_vv_f8e5m2m8_f8e5m2m8_f8e4m3m8(
// CHECK:         tail call <vscale x 64 x i8> @llvm.riscv.vfmmacc.nxv64i8.nxv64i8.i64(<vscale x 64 x i8> {{%.*}}, <vscale x 64 x i8> {{%.*}}, <vscale x 64 x i8> {{%.*}}, i64 {{%.*}})
vfloat8e5m2m8_t test_vfmmacc_vv_f8e5m2m8_f8e5m2m8_f8e4m3m8(vfloat8e5m2m8_t vd, vfloat8e5m2m8_t vs1, vfloat8e4m3m8_t vs2, size_t vl) {
  return __riscv_vfmmacc_vv_f8e5m2m8_f8e5m2m8_f8e4m3m8(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 16 x i8> @test_vfmmacc_vv_f8e4m3m2_f8e5m2m1_f8e4m3m1(
// CHECK:         tail call <vscale x 16 x i8> @llvm.riscv.vfmmacc.nxv16i8.nxv8i8.i64(<vscale x 16 x i8> {{%.*}}, <vscale x 8 x i8> {{%.*}}, <vscale x 8 x i8> {{%.*}}, i64 {{%.*}})
vfloat8e4m3m2_t test_vfmmacc_vv_f8e4m3m2_f8e5m2m1_f8e4m3m1(vfloat8e4m3m2_t vd, vfloat8e5m2m1_t vs1, vfloat8e4m3m1_t vs2, size_t vl) {
  return __riscv_vfmmacc_vv_f8e4m3m2_f8e5m2m1_f8e4m3m1(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 4 x bfloat> @test_vfwmmacc_vv_bf16m1_f8e4m3m4_ovl(
// CHECK:         tail call <vscale x 4 x bfloat> @llvm.riscv.vfwmmacc.nxv4bf16.nxv32i8.i64(<vscale x 4 x bfloat> {{%.*}}, <vscale x 32 x i8> {{%.*}}, <vscale x 32 x i8> {{%.*}}, i64 {{%.*}})
vbfloat16m1_t test_vfwmmacc_vv_bf16m1_f8e4m3m4_ovl(vbfloat16m1_t vd, vfloat8e4m3m4_t vs1, vfloat8e4m3m4_t vs2, size_t vl) {
  return __riscv_vfwmmacc_vv(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 4 x float> @test_vfqmmacc_vv_f32m2_f8e4m3m4_ovl(
// CHECK:         tail call <vscale x 4 x float> @llvm.riscv.vfqmmacc.nxv4f32.nxv32i8.i64(<vscale x 4 x float> {{%.*}}, <vscale x 32 x i8> {{%.*}}, <vscale x 32 x i8> {{%.*}}, i64 {{%.*}})
vfloat32m2_t test_vfqmmacc_vv_f32m2_f8e4m3m4_ovl(vfloat32m2_t vd, vfloat8e4m3m4_t vs1, vfloat8e4m3m4_t vs2, size_t vl) {
  return __riscv_vfqmmacc_vv(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 16 x half> @test_vfwmmacc_vv_f16m4_f8e4m3m1_f8e5m2m1_ovl(
// CHECK:         tail call <vscale x 16 x half> @llvm.riscv.vfwmmacc.nxv16f16.nxv8i8.i64(<vscale x 16 x half> {{%.*}}, <vscale x 8 x i8> {{%.*}}, <vscale x 8 x i8> {{%.*}}, i64 {{%.*}})
vfloat16m4_t test_vfwmmacc_vv_f16m4_f8e4m3m1_f8e5m2m1_ovl(vfloat16m4_t vd, vfloat8e4m3m1_t vs1, vfloat8e5m2m1_t vs2, size_t vl) {
  return __riscv_vfwmmacc_vv(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 2 x double> @test_vf8wmmacc_vv_f64m2_f8e5m2m8_f8e4m3m8_ovl(
// CHECK:         tail call <vscale x 2 x double> @llvm.riscv.vf8wmmacc.nxv2f64.nxv64i8.i64(<vscale x 2 x double> {{%.*}}, <vscale x 64 x i8> {{%.*}}, <vscale x 64 x i8> {{%.*}}, i64 {{%.*}})
vfloat64m2_t test_vf8wmmacc_vv_f64m2_f8e5m2m8_f8e4m3m8_ovl(vfloat64m2_t vd, vfloat8e5m2m8_t vs1, vfloat8e4m3m8_t vs2, size_t vl) {
  return __riscv_vf8wmmacc_vv(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 8 x i8> @test_vfmmacc_vv_f8e4m3m1_f8e4m3m1_f8e5m2m1_ovl(
// CHECK:         tail call <vscale x 8 x i8> @llvm.riscv.vfmmacc.nxv8i8.nxv8i8.i64(<vscale x 8 x i8> {{%.*}}, <vscale x 8 x i8> {{%.*}}, <vscale x 8 x i8> {{%.*}}, i64 {{%.*}})
vfloat8e4m3m1_t test_vfmmacc_vv_f8e4m3m1_f8e4m3m1_f8e5m2m1_ovl(vfloat8e4m3m1_t vd, vfloat8e4m3m1_t vs1, vfloat8e5m2m1_t vs2, size_t vl) {
  return __riscv_vfmmacc_vv(vd, vs1, vs2, vl);
}

// CHECK-LABEL: define dso_local <vscale x 32 x i8> @test_vfmmacc_vv_f8e5m2m4_lm2_ovl(
// CHECK:         tail call <vscale x 32 x i8> @llvm.riscv.vfmmacc.nxv32i8.nxv16i8.i64(<vscale x 32 x i8> {{%.*}}, <vscale x 16 x i8> {{%.*}}, <vscale x 16 x i8> {{%.*}}, i64 {{%.*}})
vfloat8e5m2m4_t test_vfmmacc_vv_f8e5m2m4_lm2_ovl(vfloat8e5m2m4_t vd, vfloat8e5m2m2_t vs1, vfloat8e5m2m2_t vs2, size_t vl) {
  return __riscv_vfmmacc_vv(vd, vs1, vs2, vl);
}
