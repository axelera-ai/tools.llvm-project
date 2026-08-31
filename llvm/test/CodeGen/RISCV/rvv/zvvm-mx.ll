; RUN: llc -mtriple=riscv64 -mattr=+v,+zvfhmin,+experimental-zvvxofp8fp16mm,+experimental-zvvxofp8fp32mm,+experimental-zvvxofp8fp64mm,+experimental-zvvxi8fp16mm,+experimental-zvvxi8fp32mm,+experimental-zvvxi8fp64mm -verify-machineinstrs < %s | FileCheck %s

; Microscaled (vm=0, v0.scale) matrix MACs. The paired E8M0 block-scale
; operand (vuint16m1_t = <vscale x 4 x i16>) is pinned to v0 by ISel; the
; trailing bs immarg records the block size the caller compiled against and
; is dropped during selection (vtype.bs carries the architectural state).
; vtype (SEW / LMUL / lambda / bs / altfmt*) is pre-established by a separate
; vsetvl_matrix in real code; these tests exercise ISel, register allocation,
; and MC lowering only.

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"

; W=2 FP scaled: MXFP8 inputs (i8 container), FP16 accumulator.
; EMUL_C = 1, LMUL = 1.
define <vscale x 4 x half> @vfwmmacc_scaled_f16m1_i8m1(<vscale x 4 x half> %vd, <vscale x 8 x i8> %vs1, <vscale x 8 x i8> %vs2, <vscale x 4 x i16> %scale, i64 %vl) {
; CHECK-LABEL: vfwmmacc_scaled_f16m1_i8m1:
; CHECK: vmv1r.v v0, v11
; CHECK: vfwmmacc.vv v8, v9, v10, v0.scale
entry:
  %0 = call <vscale x 4 x half> @llvm.riscv.vfwmmacc.scaled.nxv4f16.nxv8i8.i64(<vscale x 4 x half> %vd, <vscale x 8 x i8> %vs1, <vscale x 8 x i8> %vs2, <vscale x 4 x i16> %scale, i64 0, i64 %vl)
  ret <vscale x 4 x half> %0
}

; W=2 FP scaled, BS=16 record (same encoding; bs is vtype state).
define <vscale x 4 x half> @vfwmmacc_scaled_f16m1_i8m1_bs16(<vscale x 4 x half> %vd, <vscale x 8 x i8> %vs1, <vscale x 8 x i8> %vs2, <vscale x 4 x i16> %scale, i64 %vl) {
; CHECK-LABEL: vfwmmacc_scaled_f16m1_i8m1_bs16:
; CHECK: vmv1r.v v0, v11
; CHECK: vfwmmacc.vv v8, v9, v10, v0.scale
entry:
  %0 = call <vscale x 4 x half> @llvm.riscv.vfwmmacc.scaled.nxv4f16.nxv8i8.i64(<vscale x 4 x half> %vd, <vscale x 8 x i8> %vs1, <vscale x 8 x i8> %vs2, <vscale x 4 x i16> %scale, i64 1, i64 %vl)
  ret <vscale x 4 x half> %0
}

; W=2 FP scaled at EMUL_C = 8, LMUL = 2.
define <vscale x 32 x half> @vfwmmacc_scaled_f16m8_i8m2(<vscale x 32 x half> %vd, <vscale x 16 x i8> %vs1, <vscale x 16 x i8> %vs2, <vscale x 4 x i16> %scale, i64 %vl) {
; CHECK-LABEL: vfwmmacc_scaled_f16m8_i8m2:
; CHECK: vmv1r.v v0, v20
; CHECK: vfwmmacc.vv v8, v16, v18, v0.scale
entry:
  %0 = call <vscale x 32 x half> @llvm.riscv.vfwmmacc.scaled.nxv32f16.nxv16i8.i64(<vscale x 32 x half> %vd, <vscale x 16 x i8> %vs1, <vscale x 16 x i8> %vs2, <vscale x 4 x i16> %scale, i64 0, i64 %vl)
  ret <vscale x 32 x half> %0
}

; W=4 FP scaled: MXFP8 inputs, FP32 accumulator. EMUL_C = 2, LMUL = 1.
define <vscale x 4 x float> @vfqmmacc_scaled_f32m2_i8m1(<vscale x 4 x float> %vd, <vscale x 8 x i8> %vs1, <vscale x 8 x i8> %vs2, <vscale x 4 x i16> %scale, i64 %vl) {
; CHECK-LABEL: vfqmmacc_scaled_f32m2_i8m1:
; CHECK: vmv1r.v v0, v12
; CHECK: vfqmmacc.vv v8, v10, v11, v0.scale
entry:
  %0 = call <vscale x 4 x float> @llvm.riscv.vfqmmacc.scaled.nxv4f32.nxv8i8.i64(<vscale x 4 x float> %vd, <vscale x 8 x i8> %vs1, <vscale x 8 x i8> %vs2, <vscale x 4 x i16> %scale, i64 0, i64 %vl)
  ret <vscale x 4 x float> %0
}

; W=8 FP scaled: MXFP8 inputs, FP64 accumulator. EMUL_C = 4, LMUL = 1.
; (A and B deliberately share one source value: input register groups may
; overlap each other, only vd/input overlap is forbidden.)
define <vscale x 4 x double> @vf8wmmacc_scaled_f64m4_i8m1(<vscale x 4 x double> %vd, <vscale x 8 x i8> %vs1, <vscale x 8 x i8> %vs2, <vscale x 4 x i16> %scale, i64 %vl) {
; CHECK-LABEL: vf8wmmacc_scaled_f64m4_i8m1:
; CHECK: vmv1r.v v0, v14
; CHECK: vf8wmmacc.vv v8, v12, v12, v0.scale
entry:
  %0 = call <vscale x 4 x double> @llvm.riscv.vf8wmmacc.scaled.nxv4f64.nxv8i8.i64(<vscale x 4 x double> %vd, <vscale x 8 x i8> %vs1, <vscale x 8 x i8> %vs1, <vscale x 4 x i16> %scale, i64 0, i64 %vl)
  ret <vscale x 4 x double> %0
}

; W=2 integer-input: MXINT8 inputs, FP16 accumulator. EMUL_C = 1, LMUL = 1.
define <vscale x 4 x half> @vfwimmacc_f16m1_i8m1(<vscale x 4 x half> %vd, <vscale x 8 x i8> %vs1, <vscale x 8 x i8> %vs2, <vscale x 4 x i16> %scale, i64 %vl) {
; CHECK-LABEL: vfwimmacc_f16m1_i8m1:
; CHECK: vmv1r.v v0, v11
; CHECK: vfwimmacc.vv v8, v9, v10, v0.scale
entry:
  %0 = call <vscale x 4 x half> @llvm.riscv.vfwimmacc.nxv4f16.nxv8i8.i64(<vscale x 4 x half> %vd, <vscale x 8 x i8> %vs1, <vscale x 8 x i8> %vs2, <vscale x 4 x i16> %scale, i64 0, i64 %vl)
  ret <vscale x 4 x half> %0
}

; W=4 integer-input: MXINT8 inputs, FP32 accumulator. EMUL_C = 4, LMUL = 8.
define <vscale x 8 x float> @vfqimmacc_f32m4_i8m8(<vscale x 8 x float> %vd, <vscale x 64 x i8> %vs1, <vscale x 64 x i8> %vs2, <vscale x 4 x i16> %scale, i64 %vl) {
; CHECK-LABEL: vfqimmacc_f32m4_i8m8:
; CHECK: vmv1r.v v0, v12
; CHECK: vfqimmacc.vv v8, v16, v24, v0.scale
entry:
  %0 = call <vscale x 8 x float> @llvm.riscv.vfqimmacc.nxv8f32.nxv64i8.i64(<vscale x 8 x float> %vd, <vscale x 64 x i8> %vs1, <vscale x 64 x i8> %vs2, <vscale x 4 x i16> %scale, i64 0, i64 %vl)
  ret <vscale x 8 x float> %0
}

; W=8 integer-input: MXINT8 inputs, FP64 accumulator. EMUL_C = 1, LMUL = 1.
define <vscale x 1 x double> @vf8wimmacc_f64m1_i8m1(<vscale x 1 x double> %vd, <vscale x 8 x i8> %vs1, <vscale x 8 x i8> %vs2, <vscale x 4 x i16> %scale, i64 %vl) {
; CHECK-LABEL: vf8wimmacc_f64m1_i8m1:
; CHECK: vmv1r.v v0, v11
; CHECK: vf8wimmacc.vv v8, v9, v10, v0.scale
entry:
  %0 = call <vscale x 1 x double> @llvm.riscv.vf8wimmacc.nxv1f64.nxv8i8.i64(<vscale x 1 x double> %vd, <vscale x 8 x i8> %vs1, <vscale x 8 x i8> %vs2, <vscale x 4 x i16> %scale, i64 0, i64 %vl)
  ret <vscale x 1 x double> %0
}

; EMUL_C = 16 column: m16 tuple accumulator, MXINT8 inputs, LMUL = 1.
define target("riscv.vector.tuple", <vscale x 64 x i8>, 2) @vfwimmacc_m16_i8m1(target("riscv.vector.tuple", <vscale x 64 x i8>, 2) %vd, <vscale x 8 x i8> %vs1, <vscale x 8 x i8> %vs2, <vscale x 4 x i16> %scale, i64 %vl) {
; CHECK-LABEL: vfwimmacc_m16_i8m1:
; CHECK: vmv1r.v v0, v10
; CHECK: vfwimmacc.vv v16, v8, v9, v0.scale
entry:
  %0 = call target("riscv.vector.tuple", <vscale x 64 x i8>, 2) @llvm.riscv.vfwimmacc.m16.triscv.vector.tuple_nxv64i8_2t.nxv8i8.i64(target("riscv.vector.tuple", <vscale x 64 x i8>, 2) %vd, <vscale x 8 x i8> %vs1, <vscale x 8 x i8> %vs2, <vscale x 4 x i16> %scale, i64 0, i64 %vl)
  ret target("riscv.vector.tuple", <vscale x 64 x i8>, 2) %0
}
