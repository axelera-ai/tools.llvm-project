; Per-type extension gating of the Zvvm BF16 MAC ISel rows: each row selects
; only with its own per-type extension. The Zvvfmm family flag and the FP16
; sibling extension do not enable a BF16 row.
;
; RUN: rm -rf %t && split-file %s %t
;
; vfmmacc BF16 <- BF16 needs Zvvbf16mm (Zvvfp16mm is not enough).
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvfbfmin,+experimental-zvvbf16mm \
; RUN:   -verify-machineinstrs < %t/vfmmacc.ll | FileCheck %s --check-prefix=OK
; RUN: not --crash llc -mtriple=riscv64 -mattr=+v,+zvfbfmin,+zvfh,+experimental-zvvfmm,+experimental-zvvfp16mm \
; RUN:   < %t/vfmmacc.ll 2>&1 | FileCheck %s --check-prefix=NOSEL
;
; vfwmmacc BF16 <- OFP8 needs Zvvofp8bf16mm (Zvvofp8fp16mm is not enough).
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvfbfmin,+experimental-zvvofp8bf16mm \
; RUN:   -verify-machineinstrs < %t/vfwmmacc-ofp8.ll | FileCheck %s --check-prefix=OK
; RUN: not --crash llc -mtriple=riscv64 -mattr=+v,+zvfbfmin,+experimental-zvvofp8fp16mm \
; RUN:   < %t/vfwmmacc-ofp8.ll 2>&1 | FileCheck %s --check-prefix=NOSEL
;
; vfwmmacc FP32 <- BF16 needs Zvvbf16fp32mm (Zvvfp16fp32mm is not enough).
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvfbfmin,+experimental-zvvbf16fp32mm \
; RUN:   -verify-machineinstrs < %t/vfwmmacc-bf16.ll | FileCheck %s --check-prefix=OK
; RUN: not --crash llc -mtriple=riscv64 -mattr=+v,+zvfbfmin,+experimental-zvvfp16fp32mm \
; RUN:   < %t/vfwmmacc-bf16.ll 2>&1 | FileCheck %s --check-prefix=NOSEL
;
; vfqmmacc FP64 <- BF16 needs Zvvbf16fp64mm.
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvfbfmin,+experimental-zvvbf16fp64mm \
; RUN:   -verify-machineinstrs < %t/vfqmmacc-bf16.ll | FileCheck %s --check-prefix=OK
; RUN: not --crash llc -mtriple=riscv64 -mattr=+v,+zvfbfmin,+experimental-zvvfp16fp64mm \
; RUN:   < %t/vfqmmacc-bf16.ll 2>&1 | FileCheck %s --check-prefix=NOSEL
;
; MX: vfwmmacc vm=0 BF16 <- MXFP8 needs Zvvxofp8bf16mm (its unscaled base
; Zvvofp8bf16mm does not authorize the vm=0 form).
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvfbfmin,+experimental-zvvxofp8bf16mm \
; RUN:   -verify-machineinstrs < %t/vfwmmacc-scaled.ll | FileCheck %s --check-prefix=OK
; RUN: not --crash llc -mtriple=riscv64 -mattr=+v,+zvfbfmin,+experimental-zvvofp8bf16mm,+experimental-zvvxofp8fp16mm \
; RUN:   < %t/vfwmmacc-scaled.ll 2>&1 | FileCheck %s --check-prefix=NOSEL
;
; MX: vfwimmacc BF16 <- MXINT8 needs Zvvxi8bf16mm.
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvfbfmin,+experimental-zvvxi8bf16mm \
; RUN:   -verify-machineinstrs < %t/vfwimmacc.ll | FileCheck %s --check-prefix=OK
; RUN: not --crash llc -mtriple=riscv64 -mattr=+v,+zvfbfmin,+experimental-zvvxi8fp16mm \
; RUN:   < %t/vfwimmacc.ll 2>&1 | FileCheck %s --check-prefix=NOSEL

; OK: vsetvli zero, a0, e{{16alt|32|64}}, m1,
; OK: {{vfmmacc|vfwmmacc|vfqmmacc|vfwimmacc}}.vv
; NOSEL: LLVM ERROR: Cannot select: intrinsic %llvm.riscv.

;--- vfmmacc.ll
define <vscale x 4 x bfloat> @f(<vscale x 4 x bfloat> %c, <vscale x 4 x bfloat> %a, <vscale x 4 x bfloat> %b, i64 %vl) {
  %r = call <vscale x 4 x bfloat> @llvm.riscv.vfmmacc.nxv4bf16.nxv4bf16.i64(<vscale x 4 x bfloat> %c, <vscale x 4 x bfloat> %a, <vscale x 4 x bfloat> %b, i64 %vl)
  ret <vscale x 4 x bfloat> %r
}

;--- vfwmmacc-ofp8.ll
define <vscale x 4 x bfloat> @f(<vscale x 4 x bfloat> %c, <vscale x 8 x i8> %a, <vscale x 8 x i8> %b, i64 %vl) {
  %r = call <vscale x 4 x bfloat> @llvm.riscv.vfwmmacc.nxv4bf16.nxv8i8.i64(<vscale x 4 x bfloat> %c, <vscale x 8 x i8> %a, <vscale x 8 x i8> %b, i64 %vl)
  ret <vscale x 4 x bfloat> %r
}

;--- vfwmmacc-bf16.ll
define <vscale x 2 x float> @f(<vscale x 2 x float> %c, <vscale x 4 x bfloat> %a, <vscale x 4 x bfloat> %b, i64 %vl) {
  %r = call <vscale x 2 x float> @llvm.riscv.vfwmmacc.nxv2f32.nxv4bf16.i64(<vscale x 2 x float> %c, <vscale x 4 x bfloat> %a, <vscale x 4 x bfloat> %b, i64 %vl)
  ret <vscale x 2 x float> %r
}

;--- vfqmmacc-bf16.ll
define <vscale x 1 x double> @f(<vscale x 1 x double> %c, <vscale x 4 x bfloat> %a, <vscale x 4 x bfloat> %b, i64 %vl) {
  %r = call <vscale x 1 x double> @llvm.riscv.vfqmmacc.nxv1f64.nxv4bf16.i64(<vscale x 1 x double> %c, <vscale x 4 x bfloat> %a, <vscale x 4 x bfloat> %b, i64 %vl)
  ret <vscale x 1 x double> %r
}

;--- vfwmmacc-scaled.ll
define <vscale x 4 x bfloat> @f(<vscale x 4 x bfloat> %c, <vscale x 8 x i8> %a, <vscale x 8 x i8> %b, <vscale x 4 x i16> %s, i64 %vl) {
  %r = call <vscale x 4 x bfloat> @llvm.riscv.vfwmmacc.scaled.nxv4bf16.nxv8i8.i64(<vscale x 4 x bfloat> %c, <vscale x 8 x i8> %a, <vscale x 8 x i8> %b, <vscale x 4 x i16> %s, i64 0, i64 %vl)
  ret <vscale x 4 x bfloat> %r
}

;--- vfwimmacc.ll
define <vscale x 4 x bfloat> @f(<vscale x 4 x bfloat> %c, <vscale x 8 x i8> %a, <vscale x 8 x i8> %b, <vscale x 4 x i16> %s, i64 %vl) {
  %r = call <vscale x 4 x bfloat> @llvm.riscv.vfwimmacc.nxv4bf16.nxv8i8.i64(<vscale x 4 x bfloat> %c, <vscale x 8 x i8> %a, <vscale x 8 x i8> %b, <vscale x 4 x i16> %s, i64 0, i64 %vl)
  ret <vscale x 4 x bfloat> %r
}
