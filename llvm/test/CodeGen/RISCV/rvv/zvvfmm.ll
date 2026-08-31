; RUN: sed 's/iXLen/i32/g' %s \
; RUN:   | llc -mtriple=riscv32 \
; RUN:        -mattr=+v,+zvfh,+zve64d,+experimental-zvvofp8mm,+experimental-zvvfp16mm,+experimental-zvvfp32mm,+experimental-zvvfp64mm,+experimental-zvvofp8fp16mm,+experimental-zvvfp16fp32mm,+experimental-zvvfp32fp64mm,+experimental-zvvofp8fp32mm,+experimental-zvvfp16fp64mm,+experimental-zvvofp8fp64mm \
; RUN:        -verify-machineinstrs | FileCheck %s
; RUN: sed 's/iXLen/i64/g' %s \
; RUN:   | llc -mtriple=riscv64 \
; RUN:        -mattr=+v,+zvfh,+zve64d,+experimental-zvvofp8mm,+experimental-zvvfp16mm,+experimental-zvvfp32mm,+experimental-zvvfp64mm,+experimental-zvvofp8fp16mm,+experimental-zvvfp16fp32mm,+experimental-zvvfp32fp64mm,+experimental-zvvofp8fp32mm,+experimental-zvvfp16fp64mm,+experimental-zvvofp8fp64mm \
; RUN:        -verify-machineinstrs | FileCheck %s

; Zvvfmm floating-point matrix multiply-accumulate intrinsics.
;
; Phase 1 does not integrate with RISCVInsertVSETVLI, so no vsetvli is emitted
; for these intrinsics; the caller is responsible for programming vtype
; (SEW, LMUL, lambda, altfmt_A/altfmt_B) via vsetvl before invoking the MAC.
; The tests therefore only verify the mnemonic, source registers, and
; tied-operand constraint ($vd = $vd_wb).
;
; OFP8 inputs at LMUL=1 use the nxv8i8 IR container — LLVM has no first-class
; FP8 vector type for E4M3 / E5M2, and vtype.altfmt selects the FP8
; interpretation at execution time. This mirrors the integer Mm-acc IR types.

declare <vscale x 2 x float> @llvm.riscv.vfmmacc.nxv2f32.nxv2f32(
  <vscale x 2 x float>, <vscale x 2 x float>, <vscale x 2 x float>, iXLen)

define <vscale x 2 x float> @test_vfmmacc_nxv2f32(<vscale x 2 x float> %c,
                                                  <vscale x 2 x float> %a,
                                                  <vscale x 2 x float> %b,
                                                  iXLen %vl) nounwind {
; CHECK-LABEL: test_vfmmacc_nxv2f32:
; CHECK:       vfmmacc.vv v8, v9, v10
; CHECK:       ret
  %r = call <vscale x 2 x float> @llvm.riscv.vfmmacc.nxv2f32.nxv2f32(
    <vscale x 2 x float> %c,
    <vscale x 2 x float> %a,
    <vscale x 2 x float> %b,
    iXLen %vl)
  ret <vscale x 2 x float> %r
}

declare <vscale x 2 x float> @llvm.riscv.vfwmmacc.nxv2f32.nxv4f16(
  <vscale x 2 x float>, <vscale x 4 x half>, <vscale x 4 x half>, iXLen)

define <vscale x 2 x float> @test_vfwmmacc_nxv2f32(<vscale x 2 x float> %c,
                                                   <vscale x 4 x half> %a,
                                                   <vscale x 4 x half> %b,
                                                   iXLen %vl) nounwind {
; CHECK-LABEL: test_vfwmmacc_nxv2f32:
; CHECK:       vfwmmacc.vv v8, v9, v10
; CHECK:       ret
  %r = call <vscale x 2 x float> @llvm.riscv.vfwmmacc.nxv2f32.nxv4f16(
    <vscale x 2 x float> %c,
    <vscale x 4 x half> %a,
    <vscale x 4 x half> %b,
    iXLen %vl)
  ret <vscale x 2 x float> %r
}

declare <vscale x 2 x float> @llvm.riscv.vfqmmacc.nxv2f32.nxv8i8(
  <vscale x 2 x float>, <vscale x 8 x i8>, <vscale x 8 x i8>, iXLen)

define <vscale x 2 x float> @test_vfqmmacc_nxv2f32(<vscale x 2 x float> %c,
                                                   <vscale x 8 x i8> %a,
                                                   <vscale x 8 x i8> %b,
                                                   iXLen %vl) nounwind {
; CHECK-LABEL: test_vfqmmacc_nxv2f32:
; CHECK:       vfqmmacc.vv v8, v9, v10
; CHECK:       ret
  %r = call <vscale x 2 x float> @llvm.riscv.vfqmmacc.nxv2f32.nxv8i8(
    <vscale x 2 x float> %c,
    <vscale x 8 x i8> %a,
    <vscale x 8 x i8> %b,
    iXLen %vl)
  ret <vscale x 2 x float> %r
}

declare <vscale x 1 x double> @llvm.riscv.vf8wmmacc.nxv1f64.nxv8i8(
  <vscale x 1 x double>, <vscale x 8 x i8>, <vscale x 8 x i8>, iXLen)

define <vscale x 1 x double> @test_vf8wmmacc_nxv1f64(<vscale x 1 x double> %c,
                                                     <vscale x 8 x i8> %a,
                                                     <vscale x 8 x i8> %b,
                                                     iXLen %vl) nounwind {
; CHECK-LABEL: test_vf8wmmacc_nxv1f64:
; CHECK:       vf8wmmacc.vv v8, v9, v10
; CHECK:       ret
  %r = call <vscale x 1 x double> @llvm.riscv.vf8wmmacc.nxv1f64.nxv8i8(
    <vscale x 1 x double> %c,
    <vscale x 8 x i8> %a,
    <vscale x 8 x i8> %b,
    iXLen %vl)
  ret <vscale x 1 x double> %r
}

; Non-widening vfmmacc.vv at EMUL_C ≠ LMUL — independent vd / vs1+vs2 register
; group multipliers. Spot-check representative cells.

declare <vscale x 4 x float> @llvm.riscv.vfmmacc.nxv4f32.nxv2f32(
  <vscale x 4 x float>, <vscale x 2 x float>, <vscale x 2 x float>, iXLen)

define <vscale x 4 x float> @test_vfmmacc_m2_lm1(<vscale x 4 x float> %c,
                                                 <vscale x 2 x float> %a,
                                                 <vscale x 2 x float> %b,
                                                 iXLen %vl) nounwind {
; CHECK-LABEL: test_vfmmacc_m2_lm1:
; CHECK:       vfmmacc.vv v8, v10, v11
; CHECK:       ret
  %r = call <vscale x 4 x float> @llvm.riscv.vfmmacc.nxv4f32.nxv2f32(
    <vscale x 4 x float> %c,
    <vscale x 2 x float> %a,
    <vscale x 2 x float> %b,
    iXLen %vl)
  ret <vscale x 4 x float> %r
}

declare <vscale x 2 x float> @llvm.riscv.vfmmacc.nxv2f32.nxv8f32(
  <vscale x 2 x float>, <vscale x 8 x float>, <vscale x 8 x float>, iXLen)

define <vscale x 2 x float> @test_vfmmacc_m1_lm4(<vscale x 2 x float> %c,
                                                 <vscale x 8 x float> %a,
                                                 <vscale x 8 x float> %b,
                                                 iXLen %vl) nounwind {
; CHECK-LABEL: test_vfmmacc_m1_lm4:
; CHECK:       vfmmacc.vv v8, v12, v16
; CHECK:       ret
  %r = call <vscale x 2 x float> @llvm.riscv.vfmmacc.nxv2f32.nxv8f32(
    <vscale x 2 x float> %c,
    <vscale x 8 x float> %a,
    <vscale x 8 x float> %b,
    iXLen %vl)
  ret <vscale x 2 x float> %r
}

declare <vscale x 16 x float> @llvm.riscv.vfmmacc.nxv16f32.nxv16f32(
  <vscale x 16 x float>, <vscale x 16 x float>, <vscale x 16 x float>, iXLen)

define <vscale x 16 x float> @test_vfmmacc_m8_lm8(<vscale x 16 x float> %c,
                                                  <vscale x 16 x float> %a,
                                                  <vscale x 16 x float> %b,
                                                  iXLen %vl) nounwind {
; CHECK-LABEL: test_vfmmacc_m8_lm8:
; CHECK:       vfmmacc.vv v8, v16, v24
; CHECK:       ret
  %r = call <vscale x 16 x float> @llvm.riscv.vfmmacc.nxv16f32.nxv16f32(
    <vscale x 16 x float> %c,
    <vscale x 16 x float> %a,
    <vscale x 16 x float> %b,
    iXLen %vl)
  ret <vscale x 16 x float> %r
}

; Widening FP MACs at EMUL_C ≠ LMUL: same two-axis register-class scheme as
; the non-widening grid, plus the (SEW(C), EEW(A/B)) axis. OFP8 inputs use
; the i8 integer container. Spot-check corner and mid cells.

declare <vscale x 2 x float> @llvm.riscv.vfwmmacc.nxv2f32.nxv32f16(
  <vscale x 2 x float>, <vscale x 32 x half>, <vscale x 32 x half>, iXLen)

define <vscale x 2 x float> @test_vfwmmacc_f32_m1_lm8(<vscale x 2 x float> %c,
                                                      <vscale x 32 x half> %a,
                                                      <vscale x 32 x half> %b,
                                                      iXLen %vl) nounwind {
; CHECK-LABEL: test_vfwmmacc_f32_m1_lm8:
; CHECK:       vfwmmacc.vv v8, v16, v24
; CHECK:       ret
  %r = call <vscale x 2 x float> @llvm.riscv.vfwmmacc.nxv2f32.nxv32f16(
    <vscale x 2 x float> %c,
    <vscale x 32 x half> %a,
    <vscale x 32 x half> %b,
    iXLen %vl)
  ret <vscale x 2 x float> %r
}

declare <vscale x 16 x float> @llvm.riscv.vfwmmacc.nxv16f32.nxv4f16(
  <vscale x 16 x float>, <vscale x 4 x half>, <vscale x 4 x half>, iXLen)

define <vscale x 16 x float> @test_vfwmmacc_f32_m8_lm1(<vscale x 16 x float> %c,
                                                       <vscale x 4 x half> %a,
                                                       <vscale x 4 x half> %b,
                                                       iXLen %vl) nounwind {
; CHECK-LABEL: test_vfwmmacc_f32_m8_lm1:
; CHECK:       vfwmmacc.vv v8, v16, v17
; CHECK:       ret
  %r = call <vscale x 16 x float> @llvm.riscv.vfwmmacc.nxv16f32.nxv4f16(
    <vscale x 16 x float> %c,
    <vscale x 4 x half> %a,
    <vscale x 4 x half> %b,
    iXLen %vl)
  ret <vscale x 16 x float> %r
}

declare <vscale x 32 x half> @llvm.riscv.vfwmmacc.nxv32f16.nxv64i8(
  <vscale x 32 x half>, <vscale x 64 x i8>, <vscale x 64 x i8>, iXLen)

define <vscale x 32 x half> @test_vfwmmacc_f16_ofp8_m8_lm8(<vscale x 32 x half> %c,
                                                           <vscale x 64 x i8> %a,
                                                           <vscale x 64 x i8> %b,
                                                           iXLen %vl) nounwind {
; CHECK-LABEL: test_vfwmmacc_f16_ofp8_m8_lm8:
; CHECK:       vfwmmacc.vv v8, v16, v24
; CHECK:       ret
  %r = call <vscale x 32 x half> @llvm.riscv.vfwmmacc.nxv32f16.nxv64i8(
    <vscale x 32 x half> %c,
    <vscale x 64 x i8> %a,
    <vscale x 64 x i8> %b,
    iXLen %vl)
  ret <vscale x 32 x half> %r
}

declare <vscale x 4 x double> @llvm.riscv.vfwmmacc.nxv4f64.nxv4f32(
  <vscale x 4 x double>, <vscale x 4 x float>, <vscale x 4 x float>, iXLen)

define <vscale x 4 x double> @test_vfwmmacc_f64_m4_lm2(<vscale x 4 x double> %c,
                                                       <vscale x 4 x float> %a,
                                                       <vscale x 4 x float> %b,
                                                       iXLen %vl) nounwind {
; CHECK-LABEL: test_vfwmmacc_f64_m4_lm2:
; CHECK:       vfwmmacc.vv v8, v12, v14
; CHECK:       ret
  %r = call <vscale x 4 x double> @llvm.riscv.vfwmmacc.nxv4f64.nxv4f32(
    <vscale x 4 x double> %c,
    <vscale x 4 x float> %a,
    <vscale x 4 x float> %b,
    iXLen %vl)
  ret <vscale x 4 x double> %r
}

declare <vscale x 2 x float> @llvm.riscv.vfqmmacc.nxv2f32.nxv64i8(
  <vscale x 2 x float>, <vscale x 64 x i8>, <vscale x 64 x i8>, iXLen)

define <vscale x 2 x float> @test_vfqmmacc_f32_m1_lm8(<vscale x 2 x float> %c,
                                                      <vscale x 64 x i8> %a,
                                                      <vscale x 64 x i8> %b,
                                                      iXLen %vl) nounwind {
; CHECK-LABEL: test_vfqmmacc_f32_m1_lm8:
; CHECK:       vfqmmacc.vv v8, v16, v24
; CHECK:       ret
  %r = call <vscale x 2 x float> @llvm.riscv.vfqmmacc.nxv2f32.nxv64i8(
    <vscale x 2 x float> %c,
    <vscale x 64 x i8> %a,
    <vscale x 64 x i8> %b,
    iXLen %vl)
  ret <vscale x 2 x float> %r
}

declare <vscale x 16 x float> @llvm.riscv.vfqmmacc.nxv16f32.nxv8i8(
  <vscale x 16 x float>, <vscale x 8 x i8>, <vscale x 8 x i8>, iXLen)

define <vscale x 16 x float> @test_vfqmmacc_f32_m8_lm1(<vscale x 16 x float> %c,
                                                       <vscale x 8 x i8> %a,
                                                       <vscale x 8 x i8> %b,
                                                       iXLen %vl) nounwind {
; CHECK-LABEL: test_vfqmmacc_f32_m8_lm1:
; CHECK:       vfqmmacc.vv v8, v16, v17
; CHECK:       ret
  %r = call <vscale x 16 x float> @llvm.riscv.vfqmmacc.nxv16f32.nxv8i8(
    <vscale x 16 x float> %c,
    <vscale x 8 x i8> %a,
    <vscale x 8 x i8> %b,
    iXLen %vl)
  ret <vscale x 16 x float> %r
}

declare <vscale x 2 x double> @llvm.riscv.vfqmmacc.nxv2f64.nxv16f16(
  <vscale x 2 x double>, <vscale x 16 x half>, <vscale x 16 x half>, iXLen)

define <vscale x 2 x double> @test_vfqmmacc_f64_m2_lm4(<vscale x 2 x double> %c,
                                                       <vscale x 16 x half> %a,
                                                       <vscale x 16 x half> %b,
                                                       iXLen %vl) nounwind {
; CHECK-LABEL: test_vfqmmacc_f64_m2_lm4:
; CHECK:       vfqmmacc.vv v8, v12, v16
; CHECK:       ret
  %r = call <vscale x 2 x double> @llvm.riscv.vfqmmacc.nxv2f64.nxv16f16(
    <vscale x 2 x double> %c,
    <vscale x 16 x half> %a,
    <vscale x 16 x half> %b,
    iXLen %vl)
  ret <vscale x 2 x double> %r
}

declare <vscale x 1 x double> @llvm.riscv.vf8wmmacc.nxv1f64.nxv64i8(
  <vscale x 1 x double>, <vscale x 64 x i8>, <vscale x 64 x i8>, iXLen)

define <vscale x 1 x double> @test_vf8wmmacc_m1_lm8(<vscale x 1 x double> %c,
                                                    <vscale x 64 x i8> %a,
                                                    <vscale x 64 x i8> %b,
                                                    iXLen %vl) nounwind {
; CHECK-LABEL: test_vf8wmmacc_m1_lm8:
; CHECK:       vf8wmmacc.vv v8, v16, v24
; CHECK:       ret
  %r = call <vscale x 1 x double> @llvm.riscv.vf8wmmacc.nxv1f64.nxv64i8(
    <vscale x 1 x double> %c,
    <vscale x 64 x i8> %a,
    <vscale x 64 x i8> %b,
    iXLen %vl)
  ret <vscale x 1 x double> %r
}

declare <vscale x 8 x double> @llvm.riscv.vf8wmmacc.nxv8f64.nxv8i8(
  <vscale x 8 x double>, <vscale x 8 x i8>, <vscale x 8 x i8>, iXLen)

define <vscale x 8 x double> @test_vf8wmmacc_m8_lm1(<vscale x 8 x double> %c,
                                                    <vscale x 8 x i8> %a,
                                                    <vscale x 8 x i8> %b,
                                                    iXLen %vl) nounwind {
; CHECK-LABEL: test_vf8wmmacc_m8_lm1:
; CHECK:       vf8wmmacc.vv v8, v16, v17
; CHECK:       ret
  %r = call <vscale x 8 x double> @llvm.riscv.vf8wmmacc.nxv8f64.nxv8i8(
    <vscale x 8 x double> %c,
    <vscale x 8 x i8> %a,
    <vscale x 8 x i8> %b,
    iXLen %vl)
  ret <vscale x 8 x double> %r
}

declare <vscale x 4 x double> @llvm.riscv.vf8wmmacc.nxv4f64.nxv32i8(
  <vscale x 4 x double>, <vscale x 32 x i8>, <vscale x 32 x i8>, iXLen)

define <vscale x 4 x double> @test_vf8wmmacc_m4_lm4(<vscale x 4 x double> %c,
                                                    <vscale x 32 x i8> %a,
                                                    <vscale x 32 x i8> %b,
                                                    iXLen %vl) nounwind {
; CHECK-LABEL: test_vf8wmmacc_m4_lm4:
; CHECK:       vf8wmmacc.vv v8, v12, v16
; CHECK:       ret
  %r = call <vscale x 4 x double> @llvm.riscv.vf8wmmacc.nxv4f64.nxv32i8(
    <vscale x 4 x double> %c,
    <vscale x 32 x i8> %a,
    <vscale x 32 x i8> %b,
    iXLen %vl)
  ret <vscale x 4 x double> %r
}

; --- Non-widening FP MAC SEW coverage beyond f32 (Phase 5) ---

declare <vscale x 4 x half> @llvm.riscv.vfmmacc.nxv4f16.nxv4f16(
  <vscale x 4 x half>, <vscale x 4 x half>, <vscale x 4 x half>, iXLen)
declare <vscale x 32 x half> @llvm.riscv.vfmmacc.nxv32f16.nxv8f16(
  <vscale x 32 x half>, <vscale x 8 x half>, <vscale x 8 x half>, iXLen)
declare <vscale x 2 x double> @llvm.riscv.vfmmacc.nxv2f64.nxv4f64(
  <vscale x 2 x double>, <vscale x 4 x double>, <vscale x 4 x double>, iXLen)
declare <vscale x 8 x double> @llvm.riscv.vfmmacc.nxv8f64.nxv1f64(
  <vscale x 8 x double>, <vscale x 1 x double>, <vscale x 1 x double>, iXLen)
declare <vscale x 8 x i8> @llvm.riscv.vfmmacc.nxv8i8.nxv8i8(
  <vscale x 8 x i8>, <vscale x 8 x i8>, <vscale x 8 x i8>, iXLen)
declare <vscale x 32 x i8> @llvm.riscv.vfmmacc.nxv32i8.nxv16i8(
  <vscale x 32 x i8>, <vscale x 16 x i8>, <vscale x 16 x i8>, iXLen)

define <vscale x 4 x half> @test_vfmmacc_f16_m1_lm1(<vscale x 4 x half> %c,
                                                    <vscale x 4 x half> %a,
                                                    <vscale x 4 x half> %b,
                                                    iXLen %vl) nounwind {
; CHECK-LABEL: test_vfmmacc_f16_m1_lm1:
; CHECK:       vfmmacc.vv v8, v9, v10
; CHECK:       ret
  %r = call <vscale x 4 x half> @llvm.riscv.vfmmacc.nxv4f16.nxv4f16(
    <vscale x 4 x half> %c,
    <vscale x 4 x half> %a,
    <vscale x 4 x half> %b,
    iXLen %vl)
  ret <vscale x 4 x half> %r
}

define <vscale x 32 x half> @test_vfmmacc_f16_m8_lm2(<vscale x 32 x half> %c,
                                                     <vscale x 8 x half> %a,
                                                     <vscale x 8 x half> %b,
                                                     iXLen %vl) nounwind {
; CHECK-LABEL: test_vfmmacc_f16_m8_lm2:
; CHECK:       vfmmacc.vv v8, v16, v18
; CHECK:       ret
  %r = call <vscale x 32 x half> @llvm.riscv.vfmmacc.nxv32f16.nxv8f16(
    <vscale x 32 x half> %c,
    <vscale x 8 x half> %a,
    <vscale x 8 x half> %b,
    iXLen %vl)
  ret <vscale x 32 x half> %r
}

define <vscale x 2 x double> @test_vfmmacc_f64_m2_lm4(<vscale x 2 x double> %c,
                                                      <vscale x 4 x double> %a,
                                                      <vscale x 4 x double> %b,
                                                      iXLen %vl) nounwind {
; CHECK-LABEL: test_vfmmacc_f64_m2_lm4:
; CHECK:       vfmmacc.vv v8, v12, v16
; CHECK:       ret
  %r = call <vscale x 2 x double> @llvm.riscv.vfmmacc.nxv2f64.nxv4f64(
    <vscale x 2 x double> %c,
    <vscale x 4 x double> %a,
    <vscale x 4 x double> %b,
    iXLen %vl)
  ret <vscale x 2 x double> %r
}

define <vscale x 8 x double> @test_vfmmacc_f64_m8_lm1(<vscale x 8 x double> %c,
                                                      <vscale x 1 x double> %a,
                                                      <vscale x 1 x double> %b,
                                                      iXLen %vl) nounwind {
; CHECK-LABEL: test_vfmmacc_f64_m8_lm1:
; CHECK:       vfmmacc.vv v8, v16, v17
; CHECK:       ret
  %r = call <vscale x 8 x double> @llvm.riscv.vfmmacc.nxv8f64.nxv1f64(
    <vscale x 8 x double> %c,
    <vscale x 1 x double> %a,
    <vscale x 1 x double> %b,
    iXLen %vl)
  ret <vscale x 8 x double> %r
}

; SEW=8 accumulator row: OFP8 C tile with OFP8 inputs (Zvvofp8mm). No IR FP8
; element type exists, so both sides use the i8 container; vtype.altfmt and
; altfmt_A/altfmt_B select the E4M3/E5M2 interpretation at execution time.
define <vscale x 8 x i8> @test_vfmmacc_ofp8_m1_lm1(<vscale x 8 x i8> %c,
                                                   <vscale x 8 x i8> %a,
                                                   <vscale x 8 x i8> %b,
                                                   iXLen %vl) nounwind {
; CHECK-LABEL: test_vfmmacc_ofp8_m1_lm1:
; CHECK:       vfmmacc.vv v8, v9, v10
; CHECK:       ret
  %r = call <vscale x 8 x i8> @llvm.riscv.vfmmacc.nxv8i8.nxv8i8(
    <vscale x 8 x i8> %c,
    <vscale x 8 x i8> %a,
    <vscale x 8 x i8> %b,
    iXLen %vl)
  ret <vscale x 8 x i8> %r
}

define <vscale x 32 x i8> @test_vfmmacc_ofp8_m4_lm2(<vscale x 32 x i8> %c,
                                                    <vscale x 16 x i8> %a,
                                                    <vscale x 16 x i8> %b,
                                                    iXLen %vl) nounwind {
; CHECK-LABEL: test_vfmmacc_ofp8_m4_lm2:
; CHECK:       vfmmacc.vv v8, v12, v14
; CHECK:       ret
  %r = call <vscale x 32 x i8> @llvm.riscv.vfmmacc.nxv32i8.nxv16i8(
    <vscale x 32 x i8> %c,
    <vscale x 16 x i8> %a,
    <vscale x 16 x i8> %b,
    iXLen %vl)
  ret <vscale x 32 x i8> %r
}
