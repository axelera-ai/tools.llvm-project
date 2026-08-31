; RUN: sed 's/iXLen/i32/g' %s | llc -mtriple=riscv32 -mattr=+v,+experimental-zvvmm \
; RUN:   -verify-machineinstrs | FileCheck %s
; RUN: sed 's/iXLen/i64/g' %s | llc -mtriple=riscv64 -mattr=+v,+experimental-zvvmm \
; RUN:   -verify-machineinstrs | FileCheck %s

; Zvvmm integer matrix multiply-accumulate intrinsics.
;
; Phase 1 does not integrate with RISCVInsertVSETVLI, so no vsetvli is emitted
; for these intrinsics; the caller is responsible for programming vtype
; (SEW, LMUL, lambda, altfmt_A/altfmt_B) via vsetvl before invoking the MAC.
; The tests therefore only verify the mnemonic, source registers, and
; tied-operand constraint ($vd = $vd_wb).

declare <vscale x 2 x i32> @llvm.riscv.vmmacc.nxv2i32.nxv2i32(
  <vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i32>, iXLen)

define <vscale x 2 x i32> @test_vmmacc_nxv2i32(<vscale x 2 x i32> %c,
                                               <vscale x 2 x i32> %a,
                                               <vscale x 2 x i32> %b,
                                               iXLen %vl) nounwind {
; CHECK-LABEL: test_vmmacc_nxv2i32:
; CHECK:       vmmacc.vv v8, v9, v10
; CHECK:       ret
  %r = call <vscale x 2 x i32> @llvm.riscv.vmmacc.nxv2i32.nxv2i32(
    <vscale x 2 x i32> %c,
    <vscale x 2 x i32> %a,
    <vscale x 2 x i32> %b,
    iXLen %vl)
  ret <vscale x 2 x i32> %r
}

declare <vscale x 2 x i32> @llvm.riscv.vwmmacc.nxv2i32.nxv4i16(
  <vscale x 2 x i32>, <vscale x 4 x i16>, <vscale x 4 x i16>, iXLen)

define <vscale x 2 x i32> @test_vwmmacc_nxv2i32(<vscale x 2 x i32> %c,
                                                <vscale x 4 x i16> %a,
                                                <vscale x 4 x i16> %b,
                                                iXLen %vl) nounwind {
; CHECK-LABEL: test_vwmmacc_nxv2i32:
; CHECK:       vwmmacc.vv v8, v9, v10
; CHECK:       ret
  %r = call <vscale x 2 x i32> @llvm.riscv.vwmmacc.nxv2i32.nxv4i16(
    <vscale x 2 x i32> %c,
    <vscale x 4 x i16> %a,
    <vscale x 4 x i16> %b,
    iXLen %vl)
  ret <vscale x 2 x i32> %r
}

declare <vscale x 2 x i32> @llvm.riscv.vqmmacc.nxv2i32.nxv8i8(
  <vscale x 2 x i32>, <vscale x 8 x i8>, <vscale x 8 x i8>, iXLen)

define <vscale x 2 x i32> @test_vqmmacc_nxv2i32(<vscale x 2 x i32> %c,
                                                <vscale x 8 x i8> %a,
                                                <vscale x 8 x i8> %b,
                                                iXLen %vl) nounwind {
; CHECK-LABEL: test_vqmmacc_nxv2i32:
; CHECK:       vqmmacc.vv v8, v9, v10
; CHECK:       ret
  %r = call <vscale x 2 x i32> @llvm.riscv.vqmmacc.nxv2i32.nxv8i8(
    <vscale x 2 x i32> %c,
    <vscale x 8 x i8> %a,
    <vscale x 8 x i8> %b,
    iXLen %vl)
  ret <vscale x 2 x i32> %r
}

declare <vscale x 1 x i64> @llvm.riscv.v8wmmacc.nxv1i64.nxv8i8(
  <vscale x 1 x i64>, <vscale x 8 x i8>, <vscale x 8 x i8>, iXLen)

define <vscale x 1 x i64> @test_v8wmmacc_nxv1i64(<vscale x 1 x i64> %c,
                                                 <vscale x 8 x i8> %a,
                                                 <vscale x 8 x i8> %b,
                                                 iXLen %vl) nounwind {
; CHECK-LABEL: test_v8wmmacc_nxv1i64:
; CHECK:       v8wmmacc.vv v8, v9, v10
; CHECK:       ret
  %r = call <vscale x 1 x i64> @llvm.riscv.v8wmmacc.nxv1i64.nxv8i8(
    <vscale x 1 x i64> %c,
    <vscale x 8 x i8> %a,
    <vscale x 8 x i8> %b,
    iXLen %vl)
  ret <vscale x 1 x i64> %r
}

; Non-widening vmmacc.vv at EMUL_C ≠ LMUL: vd register class is VR/VRM2/VRM4/
; VRM8 per EMUL_C, vs1/vs2 register class is VR/VRM2/VRM4/VRM8 per LMUL — the
; matrix MAC encoding stores only the base register number; vtype.LMUL drives
; the actual register-group size at execution. Spot-check representative cells.

declare <vscale x 4 x i32> @llvm.riscv.vmmacc.nxv4i32.nxv2i32(
  <vscale x 4 x i32>, <vscale x 2 x i32>, <vscale x 2 x i32>, iXLen)

define <vscale x 4 x i32> @test_vmmacc_m2_lm1(<vscale x 4 x i32> %c,
                                              <vscale x 2 x i32> %a,
                                              <vscale x 2 x i32> %b,
                                              iXLen %vl) nounwind {
; CHECK-LABEL: test_vmmacc_m2_lm1:
; CHECK:       vmmacc.vv v8, v10, v11
; CHECK:       ret
  %r = call <vscale x 4 x i32> @llvm.riscv.vmmacc.nxv4i32.nxv2i32(
    <vscale x 4 x i32> %c,
    <vscale x 2 x i32> %a,
    <vscale x 2 x i32> %b,
    iXLen %vl)
  ret <vscale x 4 x i32> %r
}

declare <vscale x 2 x i32> @llvm.riscv.vmmacc.nxv2i32.nxv4i32(
  <vscale x 2 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, iXLen)

define <vscale x 2 x i32> @test_vmmacc_m1_lm2(<vscale x 2 x i32> %c,
                                              <vscale x 4 x i32> %a,
                                              <vscale x 4 x i32> %b,
                                              iXLen %vl) nounwind {
; CHECK-LABEL: test_vmmacc_m1_lm2:
; CHECK:       vmmacc.vv v8, v10, v12
; CHECK:       ret
  %r = call <vscale x 2 x i32> @llvm.riscv.vmmacc.nxv2i32.nxv4i32(
    <vscale x 2 x i32> %c,
    <vscale x 4 x i32> %a,
    <vscale x 4 x i32> %b,
    iXLen %vl)
  ret <vscale x 2 x i32> %r
}

declare <vscale x 8 x i32> @llvm.riscv.vmmacc.nxv8i32.nxv16i32(
  <vscale x 8 x i32>, <vscale x 16 x i32>, <vscale x 16 x i32>, iXLen)

define <vscale x 8 x i32> @test_vmmacc_m4_lm8(<vscale x 8 x i32> %c,
                                              <vscale x 16 x i32> %a,
                                              <vscale x 16 x i32> %b,
                                              iXLen %vl) nounwind {
; CHECK-LABEL: test_vmmacc_m4_lm8:
; CHECK:       vmmacc.vv v8, v16, v24
; CHECK:       ret
  %r = call <vscale x 8 x i32> @llvm.riscv.vmmacc.nxv8i32.nxv16i32(
    <vscale x 8 x i32> %c,
    <vscale x 16 x i32> %a,
    <vscale x 16 x i32> %b,
    iXLen %vl)
  ret <vscale x 8 x i32> %r
}

declare <vscale x 16 x i32> @llvm.riscv.vmmacc.nxv16i32.nxv2i32(
  <vscale x 16 x i32>, <vscale x 2 x i32>, <vscale x 2 x i32>, iXLen)

define <vscale x 16 x i32> @test_vmmacc_m8_lm1(<vscale x 16 x i32> %c,
                                               <vscale x 2 x i32> %a,
                                               <vscale x 2 x i32> %b,
                                               iXLen %vl) nounwind {
; CHECK-LABEL: test_vmmacc_m8_lm1:
; CHECK:       vmmacc.vv v8, v16, v17
; CHECK:       ret
  %r = call <vscale x 16 x i32> @llvm.riscv.vmmacc.nxv16i32.nxv2i32(
    <vscale x 16 x i32> %c,
    <vscale x 2 x i32> %a,
    <vscale x 2 x i32> %b,
    iXLen %vl)
  ret <vscale x 16 x i32> %r
}

; Widening MACs at EMUL_C ≠ LMUL: same two-axis register-class scheme as the
; non-widening grid, plus the (SEW(C), EEW(A/B)) axis. Spot-check corner and
; mid cells across the covered SEW combinations.

declare <vscale x 2 x i32> @llvm.riscv.vwmmacc.nxv2i32.nxv32i16(
  <vscale x 2 x i32>, <vscale x 32 x i16>, <vscale x 32 x i16>, iXLen)

define <vscale x 2 x i32> @test_vwmmacc_i32_m1_lm8(<vscale x 2 x i32> %c,
                                                   <vscale x 32 x i16> %a,
                                                   <vscale x 32 x i16> %b,
                                                   iXLen %vl) nounwind {
; CHECK-LABEL: test_vwmmacc_i32_m1_lm8:
; CHECK:       vwmmacc.vv v8, v16, v24
; CHECK:       ret
  %r = call <vscale x 2 x i32> @llvm.riscv.vwmmacc.nxv2i32.nxv32i16(
    <vscale x 2 x i32> %c,
    <vscale x 32 x i16> %a,
    <vscale x 32 x i16> %b,
    iXLen %vl)
  ret <vscale x 2 x i32> %r
}

declare <vscale x 16 x i32> @llvm.riscv.vwmmacc.nxv16i32.nxv4i16(
  <vscale x 16 x i32>, <vscale x 4 x i16>, <vscale x 4 x i16>, iXLen)

define <vscale x 16 x i32> @test_vwmmacc_i32_m8_lm1(<vscale x 16 x i32> %c,
                                                    <vscale x 4 x i16> %a,
                                                    <vscale x 4 x i16> %b,
                                                    iXLen %vl) nounwind {
; CHECK-LABEL: test_vwmmacc_i32_m8_lm1:
; CHECK:       vwmmacc.vv v8, v16, v17
; CHECK:       ret
  %r = call <vscale x 16 x i32> @llvm.riscv.vwmmacc.nxv16i32.nxv4i16(
    <vscale x 16 x i32> %c,
    <vscale x 4 x i16> %a,
    <vscale x 4 x i16> %b,
    iXLen %vl)
  ret <vscale x 16 x i32> %r
}

declare <vscale x 32 x i16> @llvm.riscv.vwmmacc.nxv32i16.nxv64i8(
  <vscale x 32 x i16>, <vscale x 64 x i8>, <vscale x 64 x i8>, iXLen)

define <vscale x 32 x i16> @test_vwmmacc_i16_m8_lm8(<vscale x 32 x i16> %c,
                                                    <vscale x 64 x i8> %a,
                                                    <vscale x 64 x i8> %b,
                                                    iXLen %vl) nounwind {
; CHECK-LABEL: test_vwmmacc_i16_m8_lm8:
; CHECK:       vwmmacc.vv v8, v16, v24
; CHECK:       ret
  %r = call <vscale x 32 x i16> @llvm.riscv.vwmmacc.nxv32i16.nxv64i8(
    <vscale x 32 x i16> %c,
    <vscale x 64 x i8> %a,
    <vscale x 64 x i8> %b,
    iXLen %vl)
  ret <vscale x 32 x i16> %r
}

declare <vscale x 4 x i64> @llvm.riscv.vwmmacc.nxv4i64.nxv4i32(
  <vscale x 4 x i64>, <vscale x 4 x i32>, <vscale x 4 x i32>, iXLen)

define <vscale x 4 x i64> @test_vwmmacc_i64_m4_lm2(<vscale x 4 x i64> %c,
                                                   <vscale x 4 x i32> %a,
                                                   <vscale x 4 x i32> %b,
                                                   iXLen %vl) nounwind {
; CHECK-LABEL: test_vwmmacc_i64_m4_lm2:
; CHECK:       vwmmacc.vv v8, v12, v14
; CHECK:       ret
  %r = call <vscale x 4 x i64> @llvm.riscv.vwmmacc.nxv4i64.nxv4i32(
    <vscale x 4 x i64> %c,
    <vscale x 4 x i32> %a,
    <vscale x 4 x i32> %b,
    iXLen %vl)
  ret <vscale x 4 x i64> %r
}

declare <vscale x 2 x i32> @llvm.riscv.vqmmacc.nxv2i32.nxv64i8(
  <vscale x 2 x i32>, <vscale x 64 x i8>, <vscale x 64 x i8>, iXLen)

define <vscale x 2 x i32> @test_vqmmacc_i32_m1_lm8(<vscale x 2 x i32> %c,
                                                   <vscale x 64 x i8> %a,
                                                   <vscale x 64 x i8> %b,
                                                   iXLen %vl) nounwind {
; CHECK-LABEL: test_vqmmacc_i32_m1_lm8:
; CHECK:       vqmmacc.vv v8, v16, v24
; CHECK:       ret
  %r = call <vscale x 2 x i32> @llvm.riscv.vqmmacc.nxv2i32.nxv64i8(
    <vscale x 2 x i32> %c,
    <vscale x 64 x i8> %a,
    <vscale x 64 x i8> %b,
    iXLen %vl)
  ret <vscale x 2 x i32> %r
}

declare <vscale x 16 x i32> @llvm.riscv.vqmmacc.nxv16i32.nxv8i8(
  <vscale x 16 x i32>, <vscale x 8 x i8>, <vscale x 8 x i8>, iXLen)

define <vscale x 16 x i32> @test_vqmmacc_i32_m8_lm1(<vscale x 16 x i32> %c,
                                                    <vscale x 8 x i8> %a,
                                                    <vscale x 8 x i8> %b,
                                                    iXLen %vl) nounwind {
; CHECK-LABEL: test_vqmmacc_i32_m8_lm1:
; CHECK:       vqmmacc.vv v8, v16, v17
; CHECK:       ret
  %r = call <vscale x 16 x i32> @llvm.riscv.vqmmacc.nxv16i32.nxv8i8(
    <vscale x 16 x i32> %c,
    <vscale x 8 x i8> %a,
    <vscale x 8 x i8> %b,
    iXLen %vl)
  ret <vscale x 16 x i32> %r
}

declare <vscale x 2 x i64> @llvm.riscv.vqmmacc.nxv2i64.nxv16i16(
  <vscale x 2 x i64>, <vscale x 16 x i16>, <vscale x 16 x i16>, iXLen)

define <vscale x 2 x i64> @test_vqmmacc_i64_m2_lm4(<vscale x 2 x i64> %c,
                                                   <vscale x 16 x i16> %a,
                                                   <vscale x 16 x i16> %b,
                                                   iXLen %vl) nounwind {
; CHECK-LABEL: test_vqmmacc_i64_m2_lm4:
; CHECK:       vqmmacc.vv v8, v12, v16
; CHECK:       ret
  %r = call <vscale x 2 x i64> @llvm.riscv.vqmmacc.nxv2i64.nxv16i16(
    <vscale x 2 x i64> %c,
    <vscale x 16 x i16> %a,
    <vscale x 16 x i16> %b,
    iXLen %vl)
  ret <vscale x 2 x i64> %r
}

declare <vscale x 8 x i64> @llvm.riscv.vqmmacc.nxv8i64.nxv32i16(
  <vscale x 8 x i64>, <vscale x 32 x i16>, <vscale x 32 x i16>, iXLen)

define <vscale x 8 x i64> @test_vqmmacc_i64_m8_lm8(<vscale x 8 x i64> %c,
                                                   <vscale x 32 x i16> %a,
                                                   <vscale x 32 x i16> %b,
                                                   iXLen %vl) nounwind {
; CHECK-LABEL: test_vqmmacc_i64_m8_lm8:
; CHECK:       vqmmacc.vv v8, v16, v24
; CHECK:       ret
  %r = call <vscale x 8 x i64> @llvm.riscv.vqmmacc.nxv8i64.nxv32i16(
    <vscale x 8 x i64> %c,
    <vscale x 32 x i16> %a,
    <vscale x 32 x i16> %b,
    iXLen %vl)
  ret <vscale x 8 x i64> %r
}

declare <vscale x 1 x i64> @llvm.riscv.v8wmmacc.nxv1i64.nxv64i8(
  <vscale x 1 x i64>, <vscale x 64 x i8>, <vscale x 64 x i8>, iXLen)

define <vscale x 1 x i64> @test_v8wmmacc_m1_lm8(<vscale x 1 x i64> %c,
                                                <vscale x 64 x i8> %a,
                                                <vscale x 64 x i8> %b,
                                                iXLen %vl) nounwind {
; CHECK-LABEL: test_v8wmmacc_m1_lm8:
; CHECK:       v8wmmacc.vv v8, v16, v24
; CHECK:       ret
  %r = call <vscale x 1 x i64> @llvm.riscv.v8wmmacc.nxv1i64.nxv64i8(
    <vscale x 1 x i64> %c,
    <vscale x 64 x i8> %a,
    <vscale x 64 x i8> %b,
    iXLen %vl)
  ret <vscale x 1 x i64> %r
}

declare <vscale x 8 x i64> @llvm.riscv.v8wmmacc.nxv8i64.nxv8i8(
  <vscale x 8 x i64>, <vscale x 8 x i8>, <vscale x 8 x i8>, iXLen)

define <vscale x 8 x i64> @test_v8wmmacc_m8_lm1(<vscale x 8 x i64> %c,
                                                <vscale x 8 x i8> %a,
                                                <vscale x 8 x i8> %b,
                                                iXLen %vl) nounwind {
; CHECK-LABEL: test_v8wmmacc_m8_lm1:
; CHECK:       v8wmmacc.vv v8, v16, v17
; CHECK:       ret
  %r = call <vscale x 8 x i64> @llvm.riscv.v8wmmacc.nxv8i64.nxv8i8(
    <vscale x 8 x i64> %c,
    <vscale x 8 x i8> %a,
    <vscale x 8 x i8> %b,
    iXLen %vl)
  ret <vscale x 8 x i64> %r
}

declare <vscale x 4 x i64> @llvm.riscv.v8wmmacc.nxv4i64.nxv32i8(
  <vscale x 4 x i64>, <vscale x 32 x i8>, <vscale x 32 x i8>, iXLen)

define <vscale x 4 x i64> @test_v8wmmacc_m4_lm4(<vscale x 4 x i64> %c,
                                                <vscale x 32 x i8> %a,
                                                <vscale x 32 x i8> %b,
                                                iXLen %vl) nounwind {
; CHECK-LABEL: test_v8wmmacc_m4_lm4:
; CHECK:       v8wmmacc.vv v8, v12, v16
; CHECK:       ret
  %r = call <vscale x 4 x i64> @llvm.riscv.v8wmmacc.nxv4i64.nxv32i8(
    <vscale x 4 x i64> %c,
    <vscale x 32 x i8> %a,
    <vscale x 32 x i8> %b,
    iXLen %vl)
  ret <vscale x 4 x i64> %r
}
