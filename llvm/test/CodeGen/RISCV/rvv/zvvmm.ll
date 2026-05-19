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
