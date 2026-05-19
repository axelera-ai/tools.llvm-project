; RUN: sed 's/iXLen/i32/g' %s | llc -mtriple=riscv32 -mattr=+v,+experimental-zvvmtls \
; RUN:   -verify-machineinstrs | FileCheck %s
; RUN: sed 's/iXLen/i64/g' %s | llc -mtriple=riscv64 -mattr=+v,+experimental-zvvmtls \
; RUN:   -verify-machineinstrs | FileCheck %s

; Zvvmtls order-preserving tile load/store intrinsics.
;
; Phase 2 does not integrate with RISCVInsertVSETVLI for the new vtype fields
; (lambda, altfmt_A, altfmt_B). Callers must program vtype via vsetvl before
; invoking these intrinsics. Tests verify the mnemonic and operand routing
; only.

declare <vscale x 2 x i32> @llvm.riscv.vmtl.nxv2i32.p0.iXLen(
  <vscale x 2 x i32>, ptr, iXLen, iXLen)
declare <vscale x 2 x i32> @llvm.riscv.vmtl.l4.nxv2i32.p0.iXLen(
  <vscale x 2 x i32>, ptr, iXLen, iXLen)
declare <vscale x 2 x i32> @llvm.riscv.vmtl.mask.nxv2i32.p0.iXLen(
  <vscale x 2 x i32>, ptr, iXLen, <vscale x 2 x i1>, iXLen, iXLen immarg)
declare void @llvm.riscv.vmts.nxv2i32.p0.iXLen(
  <vscale x 2 x i32>, ptr, iXLen, iXLen)
declare void @llvm.riscv.vmts.mask.nxv2i32.p0.iXLen(
  <vscale x 2 x i32>, ptr, iXLen, <vscale x 2 x i1>, iXLen)

define <vscale x 2 x i32> @test_vmtl_nxv2i32(<vscale x 2 x i32> %passthru,
                                             ptr %base, iXLen %ld,
                                             iXLen %vl) nounwind {
; CHECK-LABEL: test_vmtl_nxv2i32:
; CHECK:       vmtl.v v8, (a0), a1
; CHECK:       ret
  %r = call <vscale x 2 x i32> @llvm.riscv.vmtl.nxv2i32.p0.iXLen(
    <vscale x 2 x i32> %passthru, ptr %base, iXLen %ld, iXLen %vl)
  ret <vscale x 2 x i32> %r
}

define <vscale x 2 x i32> @test_vmtl_l4_nxv2i32(<vscale x 2 x i32> %passthru,
                                                ptr %base, iXLen %ld,
                                                iXLen %vl) nounwind {
; CHECK-LABEL: test_vmtl_l4_nxv2i32:
; CHECK:       vmtl.v v8, (a0), a1, L4
; CHECK:       ret
  %r = call <vscale x 2 x i32> @llvm.riscv.vmtl.l4.nxv2i32.p0.iXLen(
    <vscale x 2 x i32> %passthru, ptr %base, iXLen %ld, iXLen %vl)
  ret <vscale x 2 x i32> %r
}

define <vscale x 2 x i32> @test_vmtl_mask_nxv2i32(<vscale x 2 x i32> %passthru,
                                                  ptr %base, iXLen %ld,
                                                  <vscale x 2 x i1> %mask,
                                                  iXLen %vl) nounwind {
; CHECK-LABEL: test_vmtl_mask_nxv2i32:
; CHECK:       vmtl.v v8, (a0), a1, v0.t
; CHECK:       ret
  %r = call <vscale x 2 x i32> @llvm.riscv.vmtl.mask.nxv2i32.p0.iXLen(
    <vscale x 2 x i32> %passthru, ptr %base, iXLen %ld,
    <vscale x 2 x i1> %mask, iXLen %vl, iXLen 1)
  ret <vscale x 2 x i32> %r
}

define void @test_vmts_nxv2i32(<vscale x 2 x i32> %v, ptr %base,
                               iXLen %ld, iXLen %vl) nounwind {
; CHECK-LABEL: test_vmts_nxv2i32:
; CHECK:       vmts.v v8, (a0), a1
; CHECK:       ret
  call void @llvm.riscv.vmts.nxv2i32.p0.iXLen(
    <vscale x 2 x i32> %v, ptr %base, iXLen %ld, iXLen %vl)
  ret void
}

define void @test_vmts_mask_nxv2i32(<vscale x 2 x i32> %v, ptr %base,
                                    iXLen %ld, <vscale x 2 x i1> %mask,
                                    iXLen %vl) nounwind {
; CHECK-LABEL: test_vmts_mask_nxv2i32:
; CHECK:       vmts.v v8, (a0), a1, v0.t
; CHECK:       ret
  call void @llvm.riscv.vmts.mask.nxv2i32.p0.iXLen(
    <vscale x 2 x i32> %v, ptr %base, iXLen %ld,
    <vscale x 2 x i1> %mask, iXLen %vl)
  ret void
}
