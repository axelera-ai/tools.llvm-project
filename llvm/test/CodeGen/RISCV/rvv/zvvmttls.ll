; RUN: sed 's/iXLen/i32/g' %s | llc -mtriple=riscv32 -mattr=+v,+experimental-zvvmttls \
; RUN:   -verify-machineinstrs | FileCheck %s
; RUN: sed 's/iXLen/i64/g' %s | llc -mtriple=riscv64 -mattr=+v,+experimental-zvvmttls \
; RUN:   -verify-machineinstrs | FileCheck %s

; Zvvmttls transposing tile load/store intrinsics.
;
; See zvvmtls.ll for limitations.

declare <vscale x 2 x i32> @llvm.riscv.vmttl.nxv2i32.p0.iXLen(
  <vscale x 2 x i32>, ptr, iXLen, iXLen)
declare <vscale x 2 x i32> @llvm.riscv.vmttl.mask.nxv2i32.p0.iXLen(
  <vscale x 2 x i32>, ptr, iXLen, <vscale x 2 x i1>, iXLen, iXLen immarg)
declare void @llvm.riscv.vmtts.nxv2i32.p0.iXLen(
  <vscale x 2 x i32>, ptr, iXLen, iXLen)
declare void @llvm.riscv.vmtts.mask.nxv2i32.p0.iXLen(
  <vscale x 2 x i32>, ptr, iXLen, <vscale x 2 x i1>, iXLen)

define <vscale x 2 x i32> @test_vmttl_nxv2i32(<vscale x 2 x i32> %passthru,
                                              ptr %base, iXLen %ld,
                                              iXLen %vl) nounwind {
; CHECK-LABEL: test_vmttl_nxv2i32:
; CHECK:       vmttl.v v8, (a0), a1
; CHECK:       ret
  %r = call <vscale x 2 x i32> @llvm.riscv.vmttl.nxv2i32.p0.iXLen(
    <vscale x 2 x i32> %passthru, ptr %base, iXLen %ld, iXLen %vl)
  ret <vscale x 2 x i32> %r
}

define <vscale x 2 x i32> @test_vmttl_mask_nxv2i32(<vscale x 2 x i32> %passthru,
                                                   ptr %base, iXLen %ld,
                                                   <vscale x 2 x i1> %mask,
                                                   iXLen %vl) nounwind {
; CHECK-LABEL: test_vmttl_mask_nxv2i32:
; CHECK:       vmttl.v v8, (a0), a1, v0.t
; CHECK:       ret
  %r = call <vscale x 2 x i32> @llvm.riscv.vmttl.mask.nxv2i32.p0.iXLen(
    <vscale x 2 x i32> %passthru, ptr %base, iXLen %ld,
    <vscale x 2 x i1> %mask, iXLen %vl, iXLen 1)
  ret <vscale x 2 x i32> %r
}

define void @test_vmtts_nxv2i32(<vscale x 2 x i32> %v, ptr %base,
                                iXLen %ld, iXLen %vl) nounwind {
; CHECK-LABEL: test_vmtts_nxv2i32:
; CHECK:       vmtts.v v8, (a0), a1
; CHECK:       ret
  call void @llvm.riscv.vmtts.nxv2i32.p0.iXLen(
    <vscale x 2 x i32> %v, ptr %base, iXLen %ld, iXLen %vl)
  ret void
}

define void @test_vmtts_mask_nxv2i32(<vscale x 2 x i32> %v, ptr %base,
                                     iXLen %ld, <vscale x 2 x i1> %mask,
                                     iXLen %vl) nounwind {
; CHECK-LABEL: test_vmtts_mask_nxv2i32:
; CHECK:       vmtts.v v8, (a0), a1, v0.t
; CHECK:       ret
  call void @llvm.riscv.vmtts.mask.nxv2i32.p0.iXLen(
    <vscale x 2 x i32> %v, ptr %base, iXLen %ld,
    <vscale x 2 x i1> %mask, iXLen %vl)
  ret void
}
