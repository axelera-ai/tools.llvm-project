; RUN: sed 's/iXLen/i32/g' %s | llc -mtriple=riscv32 \
; RUN:   -mattr=+v,+zvfhmin,+experimental-zvvmttls \
; RUN:   -verify-machineinstrs | FileCheck %s
; RUN: sed 's/iXLen/i64/g' %s | llc -mtriple=riscv64 \
; RUN:   -mattr=+v,+zvfhmin,+experimental-zvvmttls \
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

; Element-type coverage for vmttl.v / vmtts.v. Mnemonic encoding is identical
; for every SEW (SEW comes from vtype) — these checks just confirm each ISel
; pattern matches the right LMUL=1 IR vector type. The masked i8 cases also
; exercise the per-SEW mask container type.

declare <vscale x 8 x i8>    @llvm.riscv.vmttl.nxv8i8.p0.iXLen(
  <vscale x 8 x i8>,    ptr, iXLen, iXLen)
declare <vscale x 8 x i8>    @llvm.riscv.vmttl.mask.nxv8i8.p0.iXLen(
  <vscale x 8 x i8>,    ptr, iXLen, <vscale x 8 x i1>, iXLen, iXLen immarg)
declare <vscale x 4 x i16>   @llvm.riscv.vmttl.nxv4i16.p0.iXLen(
  <vscale x 4 x i16>,   ptr, iXLen, iXLen)
declare <vscale x 1 x i64>   @llvm.riscv.vmttl.nxv1i64.p0.iXLen(
  <vscale x 1 x i64>,   ptr, iXLen, iXLen)
declare <vscale x 4 x half>  @llvm.riscv.vmttl.nxv4f16.p0.iXLen(
  <vscale x 4 x half>,  ptr, iXLen, iXLen)
declare <vscale x 2 x float> @llvm.riscv.vmttl.nxv2f32.p0.iXLen(
  <vscale x 2 x float>, ptr, iXLen, iXLen)
declare <vscale x 1 x double> @llvm.riscv.vmttl.nxv1f64.p0.iXLen(
  <vscale x 1 x double>, ptr, iXLen, iXLen)

declare void @llvm.riscv.vmtts.nxv8i8.p0.iXLen(
  <vscale x 8 x i8>,    ptr, iXLen, iXLen)
declare void @llvm.riscv.vmtts.mask.nxv8i8.p0.iXLen(
  <vscale x 8 x i8>,    ptr, iXLen, <vscale x 8 x i1>, iXLen)
declare void @llvm.riscv.vmtts.nxv4i16.p0.iXLen(
  <vscale x 4 x i16>,   ptr, iXLen, iXLen)
declare void @llvm.riscv.vmtts.nxv1i64.p0.iXLen(
  <vscale x 1 x i64>,   ptr, iXLen, iXLen)
declare void @llvm.riscv.vmtts.nxv4f16.p0.iXLen(
  <vscale x 4 x half>,  ptr, iXLen, iXLen)
declare void @llvm.riscv.vmtts.nxv2f32.p0.iXLen(
  <vscale x 2 x float>, ptr, iXLen, iXLen)
declare void @llvm.riscv.vmtts.nxv1f64.p0.iXLen(
  <vscale x 1 x double>, ptr, iXLen, iXLen)

define <vscale x 8 x i8> @test_vmttl_nxv8i8(<vscale x 8 x i8> %passthru,
                                             ptr %base, iXLen %ld,
                                             iXLen %vl) nounwind {
; CHECK-LABEL: test_vmttl_nxv8i8:
; CHECK:       vmttl.v v8, (a0), a1
; CHECK:       ret
  %r = call <vscale x 8 x i8> @llvm.riscv.vmttl.nxv8i8.p0.iXLen(
    <vscale x 8 x i8> %passthru, ptr %base, iXLen %ld, iXLen %vl)
  ret <vscale x 8 x i8> %r
}

define <vscale x 8 x i8> @test_vmttl_mask_nxv8i8(<vscale x 8 x i8> %passthru,
                                                  ptr %base, iXLen %ld,
                                                  <vscale x 8 x i1> %mask,
                                                  iXLen %vl) nounwind {
; CHECK-LABEL: test_vmttl_mask_nxv8i8:
; CHECK:       vmttl.v v8, (a0), a1, v0.t
; CHECK:       ret
  %r = call <vscale x 8 x i8> @llvm.riscv.vmttl.mask.nxv8i8.p0.iXLen(
    <vscale x 8 x i8> %passthru, ptr %base, iXLen %ld,
    <vscale x 8 x i1> %mask, iXLen %vl, iXLen 1)
  ret <vscale x 8 x i8> %r
}

define <vscale x 4 x i16> @test_vmttl_nxv4i16(<vscale x 4 x i16> %passthru,
                                               ptr %base, iXLen %ld,
                                               iXLen %vl) nounwind {
; CHECK-LABEL: test_vmttl_nxv4i16:
; CHECK:       vmttl.v v8, (a0), a1
; CHECK:       ret
  %r = call <vscale x 4 x i16> @llvm.riscv.vmttl.nxv4i16.p0.iXLen(
    <vscale x 4 x i16> %passthru, ptr %base, iXLen %ld, iXLen %vl)
  ret <vscale x 4 x i16> %r
}

define <vscale x 1 x i64> @test_vmttl_nxv1i64(<vscale x 1 x i64> %passthru,
                                               ptr %base, iXLen %ld,
                                               iXLen %vl) nounwind {
; CHECK-LABEL: test_vmttl_nxv1i64:
; CHECK:       vmttl.v v8, (a0), a1
; CHECK:       ret
  %r = call <vscale x 1 x i64> @llvm.riscv.vmttl.nxv1i64.p0.iXLen(
    <vscale x 1 x i64> %passthru, ptr %base, iXLen %ld, iXLen %vl)
  ret <vscale x 1 x i64> %r
}

define <vscale x 4 x half> @test_vmttl_nxv4f16(<vscale x 4 x half> %passthru,
                                                ptr %base, iXLen %ld,
                                                iXLen %vl) nounwind {
; CHECK-LABEL: test_vmttl_nxv4f16:
; CHECK:       vmttl.v v8, (a0), a1
; CHECK:       ret
  %r = call <vscale x 4 x half> @llvm.riscv.vmttl.nxv4f16.p0.iXLen(
    <vscale x 4 x half> %passthru, ptr %base, iXLen %ld, iXLen %vl)
  ret <vscale x 4 x half> %r
}

define <vscale x 2 x float> @test_vmttl_nxv2f32(<vscale x 2 x float> %passthru,
                                                 ptr %base, iXLen %ld,
                                                 iXLen %vl) nounwind {
; CHECK-LABEL: test_vmttl_nxv2f32:
; CHECK:       vmttl.v v8, (a0), a1
; CHECK:       ret
  %r = call <vscale x 2 x float> @llvm.riscv.vmttl.nxv2f32.p0.iXLen(
    <vscale x 2 x float> %passthru, ptr %base, iXLen %ld, iXLen %vl)
  ret <vscale x 2 x float> %r
}

define <vscale x 1 x double> @test_vmttl_nxv1f64(<vscale x 1 x double> %passthru,
                                                  ptr %base, iXLen %ld,
                                                  iXLen %vl) nounwind {
; CHECK-LABEL: test_vmttl_nxv1f64:
; CHECK:       vmttl.v v8, (a0), a1
; CHECK:       ret
  %r = call <vscale x 1 x double> @llvm.riscv.vmttl.nxv1f64.p0.iXLen(
    <vscale x 1 x double> %passthru, ptr %base, iXLen %ld, iXLen %vl)
  ret <vscale x 1 x double> %r
}

define void @test_vmtts_nxv8i8(<vscale x 8 x i8> %v, ptr %base,
                               iXLen %ld, iXLen %vl) nounwind {
; CHECK-LABEL: test_vmtts_nxv8i8:
; CHECK:       vmtts.v v8, (a0), a1
; CHECK:       ret
  call void @llvm.riscv.vmtts.nxv8i8.p0.iXLen(
    <vscale x 8 x i8> %v, ptr %base, iXLen %ld, iXLen %vl)
  ret void
}

define void @test_vmtts_mask_nxv8i8(<vscale x 8 x i8> %v, ptr %base,
                                     iXLen %ld, <vscale x 8 x i1> %mask,
                                     iXLen %vl) nounwind {
; CHECK-LABEL: test_vmtts_mask_nxv8i8:
; CHECK:       vmtts.v v8, (a0), a1, v0.t
; CHECK:       ret
  call void @llvm.riscv.vmtts.mask.nxv8i8.p0.iXLen(
    <vscale x 8 x i8> %v, ptr %base, iXLen %ld,
    <vscale x 8 x i1> %mask, iXLen %vl)
  ret void
}

define void @test_vmtts_nxv4i16(<vscale x 4 x i16> %v, ptr %base,
                                iXLen %ld, iXLen %vl) nounwind {
; CHECK-LABEL: test_vmtts_nxv4i16:
; CHECK:       vmtts.v v8, (a0), a1
; CHECK:       ret
  call void @llvm.riscv.vmtts.nxv4i16.p0.iXLen(
    <vscale x 4 x i16> %v, ptr %base, iXLen %ld, iXLen %vl)
  ret void
}

define void @test_vmtts_nxv1i64(<vscale x 1 x i64> %v, ptr %base,
                                iXLen %ld, iXLen %vl) nounwind {
; CHECK-LABEL: test_vmtts_nxv1i64:
; CHECK:       vmtts.v v8, (a0), a1
; CHECK:       ret
  call void @llvm.riscv.vmtts.nxv1i64.p0.iXLen(
    <vscale x 1 x i64> %v, ptr %base, iXLen %ld, iXLen %vl)
  ret void
}

define void @test_vmtts_nxv4f16(<vscale x 4 x half> %v, ptr %base,
                                iXLen %ld, iXLen %vl) nounwind {
; CHECK-LABEL: test_vmtts_nxv4f16:
; CHECK:       vmtts.v v8, (a0), a1
; CHECK:       ret
  call void @llvm.riscv.vmtts.nxv4f16.p0.iXLen(
    <vscale x 4 x half> %v, ptr %base, iXLen %ld, iXLen %vl)
  ret void
}

define void @test_vmtts_nxv2f32(<vscale x 2 x float> %v, ptr %base,
                                iXLen %ld, iXLen %vl) nounwind {
; CHECK-LABEL: test_vmtts_nxv2f32:
; CHECK:       vmtts.v v8, (a0), a1
; CHECK:       ret
  call void @llvm.riscv.vmtts.nxv2f32.p0.iXLen(
    <vscale x 2 x float> %v, ptr %base, iXLen %ld, iXLen %vl)
  ret void
}

define void @test_vmtts_nxv1f64(<vscale x 1 x double> %v, ptr %base,
                                iXLen %ld, iXLen %vl) nounwind {
; CHECK-LABEL: test_vmtts_nxv1f64:
; CHECK:       vmtts.v v8, (a0), a1
; CHECK:       ret
  call void @llvm.riscv.vmtts.nxv1f64.p0.iXLen(
    <vscale x 1 x double> %v, ptr %base, iXLen %ld, iXLen %vl)
  ret void
}

; LMUL > 1 spot-checks for the transposing tile load/store.

declare <vscale x 4 x i32> @llvm.riscv.vmttl.nxv4i32.p0.iXLen(
  <vscale x 4 x i32>, ptr, iXLen, iXLen)

define <vscale x 4 x i32> @test_vmttl_nxv4i32(<vscale x 4 x i32> %p,
                                              ptr %base, iXLen %ld,
                                              iXLen %vl) nounwind {
; CHECK-LABEL: test_vmttl_nxv4i32:
; CHECK:       vmttl.v v8, (a0), a1
; CHECK:       ret
  %r = call <vscale x 4 x i32> @llvm.riscv.vmttl.nxv4i32.p0.iXLen(
    <vscale x 4 x i32> %p, ptr %base, iXLen %ld, iXLen %vl)
  ret <vscale x 4 x i32> %r
}

declare void @llvm.riscv.vmtts.nxv16i32.p0.iXLen(
  <vscale x 16 x i32>, ptr, iXLen, iXLen)

define void @test_vmtts_nxv16i32(<vscale x 16 x i32> %v, ptr %base,
                                 iXLen %ld, iXLen %vl) nounwind {
; CHECK-LABEL: test_vmtts_nxv16i32:
; CHECK:       vmtts.v v8, (a0), a1
; CHECK:       ret
  call void @llvm.riscv.vmtts.nxv16i32.p0.iXLen(
    <vscale x 16 x i32> %v, ptr %base, iXLen %ld, iXLen %vl)
  ret void
}

declare <vscale x 8 x i32> @llvm.riscv.vmttl.l8.nxv8i32.p0.iXLen(
  <vscale x 8 x i32>, ptr, iXLen, iXLen)

define <vscale x 8 x i32> @test_vmttl_l8_nxv8i32(<vscale x 8 x i32> %p,
                                                 ptr %base, iXLen %ld,
                                                 iXLen %vl) nounwind {
; CHECK-LABEL: test_vmttl_l8_nxv8i32:
; CHECK:       vmttl.v v8, (a0), a1, L8
; CHECK:       ret
  %r = call <vscale x 8 x i32> @llvm.riscv.vmttl.l8.nxv8i32.p0.iXLen(
    <vscale x 8 x i32> %p, ptr %base, iXLen %ld, iXLen %vl)
  ret <vscale x 8 x i32> %r
}
