; RUN: sed 's/iXLen/i32/g' %s | llc -mtriple=riscv32 \
; RUN:   -mattr=+v,+zvfhmin,+experimental-zvvmtls \
; RUN:   -verify-machineinstrs | FileCheck %s
; RUN: sed 's/iXLen/i64/g' %s | llc -mtriple=riscv64 \
; RUN:   -mattr=+v,+zvfhmin,+experimental-zvvmtls \
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

; Element-type coverage for vmtl.v. The instruction encoding is identical for
; every SEW (SEW comes from vtype) — these checks just confirm that the ISel
; pattern matches each LMUL=1 IR vector type and selects the same pseudo. The
; masked variants additionally exercise the per-SEW mask container type.

declare <vscale x 8 x i8>    @llvm.riscv.vmtl.nxv8i8.p0.iXLen(
  <vscale x 8 x i8>,    ptr, iXLen, iXLen)
declare <vscale x 8 x i8>    @llvm.riscv.vmtl.mask.nxv8i8.p0.iXLen(
  <vscale x 8 x i8>,    ptr, iXLen, <vscale x 8 x i1>, iXLen, iXLen immarg)
declare <vscale x 4 x i16>   @llvm.riscv.vmtl.nxv4i16.p0.iXLen(
  <vscale x 4 x i16>,   ptr, iXLen, iXLen)
declare <vscale x 1 x i64>   @llvm.riscv.vmtl.nxv1i64.p0.iXLen(
  <vscale x 1 x i64>,   ptr, iXLen, iXLen)
declare <vscale x 4 x half>  @llvm.riscv.vmtl.nxv4f16.p0.iXLen(
  <vscale x 4 x half>,  ptr, iXLen, iXLen)
declare <vscale x 2 x float> @llvm.riscv.vmtl.nxv2f32.p0.iXLen(
  <vscale x 2 x float>, ptr, iXLen, iXLen)
declare <vscale x 1 x double> @llvm.riscv.vmtl.nxv1f64.p0.iXLen(
  <vscale x 1 x double>, ptr, iXLen, iXLen)

define <vscale x 8 x i8> @test_vmtl_nxv8i8(<vscale x 8 x i8> %passthru,
                                            ptr %base, iXLen %ld,
                                            iXLen %vl) nounwind {
; CHECK-LABEL: test_vmtl_nxv8i8:
; CHECK:       vmtl.v v8, (a0), a1
; CHECK:       ret
  %r = call <vscale x 8 x i8> @llvm.riscv.vmtl.nxv8i8.p0.iXLen(
    <vscale x 8 x i8> %passthru, ptr %base, iXLen %ld, iXLen %vl)
  ret <vscale x 8 x i8> %r
}

define <vscale x 8 x i8> @test_vmtl_mask_nxv8i8(<vscale x 8 x i8> %passthru,
                                                 ptr %base, iXLen %ld,
                                                 <vscale x 8 x i1> %mask,
                                                 iXLen %vl) nounwind {
; CHECK-LABEL: test_vmtl_mask_nxv8i8:
; CHECK:       vmtl.v v8, (a0), a1, v0.t
; CHECK:       ret
  %r = call <vscale x 8 x i8> @llvm.riscv.vmtl.mask.nxv8i8.p0.iXLen(
    <vscale x 8 x i8> %passthru, ptr %base, iXLen %ld,
    <vscale x 8 x i1> %mask, iXLen %vl, iXLen 1)
  ret <vscale x 8 x i8> %r
}

define <vscale x 4 x i16> @test_vmtl_nxv4i16(<vscale x 4 x i16> %passthru,
                                              ptr %base, iXLen %ld,
                                              iXLen %vl) nounwind {
; CHECK-LABEL: test_vmtl_nxv4i16:
; CHECK:       vmtl.v v8, (a0), a1
; CHECK:       ret
  %r = call <vscale x 4 x i16> @llvm.riscv.vmtl.nxv4i16.p0.iXLen(
    <vscale x 4 x i16> %passthru, ptr %base, iXLen %ld, iXLen %vl)
  ret <vscale x 4 x i16> %r
}

define <vscale x 1 x i64> @test_vmtl_nxv1i64(<vscale x 1 x i64> %passthru,
                                              ptr %base, iXLen %ld,
                                              iXLen %vl) nounwind {
; CHECK-LABEL: test_vmtl_nxv1i64:
; CHECK:       vmtl.v v8, (a0), a1
; CHECK:       ret
  %r = call <vscale x 1 x i64> @llvm.riscv.vmtl.nxv1i64.p0.iXLen(
    <vscale x 1 x i64> %passthru, ptr %base, iXLen %ld, iXLen %vl)
  ret <vscale x 1 x i64> %r
}

define <vscale x 4 x half> @test_vmtl_nxv4f16(<vscale x 4 x half> %passthru,
                                               ptr %base, iXLen %ld,
                                               iXLen %vl) nounwind {
; CHECK-LABEL: test_vmtl_nxv4f16:
; CHECK:       vmtl.v v8, (a0), a1
; CHECK:       ret
  %r = call <vscale x 4 x half> @llvm.riscv.vmtl.nxv4f16.p0.iXLen(
    <vscale x 4 x half> %passthru, ptr %base, iXLen %ld, iXLen %vl)
  ret <vscale x 4 x half> %r
}

define <vscale x 2 x float> @test_vmtl_nxv2f32(<vscale x 2 x float> %passthru,
                                                ptr %base, iXLen %ld,
                                                iXLen %vl) nounwind {
; CHECK-LABEL: test_vmtl_nxv2f32:
; CHECK:       vmtl.v v8, (a0), a1
; CHECK:       ret
  %r = call <vscale x 2 x float> @llvm.riscv.vmtl.nxv2f32.p0.iXLen(
    <vscale x 2 x float> %passthru, ptr %base, iXLen %ld, iXLen %vl)
  ret <vscale x 2 x float> %r
}

define <vscale x 1 x double> @test_vmtl_nxv1f64(<vscale x 1 x double> %passthru,
                                                 ptr %base, iXLen %ld,
                                                 iXLen %vl) nounwind {
; CHECK-LABEL: test_vmtl_nxv1f64:
; CHECK:       vmtl.v v8, (a0), a1
; CHECK:       ret
  %r = call <vscale x 1 x double> @llvm.riscv.vmtl.nxv1f64.p0.iXLen(
    <vscale x 1 x double> %passthru, ptr %base, iXLen %ld, iXLen %vl)
  ret <vscale x 1 x double> %r
}

; Element-type coverage for vmts.v. Mirrors the load coverage above; the
; masked i8 case additionally exercises the per-SEW mask container type.

declare void @llvm.riscv.vmts.nxv8i8.p0.iXLen(
  <vscale x 8 x i8>,    ptr, iXLen, iXLen)
declare void @llvm.riscv.vmts.mask.nxv8i8.p0.iXLen(
  <vscale x 8 x i8>,    ptr, iXLen, <vscale x 8 x i1>, iXLen)
declare void @llvm.riscv.vmts.nxv4i16.p0.iXLen(
  <vscale x 4 x i16>,   ptr, iXLen, iXLen)
declare void @llvm.riscv.vmts.nxv1i64.p0.iXLen(
  <vscale x 1 x i64>,   ptr, iXLen, iXLen)
declare void @llvm.riscv.vmts.nxv4f16.p0.iXLen(
  <vscale x 4 x half>,  ptr, iXLen, iXLen)
declare void @llvm.riscv.vmts.nxv2f32.p0.iXLen(
  <vscale x 2 x float>, ptr, iXLen, iXLen)
declare void @llvm.riscv.vmts.nxv1f64.p0.iXLen(
  <vscale x 1 x double>, ptr, iXLen, iXLen)

define void @test_vmts_nxv8i8(<vscale x 8 x i8> %v, ptr %base,
                              iXLen %ld, iXLen %vl) nounwind {
; CHECK-LABEL: test_vmts_nxv8i8:
; CHECK:       vmts.v v8, (a0), a1
; CHECK:       ret
  call void @llvm.riscv.vmts.nxv8i8.p0.iXLen(
    <vscale x 8 x i8> %v, ptr %base, iXLen %ld, iXLen %vl)
  ret void
}

define void @test_vmts_mask_nxv8i8(<vscale x 8 x i8> %v, ptr %base,
                                    iXLen %ld, <vscale x 8 x i1> %mask,
                                    iXLen %vl) nounwind {
; CHECK-LABEL: test_vmts_mask_nxv8i8:
; CHECK:       vmts.v v8, (a0), a1, v0.t
; CHECK:       ret
  call void @llvm.riscv.vmts.mask.nxv8i8.p0.iXLen(
    <vscale x 8 x i8> %v, ptr %base, iXLen %ld,
    <vscale x 8 x i1> %mask, iXLen %vl)
  ret void
}

define void @test_vmts_nxv4i16(<vscale x 4 x i16> %v, ptr %base,
                               iXLen %ld, iXLen %vl) nounwind {
; CHECK-LABEL: test_vmts_nxv4i16:
; CHECK:       vmts.v v8, (a0), a1
; CHECK:       ret
  call void @llvm.riscv.vmts.nxv4i16.p0.iXLen(
    <vscale x 4 x i16> %v, ptr %base, iXLen %ld, iXLen %vl)
  ret void
}

define void @test_vmts_nxv1i64(<vscale x 1 x i64> %v, ptr %base,
                               iXLen %ld, iXLen %vl) nounwind {
; CHECK-LABEL: test_vmts_nxv1i64:
; CHECK:       vmts.v v8, (a0), a1
; CHECK:       ret
  call void @llvm.riscv.vmts.nxv1i64.p0.iXLen(
    <vscale x 1 x i64> %v, ptr %base, iXLen %ld, iXLen %vl)
  ret void
}

define void @test_vmts_nxv4f16(<vscale x 4 x half> %v, ptr %base,
                               iXLen %ld, iXLen %vl) nounwind {
; CHECK-LABEL: test_vmts_nxv4f16:
; CHECK:       vmts.v v8, (a0), a1
; CHECK:       ret
  call void @llvm.riscv.vmts.nxv4f16.p0.iXLen(
    <vscale x 4 x half> %v, ptr %base, iXLen %ld, iXLen %vl)
  ret void
}

define void @test_vmts_nxv2f32(<vscale x 2 x float> %v, ptr %base,
                               iXLen %ld, iXLen %vl) nounwind {
; CHECK-LABEL: test_vmts_nxv2f32:
; CHECK:       vmts.v v8, (a0), a1
; CHECK:       ret
  call void @llvm.riscv.vmts.nxv2f32.p0.iXLen(
    <vscale x 2 x float> %v, ptr %base, iXLen %ld, iXLen %vl)
  ret void
}

define void @test_vmts_nxv1f64(<vscale x 1 x double> %v, ptr %base,
                               iXLen %ld, iXLen %vl) nounwind {
; CHECK-LABEL: test_vmts_nxv1f64:
; CHECK:       vmts.v v8, (a0), a1
; CHECK:       ret
  call void @llvm.riscv.vmts.nxv1f64.p0.iXLen(
    <vscale x 1 x double> %v, ptr %base, iXLen %ld, iXLen %vl)
  ret void
}

; LMUL > 1 spot-checks: tile load/store encoding uses the base VR register
; number; the actual register-group size comes from vtype.LMUL programmed by
; the caller's vsetvl_matrix.

declare <vscale x 4 x i32> @llvm.riscv.vmtl.nxv4i32.p0.iXLen(
  <vscale x 4 x i32>, ptr, iXLen, iXLen)

define <vscale x 4 x i32> @test_vmtl_nxv4i32(<vscale x 4 x i32> %p, ptr %base,
                                             iXLen %ld, iXLen %vl) nounwind {
; CHECK-LABEL: test_vmtl_nxv4i32:
; CHECK:       vmtl.v v8, (a0), a1
; CHECK:       ret
  %r = call <vscale x 4 x i32> @llvm.riscv.vmtl.nxv4i32.p0.iXLen(
    <vscale x 4 x i32> %p, ptr %base, iXLen %ld, iXLen %vl)
  ret <vscale x 4 x i32> %r
}

declare <vscale x 16 x i32> @llvm.riscv.vmtl.nxv16i32.p0.iXLen(
  <vscale x 16 x i32>, ptr, iXLen, iXLen)

define <vscale x 16 x i32> @test_vmtl_nxv16i32(<vscale x 16 x i32> %p, ptr %base,
                                               iXLen %ld, iXLen %vl) nounwind {
; CHECK-LABEL: test_vmtl_nxv16i32:
; CHECK:       vmtl.v v8, (a0), a1
; CHECK:       ret
  %r = call <vscale x 16 x i32> @llvm.riscv.vmtl.nxv16i32.p0.iXLen(
    <vscale x 16 x i32> %p, ptr %base, iXLen %ld, iXLen %vl)
  ret <vscale x 16 x i32> %r
}

declare void @llvm.riscv.vmts.nxv8i32.p0.iXLen(
  <vscale x 8 x i32>, ptr, iXLen, iXLen)

define void @test_vmts_nxv8i32(<vscale x 8 x i32> %v, ptr %base,
                               iXLen %ld, iXLen %vl) nounwind {
; CHECK-LABEL: test_vmts_nxv8i32:
; CHECK:       vmts.v v8, (a0), a1
; CHECK:       ret
  call void @llvm.riscv.vmts.nxv8i32.p0.iXLen(
    <vscale x 8 x i32> %v, ptr %base, iXLen %ld, iXLen %vl)
  ret void
}

declare <vscale x 8 x float> @llvm.riscv.vmtl.l4.nxv8f32.p0.iXLen(
  <vscale x 8 x float>, ptr, iXLen, iXLen)

define <vscale x 8 x float> @test_vmtl_l4_nxv8f32(<vscale x 8 x float> %p,
                                                  ptr %base, iXLen %ld,
                                                  iXLen %vl) nounwind {
; CHECK-LABEL: test_vmtl_l4_nxv8f32:
; CHECK:       vmtl.v v8, (a0), a1, L4
; CHECK:       ret
  %r = call <vscale x 8 x float> @llvm.riscv.vmtl.l4.nxv8f32.p0.iXLen(
    <vscale x 8 x float> %p, ptr %base, iXLen %ld, iXLen %vl)
  ret <vscale x 8 x float> %r
}
