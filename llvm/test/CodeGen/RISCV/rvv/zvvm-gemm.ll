; RUN: sed 's/iXLen/i64/g' %s | llc -mtriple=riscv64 \
; RUN:   -mattr=+v,+experimental-zvvi32mm,+experimental-zvvmtls,+experimental-zvvmttls \
; RUN:   -verify-machineinstrs | FileCheck %s

; Single-tile matrix multiply-accumulate kernel for the RISC-V Integrated
; Matrix Extension (Zvvm family):
;
;     C[m,n] += sum_k A[m,k] * B[k,n]
;
; Both functions stage one tile per call: load A (row-major) with vmtl.v,
; load B (col-major) by transposing with vmttl.v, load the running C
; accumulator with vmtl.v, run the matrix multiply-accumulate, and store the
; updated C back with vmts.v. A real GEMM nests these in (m, n, k) loops with
; per-iteration vsetvls; this test focuses on the per-tile operation.
;
; The caller passes the avl that vsetvli sees and the leading-dimension
; strides for each operand. vtype.lambda / altfmt_A / altfmt_B must be
; programmed by the caller via vsetvl before the call (the Phase 1/2 codegen
; path does not yet teach RISCVInsertVSETVLI about those fields).

declare iXLen @llvm.riscv.vsetvli.iXLen(iXLen, iXLen immarg, iXLen immarg)

declare <vscale x 2 x i32> @llvm.riscv.vmtl.nxv2i32.p0.iXLen(
  <vscale x 2 x i32>, ptr, iXLen, iXLen)
declare <vscale x 2 x i32> @llvm.riscv.vmttl.nxv2i32.p0.iXLen(
  <vscale x 2 x i32>, ptr, iXLen, iXLen)
declare void @llvm.riscv.vmts.nxv2i32.p0.iXLen(
  <vscale x 2 x i32>, ptr, iXLen, iXLen)

declare <vscale x 2 x i32> @llvm.riscv.vmmacc.nxv2i32.nxv2i32(
  <vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i32>, iXLen)

; -----------------------------------------------------------------------------
; gemm_tile_i32: C[m,n] += A[m,k] * B[k,n]  (no widening; SEW=32 throughout).
;
;   A: i32, row-major, leading dim %lda
;   B: i32, col-major, leading dim %ldb
;   C: i32, row-major, leading dim %ldc
;
; vsetvli e32, m1 establishes SEW=32, LMUL=1 (and uses vtype.lambda /
; altfmt_A / altfmt_B as set up by the caller's prior vsetvl).
define void @gemm_tile_i32(ptr %A, iXLen %lda,
                           ptr %B, iXLen %ldb,
                           ptr %C, iXLen %ldc,
                           iXLen %avl) nounwind {
; CHECK-LABEL: gemm_tile_i32:
; CHECK:       vsetvli {{[a-z0-9]+}}, {{[a-z0-9]+}}, e32, m1, ta, ma
; CHECK:       vmtl.v {{v[0-9]+}}, ({{[a-z0-9]+}}), {{[a-z0-9]+}}
; CHECK:       vmttl.v {{v[0-9]+}}, ({{[a-z0-9]+}}), {{[a-z0-9]+}}
; CHECK:       vmtl.v {{v[0-9]+}}, ({{[a-z0-9]+}}), {{[a-z0-9]+}}
; CHECK:       vmmacc.vv {{v[0-9]+}}, {{v[0-9]+}}, {{v[0-9]+}}
; CHECK:       vmts.v {{v[0-9]+}}, ({{[a-z0-9]+}}), {{[a-z0-9]+}}
; CHECK:       ret
entry:
  ; Program SEW=32 (vsew=2), LMUL=1 (vlmul=0) and obtain VL.
  %vl = call iXLen @llvm.riscv.vsetvli.iXLen(iXLen %avl, iXLen 2, iXLen 0)

  ; Load the A tile (row-major).
  %a = call <vscale x 2 x i32> @llvm.riscv.vmtl.nxv2i32.p0.iXLen(
    <vscale x 2 x i32> poison, ptr %A, iXLen %lda, iXLen %vl)

  ; Load the B tile (col-major) — transposing load brings columns of B into
  ; the row-major layout the multiply-accumulate expects.
  %b = call <vscale x 2 x i32> @llvm.riscv.vmttl.nxv2i32.p0.iXLen(
    <vscale x 2 x i32> poison, ptr %B, iXLen %ldb, iXLen %vl)

  ; Load the running C accumulator (row-major).
  %c_in = call <vscale x 2 x i32> @llvm.riscv.vmtl.nxv2i32.p0.iXLen(
    <vscale x 2 x i32> poison, ptr %C, iXLen %ldc, iXLen %vl)

  ; C += A * B.
  %c_out = call <vscale x 2 x i32> @llvm.riscv.vmmacc.nxv2i32.nxv2i32(
    <vscale x 2 x i32> %c_in,
    <vscale x 2 x i32> %a,
    <vscale x 2 x i32> %b,
    iXLen %vl)

  ; Write the updated C tile back.
  call void @llvm.riscv.vmts.nxv2i32.p0.iXLen(
    <vscale x 2 x i32> %c_out, ptr %C, iXLen %ldc, iXLen %vl)
  ret void
}

; Note: a widening (i16 -> i32 via vwmmacc.vv) variant is omitted because the
; Phase 2 tile-load/store patterns currently only cover nxv2i32. Broadening
; the foreach to additional (SEW, LMUL) types will enable it.
