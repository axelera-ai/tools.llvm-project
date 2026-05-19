; RUN: sed 's/iXLen/i64/g' %s | llc -mtriple=riscv64 \
; RUN:   -mattr=+v,+experimental-zvvmm,+experimental-zvvmtls \
; RUN:   -verify-machineinstrs | FileCheck %s

; Mixed Zvvm matrix + standard RVV: verifies that the InsertVSETVLI pass
; preserves the Zvvm matrix vtype fields (lambda / bs / altfmt_A / altfmt_B)
; across an inserted vsetvli for a standard RVV instruction.
;
; The driver is:
;   1. int_riscv_vsetvl_matrix programs vtype with lambda=L4, altfmt_A=1,
;      SEW=32, LMUL=m1, ta, ma.
;   2. A matrix-MAC executes.
;   3. A standard vadd.vv runs (same SEW/LMUL).
;   4. Another matrix-MAC executes (still depends on matrix vtype).
;
; Without the integration, step 3 would emit a plain PseudoVSETVLI that
; clears lambda + altfmt_A, breaking step 4. With it, the pass sees that
; the standard vector op's required SEW/LMUL/policy already match the
; running state programmed by the matrix vsetvl, so it emits NO extra
; vsetvli — matrix state is preserved by default.

declare iXLen @llvm.riscv.vsetvl.matrix.iXLen(
  iXLen, iXLen immarg, iXLen immarg, iXLen immarg, iXLen immarg,
  iXLen immarg, iXLen immarg, iXLen immarg, iXLen immarg)

declare <vscale x 2 x i32> @llvm.riscv.vmmacc.nxv2i32.nxv2i32(
  <vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i32>, iXLen)

declare <vscale x 2 x i32> @llvm.riscv.vadd.nxv2i32.nxv2i32(
  <vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i32>, iXLen)

define <vscale x 2 x i32> @mixed_matrix_standard(<vscale x 2 x i32> %c0,
                                                  <vscale x 2 x i32> %a,
                                                  <vscale x 2 x i32> %b,
                                                  <vscale x 2 x i32> %d,
                                                  iXLen %avl) nounwind {
; CHECK-LABEL: mixed_matrix_standard:
; The user-issued register-form vsetvl programs the full matrix vtype.
; The constant (13 << 58) | 208 = 0x34000000000000D0 encodes lambda=L4
; (bits 60..62 = 0b011), altfmt_A=1 (bit 58), SEW=e32 / LMUL=m1 / ta / ma.
; CHECK:       li {{[a-z0-9]+}}, 13
; CHECK:       slli {{[a-z0-9]+}}, {{[a-z0-9]+}}, 58
; CHECK:       addi {{[a-z0-9]+}}, {{[a-z0-9]+}}, 208
; CHECK:       vsetvl {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; CHECK-NEXT:  vmmacc.vv
; The pass must NOT insert a vsetvli between the matrix MAC and the
; standard vadd: doing so would clobber lambda / altfmt_A. The vadd
; instead piggybacks on the matrix vsetvl's vtype.
; CHECK-NEXT:  vadd.vv
; The second matrix MAC still observes the preserved matrix vtype.
; CHECK-NEXT:  vmmacc.vv
; CHECK-NEXT:  ret
; CHECK-NOT:   vsetvli
entry:
  ; lambda=L4 (3), altfmt_A=1, others default.
  %vl = call iXLen @llvm.riscv.vsetvl.matrix.iXLen(
    iXLen %avl, iXLen 2, iXLen 0, iXLen 1, iXLen 1,
    iXLen 3, iXLen 0, iXLen 1, iXLen 0)
  %c1 = call <vscale x 2 x i32> @llvm.riscv.vmmacc.nxv2i32.nxv2i32(
    <vscale x 2 x i32> %c0,
    <vscale x 2 x i32> %a,
    <vscale x 2 x i32> %b,
    iXLen %vl)
  %sum = call <vscale x 2 x i32> @llvm.riscv.vadd.nxv2i32.nxv2i32(
    <vscale x 2 x i32> poison,
    <vscale x 2 x i32> %c1,
    <vscale x 2 x i32> %d,
    iXLen %vl)
  %c2 = call <vscale x 2 x i32> @llvm.riscv.vmmacc.nxv2i32.nxv2i32(
    <vscale x 2 x i32> %sum,
    <vscale x 2 x i32> %a,
    <vscale x 2 x i32> %b,
    iXLen %vl)
  ret <vscale x 2 x i32> %c2
}
