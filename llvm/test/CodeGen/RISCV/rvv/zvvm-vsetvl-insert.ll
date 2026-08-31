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

declare <vscale x 1 x i64> @llvm.riscv.vadd.nxv1i64.nxv1i64(
  <vscale x 1 x i64>, <vscale x 1 x i64>, <vscale x 1 x i64>, iXLen)

declare <vscale x 4 x i32> @llvm.riscv.vadd.nxv4i32.nxv4i32(
  <vscale x 4 x i32>, <vscale x 4 x i32>, <vscale x 4 x i32>, iXLen)

; A vtype write that changes SEW is a may-DEF of vtype.lambda (the IME spec
; lets hardware replace an unsupported lambda with the largest supported
; value for the new (VLEN, SEW)). The tracked lambda must therefore be
; dropped when the pass inserts an SEW-changing config: the emitted
; register-form vsetvl carries lambda bits of 000 (preserve-or-initialize)
; while still re-asserting the retained altfmt_A.
define <vscale x 1 x i64> @matrix_state_sew_change(<vscale x 1 x i64> %d0,
                                                    <vscale x 1 x i64> %d1,
                                                    iXLen %avl) nounwind {
; CHECK-LABEL: matrix_state_sew_change:
; First config: (13 << 58) | 208 = lambda=L4, altfmt_A=1, e32/m1/ta/ma.
; CHECK:       li {{[a-z0-9]+}}, 13
; CHECK:       slli {{[a-z0-9]+}}, {{[a-z0-9]+}}, 58
; CHECK:       addi {{[a-z0-9]+}}, {{[a-z0-9]+}}, 208
; CHECK:       vsetvl {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; The vadd needs e64: SEW changes, so the tracked lambda is dropped to 000
; (preserve-or-initialize) while altfmt_A=1 is retained.
; (1 << 58) | 216 = lambda=000, altfmt_A=1, e64/m1/ta/ma.
; CHECK:       li {{[a-z0-9]+}}, 1
; CHECK:       slli {{[a-z0-9]+}}, {{[a-z0-9]+}}, 58
; CHECK:       addi {{[a-z0-9]+}}, {{[a-z0-9]+}}, 216
; CHECK:       vsetvl {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; CHECK:       vadd.vv
; CHECK-NOT:   vsetvli
entry:
  ; lambda=L4 (3), altfmt_A=1, SEW=e32, LMUL=m1, ta, ma.
  %vl = call iXLen @llvm.riscv.vsetvl.matrix.iXLen(
    iXLen %avl, iXLen 2, iXLen 0, iXLen 1, iXLen 1,
    iXLen 3, iXLen 0, iXLen 1, iXLen 0)
  %sum = call <vscale x 1 x i64> @llvm.riscv.vadd.nxv1i64.nxv1i64(
    <vscale x 1 x i64> poison,
    <vscale x 1 x i64> %d0,
    <vscale x 1 x i64> %d1,
    iXLen %vl)
  ret <vscale x 1 x i64> %sum
}

; An LMUL-only change never re-canonicalizes lambda (support is a function
; of (VLEN, SEW) only), so the tracked lambda survives and is re-encoded
; verbatim in the inserted register-form vsetvl.
define <vscale x 4 x i32> @matrix_state_lmul_only_change(<vscale x 4 x i32> %d0,
                                                          <vscale x 4 x i32> %d1,
                                                          iXLen %avl) nounwind {
; CHECK-LABEL: matrix_state_lmul_only_change:
; First config: (13 << 58) | 208 = lambda=L4, altfmt_A=1, e32/m1/ta/ma.
; CHECK:       li {{[a-z0-9]+}}, 13
; CHECK:       slli {{[a-z0-9]+}}, {{[a-z0-9]+}}, 58
; CHECK:       addi {{[a-z0-9]+}}, {{[a-z0-9]+}}, 208
; CHECK:       vsetvl {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; The vadd needs e32/m2: SEW is unchanged, so lambda=L4 is retained.
; (13 << 58) | 209 = lambda=L4, altfmt_A=1, e32/m2/ta/ma.
; CHECK:       li {{[a-z0-9]+}}, 13
; CHECK:       slli {{[a-z0-9]+}}, {{[a-z0-9]+}}, 58
; CHECK:       addi {{[a-z0-9]+}}, {{[a-z0-9]+}}, 209
; CHECK:       vsetvl {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; CHECK:       vadd.vv
; CHECK-NOT:   vsetvli
entry:
  ; lambda=L4 (3), altfmt_A=1, SEW=e32, LMUL=m1, ta, ma.
  %vl = call iXLen @llvm.riscv.vsetvl.matrix.iXLen(
    iXLen %avl, iXLen 2, iXLen 0, iXLen 1, iXLen 1,
    iXLen 3, iXLen 0, iXLen 1, iXLen 0)
  %sum = call <vscale x 4 x i32> @llvm.riscv.vadd.nxv4i32.nxv4i32(
    <vscale x 4 x i32> poison,
    <vscale x 4 x i32> %d0,
    <vscale x 4 x i32> %d1,
    iXLen %vl)
  ret <vscale x 4 x i32> %sum
}

; When lambda is the only nonzero matrix field and SEW changes, the tracked
; matrix state becomes all-zero and a *plain* vsetvli suffices: hardware
; retains bs / altfmt_A / altfmt_B verbatim and preserves-or-canonicalizes
; lambda by itself, so the cheaper encoding is both correct and preferred.
define <vscale x 1 x i64> @matrix_lambda_only_sew_change(<vscale x 1 x i64> %d0,
                                                          <vscale x 1 x i64> %d1,
                                                          iXLen %avl) nounwind {
; CHECK-LABEL: matrix_lambda_only_sew_change:
; First config: (3 << 60) | 208 = lambda=L4, altfmt_A=0, e32/m1/ta/ma.
; CHECK:       li {{[a-z0-9]+}}, 3
; CHECK:       slli {{[a-z0-9]+}}, {{[a-z0-9]+}}, 60
; CHECK:       addi {{[a-z0-9]+}}, {{[a-z0-9]+}}, 208
; CHECK:       vsetvl {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; All matrix fields are now zero in the tracked state, so the e64 config is
; a plain vsetvli (which cannot and need not touch the matrix fields).
; CHECK:       vsetvli {{[a-z0-9]+}}, {{[a-z0-9]+}}, e64, m1, ta, ma
; CHECK:       vadd.vv
entry:
  ; lambda=L4 (3), altfmt_A=0, SEW=e32, LMUL=m1, ta, ma.
  %vl = call iXLen @llvm.riscv.vsetvl.matrix.iXLen(
    iXLen %avl, iXLen 2, iXLen 0, iXLen 1, iXLen 1,
    iXLen 3, iXLen 0, iXLen 0, iXLen 0)
  %sum = call <vscale x 1 x i64> @llvm.riscv.vadd.nxv1i64.nxv1i64(
    <vscale x 1 x i64> poison,
    <vscale x 1 x i64> %d0,
    <vscale x 1 x i64> %d1,
    iXLen %vl)
  ret <vscale x 1 x i64> %sum
}
