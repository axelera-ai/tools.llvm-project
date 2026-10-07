; RUN: sed 's/iXLen/i32/g' %s | llc -mtriple=riscv32 \
; RUN:   -mattr=+v,+experimental-zvvmm \
; RUN:   -verify-machineinstrs | FileCheck %s --check-prefixes=CHECK,RV32
; RUN: sed 's/iXLen/i64/g' %s | llc -mtriple=riscv64 \
; RUN:   -mattr=+v,+experimental-zvvmm \
; RUN:   -verify-machineinstrs | FileCheck %s --check-prefixes=CHECK,RV64

; The Zvvm extension to vsetvl programs the full vtype CSR, including the
; matrix-specific fields (lambda, bs, altfmt_A, altfmt_B) that don't fit in
; the 11-bit immediate of vsetvli. Lowering: materialize the vtype constant
; into a GPR via the standard MatInt sequence, then emit a register-form
; `vsetvl rd, avl, vtype_reg`.

declare iXLen @llvm.riscv.vsetvl.matrix.iXLen(
  iXLen, iXLen immarg, iXLen immarg, iXLen immarg, iXLen immarg,
  iXLen immarg, iXLen immarg, iXLen immarg, iXLen immarg)

; -----------------------------------------------------------------------------
; Standard fields only (e32, m1, ta, ma); lambda/bs/altfmt_A/altfmt_B all 0.
; The vtype constant is 0xD0 (= 208: vma=1 at bit 7, vta=1 at bit 6,
; vsew=2 at bits 5:3, vlmul=0 at bits 2:0), small enough that constant
; materialization is a single addi.
define iXLen @vsetvl_e32m1_tama(iXLen %avl) nounwind {
; CHECK-LABEL: vsetvl_e32m1_tama:
; CHECK:       li {{[a-z0-9]+}}, 208
; CHECK:       vsetvl {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; CHECK:       ret
  %vl = call iXLen @llvm.riscv.vsetvl.matrix.iXLen(
    iXLen %avl,
    iXLen 2,    ; VSEW=2 (e32)
    iXLen 0,    ; VLMUL=0 (m1)
    iXLen 1,    ; VTA=1
    iXLen 1,    ; VMA=1
    iXLen 0,    ; LAMBDA=0 (writes invalid/dynamic encoding into vtype.lambda)
    iXLen 0,    ; BS=0
    iXLen 0,    ; ALTFMT_A=0
    iXLen 0)   ; ALTFMT_B=0
  ret iXLen %vl
}

; -----------------------------------------------------------------------------
; Lambda=L4 (encoding 3 = 0b011). On RV32 lambda lives in bits [30:28], so
; bits 28 and 29 are set, giving a vtype constant of 0x300000D0. RV64 places
; lambda 32 bits further up.
define iXLen @vsetvl_e32m1_l4(iXLen %avl) nounwind {
; CHECK-LABEL: vsetvl_e32m1_l4:
; RV32:        lui {{[a-z0-9]+}}, 196608
; RV32:        addi {{[a-z0-9]+}}, {{[a-z0-9]+}}, 208
; RV64:        slli {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[0-9]+}}
; CHECK:       vsetvl {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; CHECK:       ret
  %vl = call iXLen @llvm.riscv.vsetvl.matrix.iXLen(
    iXLen %avl,
    iXLen 2, iXLen 0, iXLen 1, iXLen 1,
    iXLen 3,    ; LAMBDA=3 (L4)
    iXLen 0, iXLen 0, iXLen 0)
  ret iXLen %vl
}

; -----------------------------------------------------------------------------
; altfmt_A = 1 (unsigned A for integer MAC). Bit XLEN-6 in the vtype constant.
define iXLen @vsetvl_altfmt_A(iXLen %avl) nounwind {
; CHECK-LABEL: vsetvl_altfmt_A:
; CHECK:       vsetvl {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; CHECK:       ret
  %vl = call iXLen @llvm.riscv.vsetvl.matrix.iXLen(
    iXLen %avl,
    iXLen 2, iXLen 0, iXLen 1, iXLen 1,
    iXLen 0, iXLen 0,
    iXLen 1,    ; ALTFMT_A=1
    iXLen 0)
  ret iXLen %vl
}

; -----------------------------------------------------------------------------
; All matrix fields set together: lambda=L8, bs=1, altfmt_A=1, altfmt_B=1.
define iXLen @vsetvl_all_matrix(iXLen %avl) nounwind {
; CHECK-LABEL: vsetvl_all_matrix:
; CHECK:       vsetvl {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; CHECK:       ret
  %vl = call iXLen @llvm.riscv.vsetvl.matrix.iXLen(
    iXLen %avl,
    iXLen 2, iXLen 0, iXLen 1, iXLen 1,
    iXLen 4,    ; LAMBDA=4 (L8)
    iXLen 1,    ; BS=1
    iXLen 1,    ; ALTFMT_A=1
    iXLen 1)   ; ALTFMT_B=1
  ret iXLen %vl
}
