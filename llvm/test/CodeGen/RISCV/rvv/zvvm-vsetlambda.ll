; RUN: sed 's/iXLen/i32/g' %s | llc -mtriple=riscv32 \
; RUN:   -mattr=+v,+experimental-zvvmm \
; RUN:   -verify-machineinstrs | FileCheck %s --check-prefixes=CHECK,RV32
; RUN: sed 's/iXLen/i64/g' %s | llc -mtriple=riscv64 \
; RUN:   -mattr=+v,+experimental-zvvmm \
; RUN:   -verify-machineinstrs | FileCheck %s --check-prefixes=CHECK,RV64

; Targeted lambda primitives, lowered via PseudoVSETLAMBDA (read-modify-write
; of vtype.lambda preserving vl and the other vtype fields) and
; PseudoQUERYLAMBDA (pure csrr+shift+mask read).
;
; The lambda field lives at vtype[XLEN-2:XLEN-4]: SHIFT = XLEN-4, which is
; 28 on RV32 and 60 on RV64.

declare iXLen @llvm.riscv.vsetlambda.iXLen(iXLen)
declare iXLen @llvm.riscv.query.lambda.iXLen()

; -----------------------------------------------------------------------------
; Encoding 1 (L1). All seven encodings exercise the same expansion shape; we
; spot-check encoding=1, 3 (L4) and 7 (L64) below.

define iXLen @vsetlambda_l1() nounwind {
; CHECK-LABEL: vsetlambda_l1:
; CHECK:       csrr [[VTYPE:[a-z0-9]+]], vtype
; CHECK-NEXT:  li [[MASK:[a-z0-9]+]], 7
; RV32-NEXT:   slli [[MASK]], [[MASK]], 28
; RV64-NEXT:   slli [[MASK]], [[MASK]], 60
; CHECK-NEXT:  not [[MASK]], [[MASK]]
; CHECK-NEXT:  and [[VTYPE]], [[VTYPE]], [[MASK]]
; CHECK-NEXT:  li [[NEWENC:[a-z0-9]+]], 1
; RV32-NEXT:   slli [[NEWENC]], [[NEWENC]], 28
; RV64-NEXT:   slli [[NEWENC]], [[NEWENC]], 60
; CHECK-NEXT:  or [[VTYPE]], [[VTYPE]], [[NEWENC]]
; CHECK-NEXT:  csrr [[AVL:[a-z0-9]+]], vl
; CHECK-NEXT:  vsetvl [[AVL]], [[AVL]], [[VTYPE]]
; CHECK-NEXT:  csrr a0, vtype
; RV32-NEXT:   srli a0, a0, 28
; RV64-NEXT:   srli a0, a0, 60
; CHECK-NEXT:  andi a0, a0, 7
; CHECK-NEXT:  ret
  %1 = call iXLen @llvm.riscv.vsetlambda.iXLen(iXLen 1)
  ret iXLen %1
}

define iXLen @vsetlambda_l4() nounwind {
; CHECK-LABEL: vsetlambda_l4:
; CHECK:       li {{[a-z0-9]+}}, 3
; CHECK:       vsetvl
  %1 = call iXLen @llvm.riscv.vsetlambda.iXLen(iXLen 3)
  ret iXLen %1
}

define iXLen @vsetlambda_l64() nounwind {
; CHECK-LABEL: vsetlambda_l64:
; CHECK:       li {{[a-z0-9]+}}, 7
; CHECK:       vsetvl
  %1 = call iXLen @llvm.riscv.vsetlambda.iXLen(iXLen 7)
  ret iXLen %1
}

; -----------------------------------------------------------------------------
; Encoding 0: the spec's preserve-or-initialize request. Still a full RMW
; vsetvl commit (hardware performs the preserve-or-initialize selection); the
; new-encoding step materializes 0.

define iXLen @vsetlambda_preserve_or_init() nounwind {
; CHECK-LABEL: vsetlambda_preserve_or_init:
; CHECK:       csrr {{[a-z0-9]+}}, vtype
; CHECK:       li {{[a-z0-9]+}}, 0
; CHECK:       vsetvl
; CHECK:       csrr a0, vtype
; RV32:        srli a0, a0, 28
; RV64:        srli a0, a0, 60
; CHECK:       andi a0, a0, 7
  %1 = call iXLen @llvm.riscv.vsetlambda.iXLen(iXLen 0)
  ret iXLen %1
}

; -----------------------------------------------------------------------------
; Runtime encoding: PseudoVSETLAMBDA_REG. Same RMW shape, but the new
; encoding comes from the argument register, defensively masked to 3 bits
; (andi + slli) instead of materialized as an immediate.

define iXLen @vsetlambda_runtime(iXLen %enc) nounwind {
; CHECK-LABEL: vsetlambda_runtime:
; CHECK:       csrr [[VTYPE:[a-z0-9]+]], vtype
; CHECK-NEXT:  li [[MASK:[a-z0-9]+]], 7
; RV32-NEXT:   slli [[MASK]], [[MASK]], 28
; RV64-NEXT:   slli [[MASK]], [[MASK]], 60
; CHECK-NEXT:  not [[MASK]], [[MASK]]
; CHECK-NEXT:  and [[VTYPE]], [[VTYPE]], [[MASK]]
; CHECK-NEXT:  andi [[NEWENC:[a-z0-9]+]], a0, 7
; RV32-NEXT:   slli [[NEWENC]], [[NEWENC]], 28
; RV64-NEXT:   slli [[NEWENC]], [[NEWENC]], 60
; CHECK-NEXT:  or [[VTYPE]], [[VTYPE]], [[NEWENC]]
; CHECK-NEXT:  csrr [[AVL:[a-z0-9]+]], vl
; CHECK-NEXT:  vsetvl [[AVL]], [[AVL]], [[VTYPE]]
; CHECK-NEXT:  csrr [[RES:[a-z0-9]+]], vtype
; RV32-NEXT:   srli [[RES]], [[RES]], 28
; RV64-NEXT:   srli [[RES]], [[RES]], 60
; CHECK-NEXT:  andi [[RES]], [[RES]], 7
  %1 = call iXLen @llvm.riscv.vsetlambda.iXLen(iXLen %enc)
  ret iXLen %1
}

; -----------------------------------------------------------------------------
; PseudoQUERYLAMBDA: read-only — csrr vtype + shift + mask, no vsetvl.

define iXLen @query_lambda() nounwind {
; CHECK-LABEL: query_lambda:
; CHECK:       csrr a0, vtype
; RV32-NEXT:   srli a0, a0, 28
; RV64-NEXT:   srli a0, a0, 60
; CHECK-NEXT:  andi a0, a0, 7
; CHECK-NEXT:  ret
; CHECK-NOT:   vsetvl
  %1 = call iXLen @llvm.riscv.query.lambda.iXLen()
  ret iXLen %1
}
