; RUN: sed 's/iXLen/i32/g' %s | llc -mtriple=riscv32 \
; RUN:   -mattr=+v,+experimental-zvvmm \
; RUN:   -verify-machineinstrs | FileCheck %s
; RUN: sed 's/iXLen/i64/g' %s | llc -mtriple=riscv64 \
; RUN:   -mattr=+v,+experimental-zvvmm \
; RUN:   -verify-machineinstrs | FileCheck %s

; llvm.riscv.vsetvli.zvvm / vsetvlimax.zvvm: the plain vsetvli as a write of
; the matrix vtype state. Same lowering as llvm.riscv.vsetvli, and a call with
; an unused result survives and stays in front of a following lambda write,
; whose request hardware canonicalizes against the SEW this vsetvli set.

declare iXLen @llvm.riscv.vsetvli.zvvm.iXLen(iXLen, iXLen, iXLen)
declare iXLen @llvm.riscv.vsetvlimax.zvvm.iXLen(iXLen, iXLen)
declare iXLen @llvm.riscv.vsetlambda.iXLen(iXLen)

define iXLen @vsetvli_reg(iXLen %avl) nounwind {
; CHECK-LABEL: vsetvli_reg:
; CHECK:       vsetvli a0, a0, e16, m4, ta, ma
; CHECK-NEXT:  ret
  %vl = call iXLen @llvm.riscv.vsetvli.zvvm.iXLen(iXLen %avl, iXLen 1, iXLen 2)
  ret iXLen %vl
}

define iXLen @vsetvli_imm() nounwind {
; CHECK-LABEL: vsetvli_imm:
; CHECK:       vsetivli a0, 16, e32, m1, ta, ma
; CHECK-NEXT:  ret
  %vl = call iXLen @llvm.riscv.vsetvli.zvvm.iXLen(iXLen 16, iXLen 2, iXLen 0)
  ret iXLen %vl
}

define iXLen @vsetvlimax() nounwind {
; CHECK-LABEL: vsetvlimax:
; CHECK:       vsetvli a0, zero, e8, m8, ta, ma
; CHECK-NEXT:  ret
  %vl = call iXLen @llvm.riscv.vsetvlimax.zvvm.iXLen(iXLen 0, iXLen 3)
  ret iXLen %vl
}

; `__riscv_vsetvl_e16m4(avl); __riscv_vsetlambda(16);` with both results
; unused: the vsetvli is emitted, and before the lambda write.
define void @vsetvli_before_lambda(iXLen %avl) nounwind {
; CHECK-LABEL: vsetvli_before_lambda:
; CHECK:       vsetvli zero, a0, e16, m4, ta, ma
; CHECK-NEXT:  csrr {{[a-z0-9]+}}, vtype
; CHECK:       vsetvl {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; CHECK:       ret
  %vl = call iXLen @llvm.riscv.vsetvli.zvvm.iXLen(iXLen %avl, iXLen 1, iXLen 2)
  %l = call iXLen @llvm.riscv.vsetlambda.iXLen(iXLen 5)
  ret void
}

; Two identical configs, each followed by a lambda write, with a config of
; another SEW between them: neither vsetvli may be merged into the other
; (MachineCSE) or moved; each lambda write sees the SEW written just before
; it.
define void @vsetvli_pairs_not_merged(iXLen %avl) nounwind {
; CHECK-LABEL: vsetvli_pairs_not_merged:
; CHECK:       vsetvli zero, a0, e16, m4, ta, ma
; CHECK-NEXT:  csrr {{[a-z0-9]+}}, vtype
; CHECK:       vsetvl {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; CHECK:       vsetvli zero, a0, e32, m1, ta, ma
; CHECK-NEXT:  csrr {{[a-z0-9]+}}, vtype
; CHECK:       vsetvl {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; CHECK:       vsetvli zero, a0, e16, m4, ta, ma
; CHECK-NEXT:  csrr {{[a-z0-9]+}}, vtype
; CHECK:       vsetvl {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; CHECK:       ret
  %v0 = call iXLen @llvm.riscv.vsetvli.zvvm.iXLen(iXLen %avl, iXLen 1, iXLen 2)
  %l0 = call iXLen @llvm.riscv.vsetlambda.iXLen(iXLen 5)
  %v1 = call iXLen @llvm.riscv.vsetvli.zvvm.iXLen(iXLen %avl, iXLen 2, iXLen 0)
  %l1 = call iXLen @llvm.riscv.vsetlambda.iXLen(iXLen 4)
  %v2 = call iXLen @llvm.riscv.vsetvli.zvvm.iXLen(iXLen %avl, iXLen 1, iXLen 2)
  %l2 = call iXLen @llvm.riscv.vsetlambda.iXLen(iXLen 5)
  ret void
}

; A loop-invariant config followed by a lambda write stays in the loop body:
; the lambda write in every iteration must follow it (no LICM hoist).
define void @vsetvli_in_loop(iXLen %n) nounwind {
; CHECK-LABEL: vsetvli_in_loop:
; CHECK-NOT:   vsetvli zero
; CHECK:       # %loop
; CHECK:       vsetvli zero, {{[a-z0-9]+}}, e16, m4, ta, ma
; CHECK:       csrr {{[a-z0-9]+}}, vtype
; CHECK:       vsetvl {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; CHECK:       bnez
; CHECK:       ret
entry:
  br label %loop
loop:
  %i = phi iXLen [ 0, %entry ], [ %i.next, %loop ]
  %vl = call iXLen @llvm.riscv.vsetvli.zvvm.iXLen(iXLen 1024, iXLen 1, iXLen 2)
  %l = call iXLen @llvm.riscv.vsetlambda.iXLen(iXLen 5)
  %i.next = add iXLen %i, 1
  %done = icmp eq iXLen %i.next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}
