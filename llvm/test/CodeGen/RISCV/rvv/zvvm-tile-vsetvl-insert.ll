; RUN: sed 's/iXLen/i64/g' %s | llc -mtriple=riscv64 \
; RUN:   -mattr=+v,+experimental-zvfbfa,+experimental-zvvmtls,+experimental-zvvmttls \
; RUN:   -verify-machineinstrs | FileCheck %s

; The tile load / store pseudos carry the SEW and LMUL of their IR vector type
; and their VL, and RISCVInsertVSETVLI programs vtype from them the way it does
; for an RVV load or store. A caller-side vsetvl cannot be relied on: the plain
; vsetvl intrinsic is memory-free, so a call whose result is unused is dead
; code, and the tile op would otherwise run under whatever vtype happens to be
; live (e.g. the e8/mf2 of a preceding narrowing convert).

declare void @llvm.riscv.vmts.l4.nxv2i32.p0.iXLen(
  <vscale x 2 x i32>, ptr, iXLen, iXLen)
declare void @llvm.riscv.vmts.nxv2i32.p0.iXLen(
  <vscale x 2 x i32>, ptr, iXLen, iXLen)
declare void @llvm.riscv.vmts.mask.nxv2i32.p0.iXLen(
  <vscale x 2 x i32>, ptr, iXLen, <vscale x 2 x i1>, iXLen)
declare void @llvm.riscv.vmtts.nxv4i16.p0.iXLen(
  <vscale x 4 x i16>, ptr, iXLen, iXLen)
declare <vscale x 16 x i8> @llvm.riscv.vmtl.nxv16i8.p0.iXLen(
  <vscale x 16 x i8>, ptr, iXLen, iXLen)
declare <vscale x 4 x i16> @llvm.riscv.vmtl.nxv4i16.p0.iXLen(
  <vscale x 4 x i16>, ptr, iXLen, iXLen)
declare <vscale x 2 x i32> @llvm.riscv.vmtl.mask.nxv2i32.p0.iXLen(
  <vscale x 2 x i32>, ptr, iXLen, <vscale x 2 x i1>, iXLen, iXLen immarg)
declare <vscale x 1 x i64> @llvm.riscv.vmttl.nxv1i64.p0.iXLen(
  <vscale x 1 x i64>, ptr, iXLen, iXLen)
declare <vscale x 8 x i8> @llvm.riscv.vadd.nxv8i8.nxv8i8(
  <vscale x 8 x i8>, <vscale x 8 x i8>, <vscale x 8 x i8>, iXLen)
declare <vscale x 4 x bfloat> @llvm.riscv.vfadd.nxv4bf16.nxv4bf16(
  <vscale x 4 x bfloat>, <vscale x 4 x bfloat>, <vscale x 4 x bfloat>, iXLen,
  iXLen)
declare iXLen @llvm.riscv.vsetvl.matrix.iXLen(
  iXLen, iXLen immarg, iXLen immarg, iXLen immarg, iXLen immarg,
  iXLen immarg, iXLen immarg, iXLen immarg, iXLen immarg)

; A store with no config in sight gets e32/m1 and its VL.
define void @store_e32m1(<vscale x 2 x i32> %v, ptr %p, iXLen %ld,
                         iXLen %vl) nounwind {
; CHECK-LABEL: store_e32m1:
; CHECK:       vsetvli zero, a2, e32, m1, ta, ma
; CHECK-NEXT:  vmts.v v8, (a0), a1, L4
; CHECK-NEXT:  ret
  call void @llvm.riscv.vmts.l4.nxv2i32.p0.iXLen(
    <vscale x 2 x i32> %v, ptr %p, iXLen %ld, iXLen %vl)
  ret void
}

; The kernel shape: an FP8 tile produced at e8 and stored as 32-bit lines.
; The store must not inherit the e8 config.
define void @store_after_e8(<vscale x 8 x i8> %a, <vscale x 8 x i8> %b,
                            ptr %p, iXLen %ld, iXLen %vl) nounwind {
; CHECK-LABEL: store_after_e8:
; CHECK:       vsetvli zero, a2, e8, m1, ta, ma
; CHECK-NEXT:  vadd.vv v8, v8, v9
; CHECK-NEXT:  vsetvli zero, a2, e32, m1, ta, ma
; CHECK-NEXT:  vmts.v v8, (a0), a1, L4
; CHECK-NEXT:  ret
  %s = call <vscale x 8 x i8> @llvm.riscv.vadd.nxv8i8.nxv8i8(
    <vscale x 8 x i8> poison, <vscale x 8 x i8> %a, <vscale x 8 x i8> %b,
    iXLen %vl)
  %t = bitcast <vscale x 8 x i8> %s to <vscale x 2 x i32>
  call void @llvm.riscv.vmts.l4.nxv2i32.p0.iXLen(
    <vscale x 2 x i32> %t, ptr %p, iXLen %ld, iXLen %vl)
  ret void
}

; A BF16 tile stored as 32-bit lines: altfmt is only legal at SEW 8 and 16,
; so the store's config must drop it rather than carry it to e32.
define void @store_after_bf16(<vscale x 4 x bfloat> %a,
                              <vscale x 4 x bfloat> %b, ptr %p, iXLen %ld,
                              iXLen %vl) nounwind {
; CHECK-LABEL: store_after_bf16:
; CHECK:       vsetvli zero, a2, e16alt, m1, ta, ma
; CHECK-NEXT:  vfadd.vv v8, v8, v9
; CHECK-NEXT:  vsetvli zero, a2, e32, m1, ta, ma
; CHECK-NEXT:  vmts.v v8, (a0), a1, L4
; CHECK-NEXT:  ret
  %s = call <vscale x 4 x bfloat> @llvm.riscv.vfadd.nxv4bf16.nxv4bf16(
    <vscale x 4 x bfloat> poison, <vscale x 4 x bfloat> %a,
    <vscale x 4 x bfloat> %b, iXLen 7, iXLen %vl)
  %t = bitcast <vscale x 4 x bfloat> %s to <vscale x 2 x i32>
  call void @llvm.riscv.vmts.l4.nxv2i32.p0.iXLen(
    <vscale x 2 x i32> %t, ptr %p, iXLen %ld, iXLen %vl)
  ret void
}

define void @store_masked(<vscale x 2 x i32> %v, ptr %p, iXLen %ld,
                          <vscale x 2 x i1> %m, iXLen %vl) nounwind {
; CHECK-LABEL: store_masked:
; CHECK:       vsetvli zero, a2, e32, m1, ta, ma
; CHECK-NEXT:  vmts.v v8, (a0), a1, v0.t
; CHECK-NEXT:  ret
  call void @llvm.riscv.vmts.mask.nxv2i32.p0.iXLen(
    <vscale x 2 x i32> %v, ptr %p, iXLen %ld, <vscale x 2 x i1> %m, iXLen %vl)
  ret void
}

define void @transposing_store_e16m1(<vscale x 4 x i16> %v, ptr %p, iXLen %ld,
                                     iXLen %vl) nounwind {
; CHECK-LABEL: transposing_store_e16m1:
; CHECK:       vsetvli zero, a2, e16, m1, ta, ma
; CHECK-NEXT:  vmtts.v v8, (a0), a1
; CHECK-NEXT:  ret
  call void @llvm.riscv.vmtts.nxv4i16.p0.iXLen(
    <vscale x 4 x i16> %v, ptr %p, iXLen %ld, iXLen %vl)
  ret void
}

; An undefined passthru leaves the tail agnostic.
define <vscale x 16 x i8> @load_e8m2(ptr %p, iXLen %ld, iXLen %vl) nounwind {
; CHECK-LABEL: load_e8m2:
; CHECK:       vsetvli zero, a2, e8, m2, ta, ma
; CHECK-NEXT:  vmtl.v v8, (a0), a1
; CHECK-NEXT:  ret
  %r = call <vscale x 16 x i8> @llvm.riscv.vmtl.nxv16i8.p0.iXLen(
    <vscale x 16 x i8> poison, ptr %p, iXLen %ld, iXLen %vl)
  ret <vscale x 16 x i8> %r
}

; A live passthru demands an undisturbed tail.
define <vscale x 4 x i16> @load_tu(<vscale x 4 x i16> %passthru, ptr %p,
                                   iXLen %ld, iXLen %vl) nounwind {
; CHECK-LABEL: load_tu:
; CHECK:       vsetvli zero, a2, e16, m1, tu, ma
; CHECK-NEXT:  vmtl.v v8, (a0), a1
; CHECK-NEXT:  ret
  %r = call <vscale x 4 x i16> @llvm.riscv.vmtl.nxv4i16.p0.iXLen(
    <vscale x 4 x i16> %passthru, ptr %p, iXLen %ld, iXLen %vl)
  ret <vscale x 4 x i16> %r
}

; A masked load with a live passthru honors its policy operand (0 = tu, mu).
define <vscale x 2 x i32> @load_masked_tumu(<vscale x 2 x i32> %passthru,
                                            ptr %p, iXLen %ld,
                                            <vscale x 2 x i1> %m,
                                            iXLen %vl) nounwind {
; CHECK-LABEL: load_masked_tumu:
; CHECK:       vsetvli zero, a2, e32, m1, tu, mu
; CHECK-NEXT:  vmtl.v v8, (a0), a1, v0.t
; CHECK-NEXT:  ret
  %r = call <vscale x 2 x i32> @llvm.riscv.vmtl.mask.nxv2i32.p0.iXLen(
    <vscale x 2 x i32> %passthru, ptr %p, iXLen %ld, <vscale x 2 x i1> %m,
    iXLen %vl, iXLen 0)
  ret <vscale x 2 x i32> %r
}

define <vscale x 1 x i64> @transposing_load_e64m1(ptr %p, iXLen %ld,
                                                  iXLen %vl) nounwind {
; CHECK-LABEL: transposing_load_e64m1:
; CHECK:       vsetvli zero, a2, e64, m1, ta, ma
; CHECK-NEXT:  vmttl.v v8, (a0), a1
; CHECK-NEXT:  ret
  %r = call <vscale x 1 x i64> @llvm.riscv.vmttl.nxv1i64.p0.iXLen(
    <vscale x 1 x i64> poison, ptr %p, iXLen %ld, iXLen %vl)
  ret <vscale x 1 x i64> %r
}

; A matching caller config is kept as is: the matrix fields it programmed
; (lambda=L4, altfmt_A=1) must survive, so no vsetvli is inserted.
define void @store_keeps_matrix_state(<vscale x 2 x i32> %v, ptr %p,
                                      iXLen %ld, iXLen %avl) nounwind {
; CHECK-LABEL: store_keeps_matrix_state:
; (13 << 58) | 208 = lambda=L4, altfmt_A=1, e32/m1/ta/ma.
; CHECK:       li {{[a-z0-9]+}}, 13
; CHECK:       slli {{[a-z0-9]+}}, {{[a-z0-9]+}}, 58
; CHECK:       addi {{[a-z0-9]+}}, {{[a-z0-9]+}}, 208
; CHECK:       vsetvl {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; CHECK-NEXT:  vmts.v v8, (a0), a1
; CHECK-NEXT:  ret
  %vl = call iXLen @llvm.riscv.vsetvl.matrix.iXLen(
    iXLen %avl, iXLen 2, iXLen 0, iXLen 1, iXLen 1,
    iXLen 3, iXLen 0, iXLen 1, iXLen 0)
  call void @llvm.riscv.vmts.nxv2i32.p0.iXLen(
    <vscale x 2 x i32> %v, ptr %p, iXLen %ld, iXLen %vl)
  ret void
}

; A tile op whose type disagrees with the caller config re-programs vtype
; through the register form so the retained matrix fields are re-asserted.
; The SEW change drops the tracked lambda to 000 (preserve-or-initialize).
define <vscale x 1 x i64> @load_changes_sew(ptr %p, iXLen %ld,
                                            iXLen %avl) nounwind {
; CHECK-LABEL: load_changes_sew:
; CHECK:       li {{[a-z0-9]+}}, 13
; CHECK:       slli {{[a-z0-9]+}}, {{[a-z0-9]+}}, 58
; CHECK:       addi {{[a-z0-9]+}}, {{[a-z0-9]+}}, 208
; CHECK:       vsetvl {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; (1 << 58) | 216 = lambda=000, altfmt_A=1, e64/m1/ta/ma.
; CHECK:       li {{[a-z0-9]+}}, 1
; CHECK:       slli {{[a-z0-9]+}}, {{[a-z0-9]+}}, 58
; CHECK:       addi {{[a-z0-9]+}}, {{[a-z0-9]+}}, 216
; CHECK:       vsetvl {{[a-z0-9]+}}, {{[a-z0-9]+}}, {{[a-z0-9]+}}
; CHECK-NEXT:  vmttl.v v8, (a0), a1
; CHECK-NEXT:  ret
  %vl = call iXLen @llvm.riscv.vsetvl.matrix.iXLen(
    iXLen %avl, iXLen 2, iXLen 0, iXLen 1, iXLen 1,
    iXLen 3, iXLen 0, iXLen 1, iXLen 0)
  %r = call <vscale x 1 x i64> @llvm.riscv.vmttl.nxv1i64.p0.iXLen(
    <vscale x 1 x i64> poison, ptr %p, iXLen %ld, iXLen %vl)
  ret <vscale x 1 x i64> %r
}
