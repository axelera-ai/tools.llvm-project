; RUN: llc -mtriple=riscv64 -mattr=+v,+experimental-zvfbfa \
; RUN:   -verify-machineinstrs < %s | FileCheck %s

; vtype.altfmt is legal only at SEW 8 and 16; at SEW 32 and 64 a set altfmt
; is vill. An instruction that does not demand altfmt keeps the incoming
; value within SEW 8 / 16 and drops it when its config moves SEW to 32 / 64.

declare <vscale x 4 x bfloat> @llvm.riscv.vfadd.nxv4bf16.nxv4bf16.i64(
  <vscale x 4 x bfloat>, <vscale x 4 x bfloat>, <vscale x 4 x bfloat>, i64, i64)
declare <vscale x 2 x i32> @llvm.riscv.vadd.nxv2i32.nxv2i32.i64(
  <vscale x 2 x i32>, <vscale x 2 x i32>, <vscale x 2 x i32>, i64)
declare <vscale x 4 x i16> @llvm.riscv.vadd.nxv4i16.nxv4i16.i64(
  <vscale x 4 x i16>, <vscale x 4 x i16>, <vscale x 4 x i16>, i64)
declare <vscale x 8 x i8> @llvm.riscv.vadd.nxv8i8.nxv8i8.i64(
  <vscale x 8 x i8>, <vscale x 8 x i8>, <vscale x 8 x i8>, i64)

define <vscale x 2 x i32> @int_e32_after_bf16(<vscale x 4 x bfloat> %a,
                                              <vscale x 4 x bfloat> %b,
                                              <vscale x 2 x i32> %c,
                                              i64 %vl) nounwind {
; CHECK-LABEL: int_e32_after_bf16:
; CHECK:       vsetvli zero, a0, e16alt, m1, ta, ma
; CHECK-NEXT:  vfadd.vv v8, v8, v9
; CHECK-NEXT:  vsetvli zero, a0, e32, m1, ta, ma
; CHECK-NEXT:  vadd.vv v8, v8, v10
; CHECK-NEXT:  ret
  %x = call <vscale x 4 x bfloat> @llvm.riscv.vfadd.nxv4bf16.nxv4bf16.i64(
    <vscale x 4 x bfloat> poison, <vscale x 4 x bfloat> %a,
    <vscale x 4 x bfloat> %b, i64 7, i64 %vl)
  %xi = bitcast <vscale x 4 x bfloat> %x to <vscale x 2 x i32>
  %r = call <vscale x 2 x i32> @llvm.riscv.vadd.nxv2i32.nxv2i32.i64(
    <vscale x 2 x i32> poison, <vscale x 2 x i32> %xi, <vscale x 2 x i32> %c,
    i64 %vl)
  ret <vscale x 2 x i32> %r
}

; SEW 8 keeps altfmt: the config stays legal and a following OFP8 op needs
; no further write.
define <vscale x 8 x i8> @int_e8_after_bf16(<vscale x 4 x bfloat> %a,
                                            <vscale x 4 x bfloat> %b,
                                            <vscale x 8 x i8> %c,
                                            i64 %vl) nounwind {
; CHECK-LABEL: int_e8_after_bf16:
; CHECK:       vsetvli zero, a0, e16alt, m1, ta, ma
; CHECK-NEXT:  vfadd.vv v8, v8, v9
; CHECK-NEXT:  vsetvli zero, a0, e8alt, m1, ta, ma
; CHECK-NEXT:  vadd.vv v8, v8, v10
; CHECK-NEXT:  ret
  %x = call <vscale x 4 x bfloat> @llvm.riscv.vfadd.nxv4bf16.nxv4bf16.i64(
    <vscale x 4 x bfloat> poison, <vscale x 4 x bfloat> %a,
    <vscale x 4 x bfloat> %b, i64 7, i64 %vl)
  %xi = bitcast <vscale x 4 x bfloat> %x to <vscale x 8 x i8>
  %r = call <vscale x 8 x i8> @llvm.riscv.vadd.nxv8i8.nxv8i8.i64(
    <vscale x 8 x i8> poison, <vscale x 8 x i8> %xi, <vscale x 8 x i8> %c,
    i64 %vl)
  ret <vscale x 8 x i8> %r
}

; Same SEW: the integer op runs under the BF16 config, no extra vsetvli.
define <vscale x 4 x i16> @int_e16_after_bf16(<vscale x 4 x bfloat> %a,
                                              <vscale x 4 x bfloat> %b,
                                              <vscale x 4 x i16> %c,
                                              i64 %vl) nounwind {
; CHECK-LABEL: int_e16_after_bf16:
; CHECK:       vsetvli zero, a0, e16alt, m1, ta, ma
; CHECK-NEXT:  vfadd.vv v8, v8, v9
; CHECK-NEXT:  vadd.vv v8, v8, v10
; CHECK-NEXT:  ret
  %x = call <vscale x 4 x bfloat> @llvm.riscv.vfadd.nxv4bf16.nxv4bf16.i64(
    <vscale x 4 x bfloat> poison, <vscale x 4 x bfloat> %a,
    <vscale x 4 x bfloat> %b, i64 7, i64 %vl)
  %xi = bitcast <vscale x 4 x bfloat> %x to <vscale x 4 x i16>
  %r = call <vscale x 4 x i16> @llvm.riscv.vadd.nxv4i16.nxv4i16.i64(
    <vscale x 4 x i16> poison, <vscale x 4 x i16> %xi, <vscale x 4 x i16> %c,
    i64 %vl)
  ret <vscale x 4 x i16> %r
}
