//===- IntrinsicsRISCVZvvm.td - RISC-V IME intrinsics ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LLVM IR intrinsics for the RISC-V Integrated Matrix
// Extension (Zvvm family). Currently covers Zvvmm (integer matrix multiply-
// accumulate), Zvvfmm (floating-point matrix multiply-accumulate), Zvvmtls
// (order-preserving tile load/store), and Zvvmttls (transposing tile
// load/store).
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "riscv" in {
  //===--------------------------------------------------------------------===//
  // Extended vsetvl: programs the full vtype CSR including the matrix-specific
  // fields (lambda, bs, altfmt_A, altfmt_B). Lowered to a register-form
  // `vsetvl` instruction — the standard `vsetvli` 11-bit immediate cannot
  // encode the matrix fields, which live at the high end of vtype.
  //
  // Operands (all integer, immarg except AVL):
  //   AVL      — XLen, requested vector length.
  //   VSEW     — 3-bit: 0=e8, 1=e16, 2=e32, 3=e64.
  //   VLMUL    — 3-bit: 0=m1, 1=m2, 2=m4, 3=m8, 5=mf8, 6=mf4, 7=mf2.
  //   VTA      — 1 bit, tail-agnostic policy.
  //   VMA      — 1 bit, mask-agnostic policy.
  //   LAMBDA   — 3 bits, written directly into vtype.lambda. Per the IME
  //              spec the encoding is 0=invalid/dynamic (the next matrix
  //              multiply-accumulate will fault), 1=L1, 2=L2, 3=L4, 4=L8,
  //              5=L16, 6=L32, 7=L64. The field is WARL: hardware may clamp
  //              an unsupported request to the largest supported value for
  //              the current (VLEN, SEW, LMUL); software should read
  //              vtype.lambda back to detect clamping. This intrinsic does
  //              NOT preserve any prior vtype.lambda — to keep a previously
  //              programmed lambda, pass the same value here.
  //   BS       — 1 bit, microscaling block size (0 = 32 elem, 1 = 16 elem).
  //   ALTFMT_A — 1 bit, A operand signedness/format selector.
  //   ALTFMT_B — 1 bit, B operand signedness/format selector.
  //===--------------------------------------------------------------------===//
  def int_riscv_vsetvl_matrix
      : DefaultAttrsIntrinsic<[llvm_anyint_ty],
                              [LLVMMatchType<0>,  // AVL
                               LLVMMatchType<0>,  // VSEW
                               LLVMMatchType<0>,  // VLMUL
                               LLVMMatchType<0>,  // VTA
                               LLVMMatchType<0>,  // VMA
                               LLVMMatchType<0>,  // LAMBDA
                               LLVMMatchType<0>,  // BS
                               LLVMMatchType<0>,  // ALTFMT_A
                               LLVMMatchType<0>], // ALTFMT_B
                              [IntrNoMem,
                               ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>,
                               ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>,
                               ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>,
                               ImmArg<ArgIndex<7>>, ImmArg<ArgIndex<8>>]>;

  //===--------------------------------------------------------------------===//
  // Targeted lambda primitives. These complement int_riscv_vsetvl_matrix for
  // the common case where software wants to establish (or read) only the
  // matrix `lambda` field of `vtype` without re-specifying the other vtype
  // fields. The C-level wrapper is `__riscv_vsetlambda` (see the IME spec
  // §"Establishing lambda from C").
  //
  // The intrinsics operate on the 3-bit `vtype.lambda` encoding directly
  // (0 = not-configured / dynamic, 1..7 = L1, L2, L4, L8, L16, L32, L64).
  // The user-facing power-of-two value (1, 2, 4, 8, 16, 32, 64) is converted
  // to/from this encoding by the Clang IR-gen layer; keeping the IR shape in
  // encoding form makes the pseudo-expansion branchless.
  //===--------------------------------------------------------------------===//

  // Write `vtype.lambda` to `encoding` (an ImmArg in 1..7) and return the
  // post-WARL-clamp established encoding. Preserves SEW, LMUL, vta, vma, bs,
  // altfmt_A, altfmt_B and vl per the spec contract.
  //
  // The encoding == 0 case (read-only query) is handled by the separate
  // `int_riscv_query_lambda` intrinsic.
  //
  // IntrNoMem mirrors the int_riscv_vsetvli convention (vtype is not part of
  // the IR memory model). At the machine level, the lowered pseudo carries
  // Defs = [VL, VTYPE] which preserves ordering with respect to subsequent
  // vector / matrix instructions through standard VL/VTYPE liveness.
  def int_riscv_vsetlambda
      : DefaultAttrsIntrinsic<[llvm_anyint_ty],
                              [LLVMMatchType<0>], // encoding
                              [IntrNoMem,
                               ImmArg<ArgIndex<0>>,
                               Range<ArgIndex<0>, 1, 8>]>;

  // Read the current `vtype.lambda` encoding (0..7) without writing vtype.
  // Pure read of `vtype`; the spec contract is that this intrinsic does not
  // perturb any vtype state.
  def int_riscv_query_lambda
      : DefaultAttrsIntrinsic<[llvm_anyint_ty], [], [IntrNoMem]>;

  // Shape shared by all integer matrix multiply-accumulate intrinsics:
  //   (C, A, B, vl) -> C
  //
  // - C is the accumulator tile (read-modify-written). Its element type sets
  //   SEW and its vector type determines EMUL_C = VLEN / (SEW * lambda^2).
  // - A and B are the input tiles; their vector type carries EEW (= SEW / W,
  //   where W is the widening factor of the specific intrinsic) and LMUL,
  //   independently of C. They are constrained to have the same IR type.
  // - vl matches XLEN.
  //
  // Signedness of A and B (altfmt_A / altfmt_B) and the lambda value are
  // selected by `vtype` and must be configured via `vsetvl` before the call.
  //
  // Memory effect is IntrNoMem, matching every other RVV intrinsic in this
  // file. Ordering with respect to vsetvl is preserved by the SSA dataflow on
  // the vl operand: a matrix-multiply-accumulate consumes the vl produced by
  // the immediately preceding vsetvl, so LLVM cannot hoist or sink it past
  // that vsetvl.
  class RISCVMatrixMAC
      : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                              [LLVMMatchType<0>,
                               llvm_anyvector_ty,
                               LLVMMatchType<1>,
                               llvm_anyint_ty],
                              [IntrNoMem]>,
        RISCVVIntrinsic {
    let VLOperand = 3;
  }

  // Zvvmm: integer matrix multiply-accumulate.
  def int_riscv_vmmacc   : RISCVMatrixMAC; // W = 1
  def int_riscv_vwmmacc  : RISCVMatrixMAC; // W = 2
  def int_riscv_vqmmacc  : RISCVMatrixMAC; // W = 4
  def int_riscv_v8wmmacc : RISCVMatrixMAC; // W = 8

  // Zvvfmm: floating-point matrix multiply-accumulate. Same operand shape as
  // RISCVMatrixMAC: (C, A, B, vl) -> C, with A and B sharing one IR vector
  // type and C an independent IR vector type. The IR types carry only width
  // (SEW) and LMUL; FP format selection (FP16 vs BF16 at 16-bit; E4M3 vs E5M2
  // at 8-bit) lives in vtype.altfmt_A / vtype.altfmt_B and must be programmed
  // via vsetvl_matrix before the call.
  //
  // vm is hardcoded to 1 (unscaled) at the encoding layer in Phase 1. The
  // spec uses vm=0 to select microscaled (MX*) operation with paired E8M0
  // scale factors in v0 — not exposed as IR intrinsics yet.
  def int_riscv_vfmmacc   : RISCVMatrixMAC; // W = 1
  def int_riscv_vfwmmacc  : RISCVMatrixMAC; // W = 2
  def int_riscv_vfqmmacc  : RISCVMatrixMAC; // W = 4
  def int_riscv_vf8wmmacc : RISCVMatrixMAC; // W = 8

  //===--------------------------------------------------------------------===//
  // Tile load/store (Zvvmtls / Zvvmttls).
  //
  // Operand shape mirrors the existing RVV strided load/store classes:
  //   Unmasked load : (passthru, ptr, ld, vl) -> vec
  //   Masked   load : (passthru, ptr, ld, mask, vl, policy) -> vec
  //   Unmasked store: (vec, ptr, ld, vl) -> ()
  //   Masked   store: (vec, ptr, ld, mask, vl) -> ()
  //
  // The lambda override (3-bit immediate at bits[31:29] of the instruction) is
  // not an IR operand: each lambda value gets its own intrinsic per the chosen
  // naming scheme. The base intrinsic name uses vtype.lambda; `_l1` ... `_l64`
  // suffixes select an explicit Lλ override. The override is WARL just like
  // vtype.lambda — hardware may clamp it down for the current (VLEN, SEW,
  // LMUL), so software targeting non-default lambda should plan a runtime
  // fallback path.
  //
  // `ld` (leading dimension in elements) is held in a GPR. `ld == 0` selects
  // a hardware-defined default stride: `lambda * LMUL` for the order-preserving
  // variants (vmtl.v / vmts.v) — a contiguous tile — and `VLEN / (SEW * lambda)`
  // for the transposing variants (vmttl.v / vmtts.v), i.e. the natural maximum
  // width of a B tile.
  //===--------------------------------------------------------------------===//

  class RISCVTileLoadIntr
      : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                              [LLVMMatchType<0>, llvm_anyptr_ty,
                               llvm_anyint_ty, LLVMMatchType<2>],
                              [NoCapture<ArgIndex<1>>, IntrReadMem,
                               IntrArgMemOnly]>,
        RISCVVIntrinsic {
    let VLOperand = 3;
  }

  class RISCVTileLoadMaskedIntr
      : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                              [LLVMMatchType<0>, llvm_anyptr_ty,
                               llvm_anyint_ty,
                               LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                               LLVMMatchType<2>, LLVMMatchType<2>],
                              [NoCapture<ArgIndex<1>>, ImmArg<ArgIndex<5>>,
                               IntrReadMem, IntrArgMemOnly]>,
        RISCVVIntrinsic {
    let VLOperand = 4;
  }

  class RISCVTileStoreIntr
      : DefaultAttrsIntrinsic<[],
                              [llvm_anyvector_ty, llvm_anyptr_ty,
                               llvm_anyint_ty, LLVMMatchType<2>],
                              [NoCapture<ArgIndex<1>>, IntrWriteMem,
                               IntrArgMemOnly]>,
        RISCVVIntrinsic {
    let VLOperand = 3;
  }

  class RISCVTileStoreMaskedIntr
      : DefaultAttrsIntrinsic<[],
                              [llvm_anyvector_ty, llvm_anyptr_ty,
                               llvm_anyint_ty,
                               LLVMScalarOrSameVectorWidth<0, llvm_i1_ty>,
                               LLVMMatchType<2>],
                              [NoCapture<ArgIndex<1>>, IntrWriteMem,
                               IntrArgMemOnly]>,
        RISCVVIntrinsic {
    let VLOperand = 4;
  }

  // Generate the 64 tile-LS intrinsics: 4 mnemonics
  // (vmtl/vmts/vmttl/vmtts) x 8 lambda variants (default + l1..l64) x
  // 2 masked variants.
  //
  // Lambda encoding in the instruction bits[31:29]:
  //   ""    -> 0b000 (use vtype.lambda)
  //   "_l1" -> 0b001, "_l2" -> 0b010, "_l4" -> 0b011, "_l8" -> 0b100,
  //   "_l16"-> 0b101, "_l32"-> 0b110, "_l64"-> 0b111.
  foreach lsuf = ["", "_l1", "_l2", "_l4", "_l8", "_l16", "_l32", "_l64"] in {
    // Zvvmtls: order-preserving tile load/store.
    def int_riscv_vmtl#lsuf      : RISCVTileLoadIntr;
    def int_riscv_vmtl#lsuf#_mask : RISCVTileLoadMaskedIntr;
    def int_riscv_vmts#lsuf      : RISCVTileStoreIntr;
    def int_riscv_vmts#lsuf#_mask : RISCVTileStoreMaskedIntr;

    // Zvvmttls: transposing tile load/store.
    def int_riscv_vmttl#lsuf      : RISCVTileLoadIntr;
    def int_riscv_vmttl#lsuf#_mask : RISCVTileLoadMaskedIntr;
    def int_riscv_vmtts#lsuf      : RISCVTileStoreIntr;
    def int_riscv_vmtts#lsuf#_mask : RISCVTileStoreMaskedIntr;
  }
}
