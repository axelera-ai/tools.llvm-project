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

// Concrete type of the Zvvm paired-scale register: one m1 register of 16-bit
// (scale_A, scale_B) pairs = vuint16m1_t. (Named locally; the llvm_nxv*_ty
// shorthands are private to other targets' .td files.)
def llvm_zvvm_scale_ty : LLVMType<nxv4i16>;

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
  //   LAMBDA   — 3 bits, supplied as the vtype.lambda write request. Per the
  //              IME spec the encoding is 1=L1, 2=L2, 3=L4, 4=L8, 5=L16,
  //              6=L32, 7=L64, and 0 is a preserve-or-initialize request:
  //              hardware retains the current lambda if it is supported for
  //              the resulting (VLEN, SEW) configuration, otherwise selects
  //              the largest supported nonzero value (only outside the
  //              IME-legal domain does the field remain 000). A nonzero
  //              request is WARL-canonicalized: if unsupported, hardware
  //              selects the largest supported value less than or equal to
  //              the request, or the smallest supported value when none is.
  //              Lambda support is a function of (VLEN, SEW) only — never
  //              LMUL. Software should read vtype.lambda back to observe the
  //              established value.
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
  // (0 = preserve-or-initialize request, 1..7 = L1, L2, L4, L8, L16, L32,
  // L64). The user-facing power-of-two value (0, 1, 2, 4, 8, 16, 32, 64) is
  // converted to/from this encoding by the Clang IR-gen layer.
  //===--------------------------------------------------------------------===//

  // Submit `encoding` as a vtype.lambda write request and return the
  // established (post-WARL) encoding. Preserves SEW, LMUL, vta, vma, bs,
  // altfmt_A, altfmt_B and vl per the spec contract.
  //
  // `encoding` may be a runtime value; values outside 0..7 are undefined
  // behavior at this interface (the pseudo-expansion masks the register form
  // to 3 bits defensively). Encoding 0 is the spec's preserve-or-initialize
  // request: hardware retains a supported current lambda, otherwise selects
  // the largest supported nonzero value for the current (VLEN, SEW)
  // configuration — it is a write, not a read; use `int_riscv_query_lambda`
  // for a pure read. A nonzero unsupported request is canonicalized to the
  // largest supported value <= the request, or the smallest supported value
  // when none is. Lambda support depends on (VLEN, SEW) only — never LMUL.
  //
  // IntrNoMem mirrors the int_riscv_vsetvli convention (vtype is not part of
  // the IR memory model). At the machine level, the lowered pseudo carries
  // Defs = [VL, VTYPE] which preserves ordering with respect to subsequent
  // vector / matrix instructions through standard VL/VTYPE liveness.
  def int_riscv_vsetlambda
      : DefaultAttrsIntrinsic<[llvm_anyint_ty],
                              [LLVMMatchType<0>], // encoding
                              [IntrNoMem]>;

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
  // These are the unscaled (vm=1) forms; the microscaled (vm=0) forms are
  // the separate *_scaled intrinsics below.
  def int_riscv_vfmmacc   : RISCVMatrixMAC; // W = 1
  def int_riscv_vfwmmacc  : RISCVMatrixMAC; // W = 2
  def int_riscv_vfqmmacc  : RISCVMatrixMAC; // W = 4
  def int_riscv_vf8wmmacc : RISCVMatrixMAC; // W = 8

  // Microscaled (vm=0, `v0.scale`) matrix multiply-accumulate:
  //   (C, A, B, scale, bs, vl) -> C
  //
  // - C / A / B as in RISCVMatrixMAC.
  // - scale is the paired E8M0 block-scale register: one full vector
  //   register of 16-bit (scale_A, scale_B) pairs (vuint16m1_t = nxv4i16).
  //   The instruction reads it from v0; ISel constrains the operand to the
  //   VMV0Pair register class.
  // - bs is an ImmArg in {0, 1} recording the microscaling block size the
  //   caller compiled against (0 = 32 elements, 1 = 16 elements). It is a
  //   SEMANTIC RECORD ONLY and is dropped during selection: the
  //   architectural block size comes from vtype.bs, which the caller must
  //   program via vsetvl_matrix — exactly as input signedness (altfmt_A/B)
  //   and the FP formats are vtype state that the sign/format-specific C
  //   entry points record but do not themselves program.
  // - vl matches XLEN.
  //
  // The FP-input scaled forms pair with vfwmmacc/vfqmmacc/vf8wmmacc (the
  // vm=1 encodings of the same opcodes); vfmmacc has no scaled form (vm=0
  // reserved). The integer-input forms (vfwimmacc / vfqimmacc / vf8wimmacc,
  // MXINT inputs, FP accumulator) exist only in microscaled form — their
  // vm=1 encodings are vwmmacc / vqmmacc / v8wmmacc.
  class RISCVMatrixMACScaled
      : DefaultAttrsIntrinsic<[llvm_anyvector_ty],
                              [LLVMMatchType<0>,     // C
                               llvm_anyvector_ty,    // A
                               LLVMMatchType<1>,     // B
                               llvm_zvvm_scale_ty,      // paired E8M0 scales
                               llvm_anyint_ty,       // bs (semantic record)
                               LLVMMatchType<2>],    // vl
                              [IntrNoMem, ImmArg<ArgIndex<4>>,
                               Range<ArgIndex<4>, 0, 2>]>,
        RISCVVIntrinsic {
    let VLOperand = 5;
  }

  def int_riscv_vfwmmacc_scaled  : RISCVMatrixMACScaled; // W = 2, FP inputs
  def int_riscv_vfqmmacc_scaled  : RISCVMatrixMACScaled; // W = 4, FP inputs
  def int_riscv_vf8wmmacc_scaled : RISCVMatrixMACScaled; // W = 8, FP inputs
  def int_riscv_vfwimmacc  : RISCVMatrixMACScaled; // W = 2, MXINT8 inputs
  def int_riscv_vfqimmacc  : RISCVMatrixMACScaled; // W = 4, MXINT inputs
  def int_riscv_vf8wimmacc : RISCVMatrixMACScaled; // W = 8, MXINT inputs

  // EMUL_C = 16 (m16 accumulator) matrix multiply-accumulate: same
  // (C, A, B, vl) -> C shape, but the C accumulator is a 16-register-aligned
  // pair of M8 groups represented as the RISC-V vector tuple target type
  // target("riscv.vector.tuple", <vscale x 64 x i8>, 2). The tuple type is
  // byte-typed and SEW-agnostic — the accumulator SEW comes from vtype; the
  // A/B vector type still carries the input EEW and LMUL, so each
  // (intrinsic, A/B type) pair identifies exactly one spec row. The C-level
  // __riscv_ime_vpair / vunpairlo / vunpairhi pseudo-intrinsics compose and
  // decompose the tuple purely via llvm.riscv.tuple.insert/extract — no
  // dedicated IR intrinsics exist for them.
  class RISCVMatrixMACM16
      : DefaultAttrsIntrinsic<[llvm_any_ty],
                              [LLVMMatchType<0>,
                               llvm_anyvector_ty,
                               LLVMMatchType<1>,
                               llvm_anyint_ty],
                              [IntrNoMem]>,
        RISCVVIntrinsic {
    let VLOperand = 3;
  }

  def int_riscv_vmmacc_m16    : RISCVMatrixMACM16; // W = 1, integer
  def int_riscv_vwmmacc_m16   : RISCVMatrixMACM16; // W = 2, integer
  def int_riscv_vqmmacc_m16   : RISCVMatrixMACM16; // W = 4, integer
  def int_riscv_v8wmmacc_m16  : RISCVMatrixMACM16; // W = 8, integer
  def int_riscv_vfmmacc_m16   : RISCVMatrixMACM16; // W = 1, FP
  def int_riscv_vfwmmacc_m16  : RISCVMatrixMACM16; // W = 2, FP
  def int_riscv_vfqmmacc_m16  : RISCVMatrixMACM16; // W = 4, FP
  def int_riscv_vf8wmmacc_m16 : RISCVMatrixMACM16; // W = 8, FP

  // EMUL_C = 16 microscaled forms: RISCVMatrixMACScaled with the tuple-typed
  // m16 accumulator. Same scale / bs operand semantics as the M1..M8 scaled
  // class (bs is a semantic record; vtype.bs is the architectural state).
  class RISCVMatrixMACScaledM16
      : DefaultAttrsIntrinsic<[llvm_any_ty],
                              [LLVMMatchType<0>,     // C (m16 tuple)
                               llvm_anyvector_ty,    // A
                               LLVMMatchType<1>,     // B
                               llvm_zvvm_scale_ty,      // paired E8M0 scales
                               llvm_anyint_ty,       // bs (semantic record)
                               LLVMMatchType<2>],    // vl
                              [IntrNoMem, ImmArg<ArgIndex<4>>,
                               Range<ArgIndex<4>, 0, 2>]>,
        RISCVVIntrinsic {
    let VLOperand = 5;
  }

  def int_riscv_vfwmmacc_scaled_m16  : RISCVMatrixMACScaledM16; // W = 2, FP
  def int_riscv_vfqmmacc_scaled_m16  : RISCVMatrixMACScaledM16; // W = 4, FP
  def int_riscv_vf8wmmacc_scaled_m16 : RISCVMatrixMACScaledM16; // W = 8, FP
  def int_riscv_vfwimmacc_m16  : RISCVMatrixMACScaledM16; // W = 2, MXINT8
  def int_riscv_vfqimmacc_m16  : RISCVMatrixMACScaledM16; // W = 4, MXINT
  def int_riscv_vf8wimmacc_m16 : RISCVMatrixMACScaledM16; // W = 8, MXINT

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
  // suffixes select an explicit Lλ override. Unlike the vtype.lambda field,
  // the instruction-immediate override is NOT WARL-clamped: a nonzero
  // immediate requests its exact lambda value and raises an
  // illegal-instruction exception if that value is unsupported for the
  // current (VLEN, SEW) configuration. Software that needs to discover the
  // supported set must probe via the vtype-write path (vsetvl /
  // __riscv_vsetlambda readback), not via the instruction immediate.
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
