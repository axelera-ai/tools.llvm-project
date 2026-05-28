//==--- riscv_zvvm_vector.td - RISC-V Zvvm Builtin function list ---------===//
//
//  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Clang builtins for the RISC-V Integrated Matrix
// Extension (Zvvm family): Zvvmm (integer matrix multiply-accumulate),
// Zvvfmm (floating-point matrix multiply-accumulate), Zvvmtls (tile
// load/store), Zvvmttls (transposing tile load/store).
//
// Spec: https://github.com/riscv/integrated-matrix-extension
//
// Current scope:
//   * vsetvl_matrix — full-vtype write (covers lambda / bs / altfmt_A / altfmt_B).
//   * vmmacc_vv     — non-widening integer matrix multiply-accumulate, SEW=32
//                     LMUL=1 only. (Widening variants and broader (SEW, LMUL)
//                     coverage land in follow-up patches.)
//   * vfmmacc_vv    — non-widening FP matrix multiply-accumulate, FP32 LMUL=1
//                     only. (Widening vfwmmacc/vfqmmacc/vf8wmmacc share the
//                     same EMUL_C ≠ LMUL machinery gap as the integer widening
//                     variants and land in the same follow-up.)
//   * vmtl_v / vmts_v   — order-preserving tile load / store (Zvvmtls).
//   * vmttl_v / vmtts_v — transposing tile load / store (Zvvmttls).
//     All four cover LMUL=1, default-lambda unmasked, every element type at
//     i8/i16/i32/i64/f16/f32/f64. (Lambda override _L{N} and masked variants
//     land in follow-ups.)
//
//===----------------------------------------------------------------------===//

include "riscv_vector_common.td"

//===----------------------------------------------------------------------===//
// Zvvm: matrix vsetvl. Programs the full vtype CSR — including the matrix
// fields (lambda, bs, altfmt_A, altfmt_B) — and returns the established VL.
// Lowered to the register-form `vsetvl`; the matrix bits live outside the
// 11-bit immediate window that `vsetvli` can encode.
//
// Generated C signature:
//   size_t __riscv_vsetvl_matrix(size_t avl,
//                                size_t vsew, size_t vlmul,
//                                size_t vta,  size_t vma,
//                                size_t lambda,
//                                size_t bs,
//                                size_t altfmt_a, size_t altfmt_b);
//===----------------------------------------------------------------------===//

let RequiredFeatures = ["experimental-zvvmm"],
    SupportOverloading = false,
    HasVL = false,
    HasMasked = false,
    Log2LMUL = [0],
    Name = "vsetvl_matrix",
    OverloadedName = "vsetvl_matrix",
    IRName = "vsetvl_matrix",
    // The IR intrinsic is overloaded on its (single) integer type; supply the
    // result type so getIntrinsic picks the right variant.
    ManualCodegen = [{IntrinsicTypes = {ResultType};}] in
  def vsetvl_matrix : RVVBuiltin<"", "zzKzKzKzKzKzKzKzKz", "i">;

//===----------------------------------------------------------------------===//
// Zvvm: targeted lambda primitive (spec §"Establishing lambda from C").
//
// Generated C signature:
//   size_t __riscv_vsetlambda(size_t requested_lambda);
//
// `requested_lambda` is an integer constant expression in
// {0, 1, 2, 4, 8, 16, 32, 64} (validated in SemaRISCV).
//
//   * 0     -> read-only query (does not modify vtype).
//   * 1..64 -> writes vtype.lambda = encoding-of(L), preserves all other
//              vtype fields and vl. Hardware may WARL-clamp; the returned
//              value reflects the actually-established lambda.
//
// The return value is the established lambda in the same power-of-two form
// (0 if not configured; otherwise one of 1, 2, 4, 8, 16, 32, 64).
//
// Implementation: this lowers to either int_riscv_query_lambda (L == 0) or
// int_riscv_vsetlambda (L != 0). The IR intrinsics operate on the 3-bit
// vtype.lambda encoding; the conversion between the user-facing power-of-two
// value and the encoding happens entirely in IR-gen.
//===----------------------------------------------------------------------===//

let RequiredFeatures = ["experimental-zvvmm"],
    SupportOverloading = false,
    HasVL = false,
    HasMasked = false,
    Log2LMUL = [0],
    Name = "vsetlambda",
    OverloadedName = "vsetlambda",
    IRName = "vsetlambda",
    ManualCodegen = [{
      {
        // Ops[0] = requested_lambda (compile-time constant; validated by
        // Sema).
        auto *LConst = cast<llvm::ConstantInt>(Ops[0]);
        uint64_t L = LConst->getZExtValue();

        llvm::Value *EncResult;
        if (L == 0) {
          // Read-only query: no vtype write.
          llvm::Function *F = CGM.getIntrinsic(Intrinsic::riscv_query_lambda,
                                               {ResultType});
          EncResult = Builder.CreateCall(F);
        } else {
          // L is a power of two in {1,2,4,8,16,32,64}; the 3-bit encoding is
          // ctz(L) + 1.
          unsigned Encoding = llvm::Log2_64(L) + 1;
          llvm::Function *F = CGM.getIntrinsic(Intrinsic::riscv_vsetlambda,
                                               {ResultType});
          EncResult = Builder.CreateCall(
              F, {llvm::ConstantInt::get(ResultType, Encoding)});
        }

        // Convert established encoding (0..7) back to the power-of-two form
        // the user expects: enc == 0 ? 0 : 1 << (enc - 1).
        llvm::Value *Zero = llvm::ConstantInt::get(ResultType, 0);
        llvm::Value *One = llvm::ConstantInt::get(ResultType, 1);
        llvm::Value *IsZero = Builder.CreateICmpEQ(EncResult, Zero);
        llvm::Value *Shift = Builder.CreateSub(EncResult, One);
        llvm::Value *POT = Builder.CreateShl(One, Shift);
        return Builder.CreateSelect(IsZero, Zero, POT);
      }
    }] in
  def vsetlambda : RVVBuiltin<"", "zKz", "i">;

//===----------------------------------------------------------------------===//
// Zvvmm / Zvvfmm: non-widening matrix multiply-accumulate (vmmacc.vv,
// vfmmacc.vv).
//
// The IME spec models two independent register-group axes for matrix MAC:
//   * vd  uses EMUL_C = VLEN / (SEW(C) * lambda^2)  — independent of LMUL.
//   * vs1, vs2 use vtype.LMUL.
// We expose the full EMUL_C ∈ {1,2,4,8} × LMUL ∈ {1,2,4,8} cross product as
// 16 builtins per MAC, named `__riscv_v(f)mmacc_vv_<sew>m{EMUL_C}_lm{LMUL}`
// per the IME C-naming convention (see [[zvvm-mac-lmul-naming]]). The C-tile
// shape suffix `m{EMUL_C}` is emitted by the standard RVV emitter from
// `Log2LMUL = [cLog2]`; the trailing `_lm{LMUL}` is appended via the explicit
// `Name=` field. Vs1/vs2's prototype uses the existing per-operand
// `(S/SE/L)FixedLog2LMUL:N` transformer to pin them to the LMUL-sized type
// independently of vd's Log2LMUL.
//
// Matrix multiply-accumulate is never masked (encoding requires vm=1) and
// exposes no per-instruction policy operand — vta/vma come from vtype.
// vd is read+written, so the builtin's first vector argument is the
// passthrough accumulator.
//
// Widening MACs (vwmmacc / vqmmacc / v8wmmacc and the FP siblings) are
// deferred: they additionally need varying SEW(A) ≠ SEW(C) on top of
// EMUL_C ≠ LMUL.
//===----------------------------------------------------------------------===//

// Select the right per-operand Log2LMUL transformer based on the
// (cLog2, abLog2) relationship — the underlying transformers are constrained:
//   SFixedLog2LMUL  requires target < builtin
//   SEFixedLog2LMUL requires target ≤ builtin
//   LFixedLog2LMUL  requires target > builtin
class ABLog2LMULTransformer<int cLog2, int abLog2> {
  string Str = !cond(!lt(abLog2, cLog2): "(SFixedLog2LMUL:"#abLog2#")",
                     !eq(abLog2, cLog2): "(SEFixedLog2LMUL:"#abLog2#")",
                     !gt(abLog2, cLog2): "(LFixedLog2LMUL:"#abLog2#")");
}

// Each (cLog2, abLog2) cell becomes one builtin. The C-API name encodes
// both EMUL_C (= 2^cLog2, in the i32m{N} / f32m{N} segment) and LMUL
// (= 2^abLog2, in the trailing _lm{N} segment) explicitly via the `Name`
// field — Suffix is empty so the emitter does NOT append a type-derived
// suffix. This is the only way to control the IME spec's ordering
// (`<type>_lm{LMUL}`) since the default auto-suffix produces
// `<name>_<type>`, not `<name>_<typewithLMUL>`. Each builtin is therefore
// fully qualified at the C-API layer; `OverloadedName` is set equal to
// `Name` to disable generic-name overload resolution, which would
// otherwise erase the LMUL distinction.
foreach cLog2 = [0, 1, 2, 3] in {     // EMUL_C ∈ {1,2,4,8}
  foreach abLog2 = [0, 1, 2, 3] in {  // LMUL   ∈ {1,2,4,8}
    defvar abTfm  = ABLog2LMULTransformer<cLog2, abLog2>.Str;
    defvar emul_c = !shl(1, cLog2);
    defvar lmul   = !shl(1, abLog2);
    defvar intName = "vmmacc_vv_i32m" # emul_c # "_lm" # lmul;
    defvar fpName  = "vfmmacc_vv_f32m" # emul_c # "_lm" # lmul;

    let RequiredFeatures = ["experimental-zvvmm"],
        HasMasked = false,
        UnMaskedPolicyScheme = NonePolicy,
        Log2LMUL = [cLog2],
        // IR intrinsic has two independent vector-type overloads: C
        // (return type, index -1) and A/B (Ops[1]).
        IntrinsicTypes = [-1, 1],
        Name = intName,
        OverloadedName = intName,
        IRName = "vmmacc" in
      def : RVVBuiltin<"", "vv"#abTfm#"v"#abTfm#"v", "i">;

    let RequiredFeatures = ["experimental-zvvfmm"],
        HasMasked = false,
        UnMaskedPolicyScheme = NonePolicy,
        Log2LMUL = [cLog2],
        IntrinsicTypes = [-1, 1],
        Name = fpName,
        OverloadedName = fpName,
        IRName = "vfmmacc" in
      def : RVVBuiltin<"", "vv"#abTfm#"v"#abTfm#"v", "f">;
  }
}

//===----------------------------------------------------------------------===//
// Zvvmtls / Zvvmttls: tile load and store.
//
// Four mnemonics, each at LMUL=1 over TypeRange "csilxfd":
//   vmtl.v  — order-preserving load  (Zvvmtls)
//   vmts.v  — order-preserving store (Zvvmtls)
//   vmttl.v — transposing load       (Zvvmttls)
//   vmtts.v — transposing store      (Zvvmttls)
//
// Generated C signatures (one per element type, e.g. i32m1 shown):
//   vint32m1_t __riscv_vmtl_v_i32m1 (const int32_t *base, size_t ld, size_t vl);
//   vint32m1_t __riscv_vmttl_v_i32m1(const int32_t *base, size_t ld, size_t vl);
//   void       __riscv_vmts_v_i32m1 (int32_t *base, size_t ld,
//                                    vint32m1_t vs3, size_t vl);
//   void       __riscv_vmtts_v_i32m1(int32_t *base, size_t ld,
//                                    vint32m1_t vs3, size_t vl);
//
// The instruction encoding is element-type-agnostic — SEW lives in vtype —
// so every LMUL=1 element type maps to the same pseudo with the matching IR
// overload. Per-instruction lambda override (_L{N} variants) and masked
// variants are not exposed yet; they land in follow-up patches.
//===----------------------------------------------------------------------===//

// Tile load: passthru is poisoned for unmasked builds (standard RVV pattern
// for tail-undisturbed); the IR intrinsic threads it through arg 0.
//
// The default-lambda form (no suffix) uses vtype.lambda. The `_l{N}` variants
// override vtype.lambda for the duration of the instruction via the per-
// instruction Lλ field in the encoding; the matching IR intrinsics are
// int_riscv_vmtl_l{1,2,4,8,16,32,64} and friends.
//
// Log2LMUL = [0,1,2,3] expands the C-surface to LMUL ∈ {1,2,4,8}, i.e.
// `__riscv_vmtl_v_<T>m{1,2,4,8}` (and the same for the `_l{N}` variants).
// Fractional LMUL is spec-reserved for the IME family. Tile L/S has only
// one tile, so there's no EMUL_C ≠ LMUL distinction: A/B/C-agnostic loads
// scale uniformly with vtype.LMUL.
let HasMasked = false,
    UnMaskedPolicyScheme = HasPassthruOperand,
    Log2LMUL = [0, 1, 2, 3] in {
  let RequiredFeatures = ["experimental-zvvmtls"],
      Name = "vmtl_v",
      IRName = "vmtl" in
    def vmtl_v : RVVOutOp0Builtin<"v", "vPCez", "csilxfd">;

  let RequiredFeatures = ["experimental-zvvmttls"],
      Name = "vmttl_v",
      IRName = "vmttl" in
    def vmttl_v : RVVOutOp0Builtin<"v", "vPCez", "csilxfd">;

  // Per-lambda override variants.
  foreach lsuf = ["_l1", "_l2", "_l4", "_l8", "_l16", "_l32", "_l64"] in {
    let RequiredFeatures = ["experimental-zvvmtls"],
        Name = "vmtl_v" # lsuf,
        IRName = "vmtl" # lsuf in
      def : RVVOutOp0Builtin<"v", "vPCez", "csilxfd">;

    let RequiredFeatures = ["experimental-zvvmttls"],
        Name = "vmttl_v" # lsuf,
        IRName = "vmttl" # lsuf in
      def : RVVOutOp0Builtin<"v", "vPCez", "csilxfd">;
  }
}

// Tile store: builtin order is (ptr, ld, vec, vl), IR intrinsic order is
// (vec, ptr, ld, vl). Rotate to match the IR shape. The rotation is operand-
// order-only and is LMUL-agnostic; Log2LMUL = [0,1,2,3] exposes the same
// rotation for every LMUL ∈ {1,2,4,8}.
let HasMasked = false,
    Log2LMUL = [0, 1, 2, 3],
    ManualCodegen = [{
      // Builtin: (ptr, ld, value, vl). Intrinsic: (value, ptr, ld, vl).
      std::rotate(Ops.begin(), Ops.begin() + 2, Ops.begin() + 3);
      IntrinsicTypes = {Ops[0]->getType(), Ops[1]->getType(),
                        Ops[2]->getType()};
    }] in {
  let RequiredFeatures = ["experimental-zvvmtls"],
      Name = "vmts_v",
      IRName = "vmts" in
    def vmts_v : RVVBuiltin<"v", "0Pezv", "csilxfd">;

  let RequiredFeatures = ["experimental-zvvmttls"],
      Name = "vmtts_v",
      IRName = "vmtts" in
    def vmtts_v : RVVBuiltin<"v", "0Pezv", "csilxfd">;

  // Per-lambda override variants for stores.
  foreach lsuf = ["_l1", "_l2", "_l4", "_l8", "_l16", "_l32", "_l64"] in {
    let RequiredFeatures = ["experimental-zvvmtls"],
        Name = "vmts_v" # lsuf,
        IRName = "vmts" # lsuf in
      def : RVVBuiltin<"v", "0Pezv", "csilxfd">;

    let RequiredFeatures = ["experimental-zvvmttls"],
        Name = "vmtts_v" # lsuf,
        IRName = "vmtts" # lsuf in
      def : RVVBuiltin<"v", "0Pezv", "csilxfd">;
  }
}
