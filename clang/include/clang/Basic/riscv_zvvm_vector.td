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
// `requested_lambda` denotes a semantic lambda value in
// {0, 1, 2, 4, 8, 16, 32, 64}. It may be a runtime value; a compile-time
// constant outside the domain is diagnosed in SemaRISCV, and an
// out-of-domain runtime value is undefined behavior per the IME spec.
//
//   * 0     -> preserve-or-initialize WRITE: hardware retains the current
//              lambda if it is supported for the current (VLEN, SEW)
//              configuration, otherwise establishes the largest supported
//              nonzero lambda (zero only outside the IME-legal domain).
//              This may change vtype.lambda — it is not a read-only query;
//              use __riscv_ime_lambda() to inspect without writing.
//   * 1..64 -> requests vtype.lambda = encoding-of(L), preserving all other
//              vtype fields and vl. An unsupported request is
//              WARL-canonicalized (largest supported value <= L, else the
//              smallest supported value); the returned value reflects the
//              actually-established lambda.
//
// The return value is the established lambda in the same power-of-two form
// (0 if not configured; otherwise one of 1, 2, 4, 8, 16, 32, 64).
//
// Implementation: lowers to int_riscv_vsetlambda on the 3-bit vtype.lambda
// encoding. The power-of-two <-> encoding conversion is folded at IR-gen
// time for constants and emitted as cttz/select for runtime requests.
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
        // Ops[0] = requested_lambda: 0 (preserve-or-initialize) or a power
        // of two in {1,...,64}. Compile-time constants are validated by
        // Sema; runtime values outside the domain are UB.
        llvm::Value *EncArg;
        if (auto *LConst = dyn_cast<llvm::ConstantInt>(Ops[0])) {
          uint64_t L = LConst->getZExtValue();
          // Encoding: 0 stays 0 (preserve-or-initialize request); a power
          // of two L maps to log2(L) + 1.
          uint64_t Encoding = (L == 0) ? 0 : llvm::Log2_64(L) + 1;
          EncArg = llvm::ConstantInt::get(ResultType, Encoding);
        } else {
          // Runtime request: enc = (L == 0) ? 0 : cttz(L) + 1.
          llvm::Value *L = Ops[0];
          llvm::Value *Zero = llvm::ConstantInt::get(ResultType, 0);
          llvm::Value *One = llvm::ConstantInt::get(ResultType, 1);
          llvm::Function *Cttz =
              CGM.getIntrinsic(llvm::Intrinsic::cttz, {ResultType});
          llvm::Value *Tz =
              Builder.CreateCall(Cttz, {L, Builder.getTrue()});
          llvm::Value *EncNZ = Builder.CreateAdd(Tz, One);
          llvm::Value *IsZeroReq = Builder.CreateICmpEQ(L, Zero);
          EncArg = Builder.CreateSelect(IsZeroReq, Zero, EncNZ);
        }

        llvm::Function *F = CGM.getIntrinsic(Intrinsic::riscv_vsetlambda,
                                             {ResultType});
        llvm::Value *EncResult = Builder.CreateCall(F, {EncArg});

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
  def vsetlambda : RVVBuiltin<"", "zz", "i">;

//===----------------------------------------------------------------------===//
// Zvvm: geometry-query intrinsics (spec §"Geometry-query intrinsics").
//
// Generated C signatures:
//   size_t __riscv_ime_vlen(void);    // architectural VLEN in bits
//   size_t __riscv_ime_lambda(void);  // currently selected lambda (0 if
//                                     // none is selected)
//
// Both are non-instruction intrinsics and take no vl argument.
// __riscv_ime_lambda() is a pure read of vtype.lambda — it never writes
// vtype (contrast __riscv_vsetlambda(0), which is a preserve-or-initialize
// write). __riscv_ime_vlen() reads the vlenb CSR and scales to bits; the
// compiler may fold it to a constant when VLEN is statically known.
//===----------------------------------------------------------------------===//

let RequiredFeatures = ["experimental-zvvmm"],
    SupportOverloading = false,
    HasVL = false,
    HasMasked = false,
    Log2LMUL = [0],
    Name = "ime_lambda",
    OverloadedName = "ime_lambda",
    IRName = "query_lambda",
    ManualCodegen = [{
      {
        llvm::Function *F = CGM.getIntrinsic(Intrinsic::riscv_query_lambda,
                                             {ResultType});
        llvm::Value *EncResult = Builder.CreateCall(F);
        // Convert the encoding (0..7) to the power-of-two form:
        // enc == 0 ? 0 : 1 << (enc - 1).
        llvm::Value *Zero = llvm::ConstantInt::get(ResultType, 0);
        llvm::Value *One = llvm::ConstantInt::get(ResultType, 1);
        llvm::Value *IsZero = Builder.CreateICmpEQ(EncResult, Zero);
        llvm::Value *Shift = Builder.CreateSub(EncResult, One);
        llvm::Value *POT = Builder.CreateShl(One, Shift);
        return Builder.CreateSelect(IsZero, Zero, POT);
      }
    }] in
  def ime_lambda : RVVBuiltin<"", "z", "i">;

let RequiredFeatures = ["experimental-zvvmm"],
    SupportOverloading = false,
    HasVL = false,
    HasMasked = false,
    Log2LMUL = [0],
    Name = "ime_vlen",
    OverloadedName = "ime_vlen",
    IRName = "",
    ManualCodegen = [{
      {
        // Read the vlenb CSR (VLEN in bytes) via read_register — the same
        // mechanism as __riscv_vlenb() — and scale to bits.
        llvm::LLVMContext &Ctx = CGM.getLLVMContext();
        llvm::Metadata *OpsMD[] = {llvm::MDString::get(Ctx, "vlenb")};
        llvm::Value *RegName = llvm::MetadataAsValue::get(
            Ctx, llvm::MDNode::get(Ctx, OpsMD));
        llvm::Function *F =
            CGM.getIntrinsic(llvm::Intrinsic::read_register, {ResultType});
        llvm::Value *VLenB = Builder.CreateCall(F, RegName);
        return Builder.CreateShl(VLenB,
                                 llvm::ConstantInt::get(ResultType, 3));
      }
    }] in
  def ime_vlen : RVVBuiltin<"", "z", "i">;

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
