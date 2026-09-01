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
//   * vmmacc_vv     — non-widening integer matrix multiply-accumulate, SEW=32,
//                     EMUL_C × LMUL ∈ {1,2,4,8}². (Widening variants land in
//                     follow-up patches.)
//   * vfmmacc_vv    — non-widening FP matrix multiply-accumulate, FP32,
//                     EMUL_C × LMUL ∈ {1,2,4,8}². (Widening
//                     vfwmmacc/vfqmmacc/vf8wmmacc share the SEW(A) ≠ SEW(C)
//                     machinery gap with the integer widening variants and
//                     land in the same follow-up.)
//   * vmtl_v / vmts_v   — order-preserving tile load / store (Zvvmtls).
//   * vmttl_v / vmtts_v — transposing tile load / store (Zvvmttls).
//     All four cover LMUL ∈ {1,2,4,8}, every element type at
//     i8/i16/i32/i64/u8/u16/u32/u64/f16/f32/f64, in default-lambda and
//     per-instruction `_L{N}` override forms, unmasked and masked (`_m`,
//     mask-first per the spec's canonical suffix order
//     {type-suffix}[_as_e{S}][_L{N}][_m]). The order-preserving pair
//     additionally has the storage-width-qualified `_as_e{S}` forms
//     (S ∈ {8,16,32,64} ≥ logical width) for packed narrow tiles.
//     Non-ISO element types (bf16, OFP8, OFP4, int4) are deferred with the
//     MX/format phase.
//   * Overloaded short forms (spec §"Overloaded short forms"): every MAC is
//     reachable as `__riscv_{mnemonic}_vv` (argument types select the cell,
//     including sign combos and m16); pair/unpair as `__riscv_ime_vpair` /
//     `_vunpairlo` / `_vunpairhi`; tile stores and masked tile loads as
//     `__riscv_{mnemonic}` (value/mask/pointer types disambiguate). The
//     integer-input MX MACs keep the `_bs{N}` qualifier in their short form
//     (`__riscv_vfwimmacc_vv_bs32`) because the two block sizes share one C
//     signature. Unmasked tile-load short calls are inherently unresolvable
//     (return-type-only variance) and diagnose as ambiguous, mirroring the
//     upstream unmasked-`vle` behavior; `_L{N}` lambda-override variants and
//     `_as_e{S}` loads are reachable only through their long names.
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

// Configuration and query primitives stay gated on the FAMILY flag
// (experimental-zvvmm): the spec ties vtype configuration to the Zvvm family
// as a whole, not to any per-type extension. MAC builtins below are gated on
// their per-type IME extension (which implies the family flag).
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
// The widening MACs (vwmmacc / vqmmacc / v8wmmacc and the FP siblings)
// additionally vary SEW(A/B) = SEW(C)/W and are emitted further below over
// the same (EMUL_C, LMUL) grid, one set per (SEW(C), EEW(A/B)) combination.
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
// `<name>_<type>`, not `<name>_<typewithLMUL>`.
//
// Every MAC cell additionally registers under the spec's overloaded short
// form (`OverloadedName = "{mnemonic}_vv"`, e.g. `__riscv_vmmacc_vv`): the
// (vd, vs1, vs2) argument types fully determine SEW, EMUL_C (including the
// m16 tuple types), the A/B LMUL, and the sign combination, so overload
// resolution selects the unique cell — no `_lm{N}`, input-type, or
// `_su`/`_us` suffix appears in user code (spec §"Overloaded short forms").
//
// Per the IME spec's canonical suffix order, the `_lm{N}` qualifier is
// omitted for LMUL=1: the canonical LMUL=1 long-form name is
// `__riscv_vmmacc_vv_i32m{EMUL_C}` with no qualifier; `_lm2` / `_lm4` /
// `_lm8` name the LMUL>1 variants.
//
// Integer sign-combination axis (spec suffix order
// `{type-suffix}[_su|_us][_lm{N}]`). The architectural signedness of the A
// and B inputs is selected by vtype.altfmt_A / altfmt_B — programmed via
// __riscv_vsetvl_matrix — NOT by the builtin: all four sign combos lower to
// the same signless IR intrinsic (vint and vuint share the LLVM vector
// type). The C-level variants are type-surface only, so user code
// type-checks against the data it actually has:
//   ss: signed A × signed B     → `i{sew}` accumulator token, no suffix
//   uu: unsigned A × unsigned B → `u{sew}` accumulator token
//   su: signed A × unsigned B   → `i{sew}` accumulator token + `_su`
//   us: unsigned A × signed B   → `i{sew}` accumulator token + `_us`
// The accumulator element is architecturally always signed; the `u{sew}`
// accumulator type of the uu variant mirrors the spec's C API, which types
// the whole computation over vuint for unsigned data.
// All four sign combos of a row share that row's per-type IME extension:
// the encoding-map rows differ only in the altfmt_A/altfmt_B bits, which
// belong to the same extension.
// Fields: [accum token prefix, name suffix, vd/ret char, vs1 char, vs2 char]
defvar ZvvmIntMACSignCombos = [
  ["i", "",    "v",  "v",  "v"],
  ["u", "",    "Uv", "Uv", "Uv"],
  ["i", "_su", "v",  "v",  "Uv"],
  ["i", "_us", "v",  "Uv", "v"],
];

// Non-widening SEW rows: [TypeRange char, SEW digits] for the integer side
// and [TypeRange char, C-type token] for the FP side. BF16 and OFP8
// accumulator types are deferred (no Clang vector types / altfmt-driven —
// they land with the MX / alternate-format work).
defvar ZvvmNonWideningIntMACSEWs = [
  ["c", "8",  "experimental-zvvi8mm"],
  ["s", "16", "experimental-zvvi16mm"],
  ["i", "32", "experimental-zvvi32mm"],
  ["l", "64", "experimental-zvvi64mm"],
];
defvar ZvvmNonWideningFPMACRows  = [
  ["x", "f16", "experimental-zvvfp16mm"],
  ["f", "f32", "experimental-zvvfp32mm"],
  ["d", "f64", "experimental-zvvfp64mm"],
];

foreach cLog2 = [0, 1, 2, 3] in {     // EMUL_C ∈ {1,2,4,8}
  foreach abLog2 = [0, 1, 2, 3] in {  // LMUL   ∈ {1,2,4,8}
    defvar abTfm  = ABLog2LMULTransformer<cLog2, abLog2>.Str;
    defvar emul_c = !shl(1, cLog2);
    defvar lmul   = !shl(1, abLog2);
    defvar lmSuffix = !if(!eq(abLog2, 0), "", "_lm" # lmul);

    foreach sew = ZvvmNonWideningIntMACSEWs in {
      foreach sc = ZvvmIntMACSignCombos in {
        defvar intName = "vmmacc_vv_" # sc[0] # sew[1] # "m" # emul_c
                         # sc[1] # lmSuffix;
        let RequiredFeatures = [sew[2]],
            HasMasked = false,
            UnMaskedPolicyScheme = NonePolicy,
            Log2LMUL = [cLog2],
            // IR intrinsic has two independent vector-type overloads: C
            // (return type, index -1) and A/B (Ops[1]).
            IntrinsicTypes = [-1, 1],
            Name = intName,
            OverloadedName = "vmmacc_vv",
            IRName = "vmmacc" in
          def : RVVBuiltin<"", sc[2] # sc[2] # abTfm # sc[3] # abTfm # sc[4],
                           sew[0]>;
      }
    }

    foreach fpRow = ZvvmNonWideningFPMACRows in {
      defvar fpName = "vfmmacc_vv_" # fpRow[1] # "m" # emul_c # lmSuffix;
      let RequiredFeatures = [fpRow[2]],
          HasMasked = false,
          UnMaskedPolicyScheme = NonePolicy,
          Log2LMUL = [cLog2],
          IntrinsicTypes = [-1, 1],
          Name = fpName,
          OverloadedName = "vfmmacc_vv",
          IRName = "vfmmacc" in
        def : RVVBuiltin<"", "vv"#abTfm#"v"#abTfm#"v", fpRow[0]>;
    }
  }
}

//===----------------------------------------------------------------------===//
// EMUL_C = 16 column of the non-widening MAC grid.
//
// The m16 accumulator is the IME-only v<elt>m16_t type — a
// 16-register-aligned pair of M8 groups, represented as the vector tuple
// target("riscv.vector.tuple", <vscale x 64 x i8>, 2). Each def is based at
// Log2LMUL = 3 (M8) so vd/ret is simply `(Tuple:2)v` (a pair of the base M8
// type = m16); vs1/vs2 pin the A/B LMUL with the usual
// `(S/SE FixedLog2LMUL:N)` transformer. The IR intrinsics are the dedicated
// `*_m16` family: at the IR level the tuple accumulator is byte-typed (the C
// SEW lives in vtype), so the (intrinsic, A/B type) pair identifies the row.
//
// The widening MACs' m16 cells are NOT exposed in C yet: their inputs need
// FixedSEW *and* FixedLog2LMUL on one operand, and the prototype language
// allows a single vector-type transformer per operand. Backend ISel for them
// exists; the C surface needs a combined transformer first (deferred).
//===----------------------------------------------------------------------===//

foreach abLog2 = [0, 1, 2, 3] in {    // LMUL ∈ {1,2,4,8}; EMUL_C fixed at 16
  defvar m16abTfm = ABLog2LMULTransformer<3, abLog2>.Str;
  defvar m16lmul = !shl(1, abLog2);
  defvar m16lmSuffix = !if(!eq(abLog2, 0), "", "_lm" # m16lmul);

  foreach sew = ZvvmNonWideningIntMACSEWs in {
    foreach sc = ZvvmIntMACSignCombos in {
      defvar intName = "vmmacc_vv_" # sc[0] # sew[1] # "m16"
                       # sc[1] # m16lmSuffix;
      let RequiredFeatures = [sew[2]],
          HasMasked = false,
          UnMaskedPolicyScheme = NonePolicy,
          Log2LMUL = [3],
          IntrinsicTypes = [-1, 1],
          Name = intName,
          OverloadedName = "vmmacc_vv",
          IRName = "vmmacc_m16" in
        def : RVVBuiltin<"", "(Tuple:2)" # sc[2] # "(Tuple:2)" # sc[2]
                             # m16abTfm # sc[3] # m16abTfm # sc[4], sew[0]>;
    }
  }

  foreach fpRow = ZvvmNonWideningFPMACRows in {
    defvar fpName = "vfmmacc_vv_" # fpRow[1] # "m16" # m16lmSuffix;
    let RequiredFeatures = [fpRow[2]],
        HasMasked = false,
        UnMaskedPolicyScheme = NonePolicy,
        Log2LMUL = [3],
        IntrinsicTypes = [-1, 1],
        Name = fpName,
        OverloadedName = "vfmmacc_vv",
        IRName = "vfmmacc_m16" in
      def : RVVBuiltin<"", "(Tuple:2)v(Tuple:2)v" # m16abTfm # "v"
                           # m16abTfm # "v", fpRow[0]>;
  }
}

//===----------------------------------------------------------------------===//
// IME m16 pair/unpair pseudo-intrinsics (spec §"Pairing and unpairing m16
// accumulator values").
//
//   v<elt>m16_t __riscv_ime_vpair_<elt>m16   (v<elt>m8_t lo, v<elt>m8_t hi);
//   v<elt>m8_t  __riscv_ime_vunpairlo_<elt>m16(v<elt>m16_t src);
//   v<elt>m8_t  __riscv_ime_vunpairhi_<elt>m16(v<elt>m16_t src);
//
// Pure value composition/extraction — no new ISA instruction and no vl
// argument. They lower to llvm.riscv.tuple.insert/extract, whose backend
// coalescing gives zero-move lowering when allocation permits (the spec's
// compiler-modeling table). Bitwise semantics: every element bit, including
// NaN payloads, is preserved. The low half (index 0) is the first eight
// architectural registers of the group; the high half (index 1) the last
// eight.
//===----------------------------------------------------------------------===//

// Fields: [TypeRange char, elt token, vector char, gating extension].
// bf16 has an m16 type (the accumulator type set is uniform) but no MAC
// C surface yet; its pair/unpair still exists per spec.
defvar ZvvmM16PairElts = [
  ["c", "i8",   "v",  "experimental-zvvmm"],
  ["s", "i16",  "v",  "experimental-zvvmm"],
  ["i", "i32",  "v",  "experimental-zvvmm"],
  ["l", "i64",  "v",  "experimental-zvvmm"],
  ["c", "u8",   "Uv", "experimental-zvvmm"],
  ["s", "u16",  "Uv", "experimental-zvvmm"],
  ["i", "u32",  "Uv", "experimental-zvvmm"],
  ["l", "u64",  "Uv", "experimental-zvvmm"],
  ["x", "f16",  "v",  "experimental-zvvfmm"],
  ["f", "f32",  "v",  "experimental-zvvfmm"],
  ["d", "f64",  "v",  "experimental-zvvfmm"],
  ["y", "bf16", "v",  "experimental-zvvfmm"],
];

foreach elt = ZvvmM16PairElts in {
  defvar pairName     = "ime_vpair_"     # elt[1] # "m16";
  defvar unpairloName = "ime_vunpairlo_" # elt[1] # "m16";
  defvar unpairhiName = "ime_vunpairhi_" # elt[1] # "m16";

  let RequiredFeatures = [elt[3]],
      HasMasked = false,
      HasVL = false,
      UnMaskedPolicyScheme = NonePolicy,
      Log2LMUL = [3] in {
    let Name = pairName, OverloadedName = "ime_vpair", IRName = "tuple_insert",
        ManualCodegen = [{
          {
            // (lo, hi) -> m16 tuple: two inserts on poison.
            llvm::Function *TInsert = CGM.getIntrinsic(
                Intrinsic::riscv_tuple_insert,
                {ResultType, Ops[0]->getType()});
            llvm::Value *V = llvm::PoisonValue::get(ResultType);
            V = Builder.CreateCall(TInsert, {V, Ops[0], Builder.getInt32(0)});
            return Builder.CreateCall(TInsert,
                                      {V, Ops[1], Builder.getInt32(1)});
          }
        }] in
      def : RVVBuiltin<"", "(Tuple:2)" # elt[2] # elt[2] # elt[2], elt[0]>;

    let Name = unpairloName, OverloadedName = "ime_vunpairlo",
        IRName = "tuple_extract",
        ManualCodegen = [{
          {
            llvm::Function *TExtract = CGM.getIntrinsic(
                Intrinsic::riscv_tuple_extract,
                {ResultType, Ops[0]->getType()});
            return Builder.CreateCall(TExtract, {Ops[0], Builder.getInt32(0)});
          }
        }] in
      def : RVVBuiltin<"", elt[2] # "(Tuple:2)" # elt[2], elt[0]>;

    let Name = unpairhiName, OverloadedName = "ime_vunpairhi",
        IRName = "tuple_extract",
        ManualCodegen = [{
          {
            llvm::Function *TExtract = CGM.getIntrinsic(
                Intrinsic::riscv_tuple_extract,
                {ResultType, Ops[0]->getType()});
            return Builder.CreateCall(TExtract, {Ops[0], Builder.getInt32(1)});
          }
        }] in
      def : RVVBuiltin<"", elt[2] # "(Tuple:2)" # elt[2], elt[0]>;
  }
}

//===----------------------------------------------------------------------===//
// Zvvmm / Zvvfmm: widening matrix multiply-accumulate (vwmmacc.vv,
// vqmmacc.vv, v8wmmacc.vv, vfwmmacc.vv, vfqmmacc.vv).
//
// Same two-axis (EMUL_C, LMUL) grid as the non-widening MACs, plus the
// widening axis: vs1/vs2 have EEW = SEW(C) / W. Because the prototype
// language allows only one vector-type transformer per operand, each def is
// based on the C element type at Log2LMUL = abLog2 (the A/B LMUL):
//   * vd / return use `(S/SE/L FixedLog2LMUL:cLog2)v` — C type at EMUL_C
//     (FixedLog2LMUL keeps the element type, swaps LMUL);
//   * vs1 / vs2 use `(FixedSEW:EEW)v` — input element width at the def's
//     own LMUL (FixedSEW keeps LMUL and the scalar kind, so an FP base
//     yields _Float16/float inputs and an integer base yields intN_t).
//
// One name per (SEW(C), EEW) row: the type-suffix encodes the accumulator
// type and EMUL_C; `_lm{N}` (N > 1) encodes the A/B LMUL. Rows with 4-bit
// inputs (Int4 / OFP4) and OFP8 inputs are deferred — Clang has no vector
// types for them yet (they land with the MX / alternate-format work).
//
// Covered rows: [IR name, TypeRange char (C type), C-SEW digits, input EEW]
// (FP rows carry the full C-type token instead of digits since they take no
// sign axis.)
//   vwmmacc  (W=2): i16←i8, i32←i16, i64←i32
//   vqmmacc  (W=4): i32←i8, i64←i16
//   v8wmmacc (W=8): i64←i8
//   vfwmmacc (W=2): f32←f16, f64←f32
//   vfqmmacc (W=4): f64←f16
// (vf8wmmacc's only rows are OFP4/OFP8-input — fully deferred.)
defvar ZvvmIntWideningMACRows = [
  ["vwmmacc",  "s", "16", "8",  "experimental-zvvi8i16mm"],
  ["vwmmacc",  "i", "32", "16", "experimental-zvvi16i32mm"],
  ["vwmmacc",  "l", "64", "32", "experimental-zvvi32i64mm"],
  ["vqmmacc",  "i", "32", "8",  "experimental-zvvi8i32mm"],
  ["vqmmacc",  "l", "64", "16", "experimental-zvvi16i64mm"],
  ["v8wmmacc", "l", "64", "8",  "experimental-zvvi8i64mm"],
];
defvar ZvvmFPWideningMACRows = [
  ["vfwmmacc", "f", "f32", "16", "experimental-zvvfp16fp32mm"],
  ["vfwmmacc", "d", "f64", "32", "experimental-zvvfp32fp64mm"],
  ["vfqmmacc", "d", "f64", "16", "experimental-zvvfp16fp64mm"],
];

foreach cLog2 = [0, 1, 2, 3] in {     // EMUL_C ∈ {1,2,4,8}
  foreach abLog2 = [0, 1, 2, 3] in {  // LMUL   ∈ {1,2,4,8}
    // Builtin base Log2LMUL is abLog2, so the vd transformer pins cLog2
    // relative to it (arguments swapped vs the non-widening grid above).
    defvar cdTfm  = ABLog2LMULTransformer<abLog2, cLog2>.Str;
    defvar emul_c = !shl(1, cLog2);
    defvar lmul   = !shl(1, abLog2);
    defvar lmSuffix = !if(!eq(abLog2, 0), "", "_lm" # lmul);

    // The integer rows carry the same four-way sign-combination axis as the
    // non-widening grid (see ZvvmIntMACSignCombos above): the U modifier
    // composes with both FixedLog2LMUL (vd) and FixedSEW (vs1/vs2), so e.g.
    // `_su` yields (vint C, vint A, vuint B) at the widened/narrowed types.
    foreach row = ZvvmIntWideningMACRows in {
      defvar abTfm = "(FixedSEW:" # row[3] # ")";
      foreach sc = ZvvmIntMACSignCombos in {
        defvar wName = row[0] # "_vv_" # sc[0] # row[2] # "m" # emul_c
                       # sc[1] # lmSuffix;
        let RequiredFeatures = [row[4]],
            HasMasked = false,
            UnMaskedPolicyScheme = NonePolicy,
            Log2LMUL = [abLog2],
            IntrinsicTypes = [-1, 1],
            Name = wName,
            OverloadedName = row[0] # "_vv",
            IRName = row[0] in
          def : RVVBuiltin<"", cdTfm # sc[2] # cdTfm # sc[2]
                               # abTfm # sc[3] # abTfm # sc[4], row[1]>;
      }
    }

    foreach row = ZvvmFPWideningMACRows in {
      defvar wName = row[0] # "_vv_" # row[2] # "m" # emul_c # lmSuffix;
      defvar abTfm = "(FixedSEW:" # row[3] # ")";
      let RequiredFeatures = [row[4]],
          HasMasked = false,
          UnMaskedPolicyScheme = NonePolicy,
          Log2LMUL = [abLog2],
          IntrinsicTypes = [-1, 1],
          Name = wName,
          OverloadedName = row[0] # "_vv",
          IRName = row[0] in
        def : RVVBuiltin<"", cdTfm#"v"#cdTfm#"v"#abTfm#"v"#abTfm#"v", row[1]>;
    }
  }
}

//===----------------------------------------------------------------------===//
// Integer-input microscaled MACs: vfwimmacc.vv / vfqimmacc.vv /
// vf8wimmacc.vv (MXINT8 inputs, FP accumulator; MX-only instructions —
// their vm=1 encodings are vwmmacc/vqmmacc/v8wmmacc).
//
// Naming per the spec's spelled prototypes
// (e.g. `vfloat64m4_t __riscv_vf8wimmacc_vv_f64m4_i8m1_bs32(...)`):
//
//   __riscv_vf{w,q,8w}immacc_vv_{f16,f32,f64}m{c}_i8m{ab}_bs{32,16}
//
// The `_i8m{ab}` input-type token is present because an integer input type
// is not derivable from the FP accumulator type; the token carries the A/B
// LMUL itself, so no separate `_lm{N}` qualifier appears (the canonical
// suffix grammar's `_lm{N}` slot is used only by names without an
// input-type token). The `_i8m1` spelling is kept for LMUL=1 — the
// `_lm1`-omission rule applies to the `_lm{N}` qualifier, not to type
// tokens. `_bs{N}` is last per the canonical microscaled suffix order.
//
// Signature (f16m1/i8m1/bs32 shown):
//   vfloat16m1_t __riscv_vfwimmacc_vv_f16m1_i8m1_bs32(
//       vfloat16m1_t vd, vint8m1_t vs1, vint8m1_t vs2,
//       vuint16m1_t v0, size_t vl);
//
// MXINT inputs are always signed — no unsigned variants exist. The paired
// E8M0 block-scale argument is always vuint16m1_t (one register of 16-bit
// pairs), spelled with the dedicated `(ScalePair:16)Uv` transformer since it
// changes SEW, LMUL, and scalar kind at once. The `_bs{N}` suffix and the
// bs immarg it lowers to are a SEMANTIC RECORD of the block size the code
// was written against: the architectural block size is vtype.bs, which the
// caller must program via __riscv_vsetvl_matrix — exactly as the `_su`/`_us`
// sign variants record what vtype.altfmt_A/altfmt_B must be. BS=32 cells
// are gated on the Zvvxi8* extensions, BS=16 cells on Zvvxni8* (which imply
// their BS=32 siblings).
//
// The FP-input microscaled forms (vfwmmacc/vfqmmacc/vf8wmmacc with
// v0.scale) have IR intrinsics and ISel but no C surface yet: every FP MX
// row has OFP4/OFP8 inputs, which still lack Clang vector types. m16
// accumulator cells are likewise deferred (the m16 tuple vd and the
// FixedSEW inputs cannot yet be spelled in one prototype).
//===----------------------------------------------------------------------===//

// Rows: [IR name, TypeRange char (C type), C-type token, input EEW,
//        BS=32 feature, BS=16 feature].
defvar ZvvmIntMXMACRows = [
  ["vfwimmacc",  "x", "f16", "8", "experimental-zvvxi8fp16mm",
                                  "experimental-zvvxni8fp16mm"],
  ["vfqimmacc",  "f", "f32", "8", "experimental-zvvxi8fp32mm",
                                  "experimental-zvvxni8fp32mm"],
  ["vf8wimmacc", "d", "f64", "8", "experimental-zvvxi8fp64mm",
                                  "experimental-zvvxni8fp64mm"],
];

foreach cLog2 = [0, 1, 2, 3] in {     // EMUL_C ∈ {1,2,4,8}
  foreach abLog2 = [0, 1, 2, 3] in {  // LMUL   ∈ {1,2,4,8}
    defvar cdTfm  = ABLog2LMULTransformer<abLog2, cLog2>.Str;
    defvar emul_c = !shl(1, cLog2);
    defvar lmul   = !shl(1, abLog2);

    foreach row = ZvvmIntMXMACRows in {
      defvar abTfm = "(FixedSEW:" # row[3] # ")I";
      foreach bs = [["_bs32", "0"], ["_bs16", "1"]] in {
        defvar mxName = row[0] # "_vv_" # row[2] # "m" # emul_c
                        # "_i" # row[3] # "m" # lmul # bs[0];
        let RequiredFeatures = [!if(!eq(bs[1], "0"), row[4], row[5])],
            HasMasked = false,
            UnMaskedPolicyScheme = NonePolicy,
            Log2LMUL = [abLog2],
            ManualCodegen = [{
              // Ops = (vd, vs1, vs2, scale, vl). Insert the bs immarg
              // (semantic record; vtype.bs carries the architectural state)
              // before vl.
              Ops.insert(Ops.end() - 1,
                         llvm::ConstantInt::get(Ops.back()->getType(), }]
                       # bs[1] # [{));
              IntrinsicTypes = {ResultType, Ops[1]->getType(),
                                Ops.back()->getType()};
            }],
            Name = mxName,
            OverloadedName = row[0] # "_vv" # bs[0],
            IRName = row[0] in
          def : RVVBuiltin<"", cdTfm # "v" # cdTfm # "v" # abTfm # "v"
                               # abTfm # "v" # "(ScalePair:16)Uv", row[1]>;
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Zvvmtls / Zvvmttls: tile load and store.
//
// Four mnemonics, each over TypeRange "csilxfd" at LMUL ∈ {1,2,4,8}:
//   vmtl.v  — order-preserving load  (Zvvmtls)
//   vmts.v  — order-preserving store (Zvvmtls)
//   vmttl.v — transposing load       (Zvvmttls)
//   vmtts.v — transposing store      (Zvvmttls)
//
// Generated C signatures (i32m1 shown), following the spec's canonical
// long-form suffix order {type-suffix}[_L{N}][_m]:
//   vint32m1_t __riscv_vmtl_v_i32m1     (const int32_t *base, size_t ld,
//                                        size_t vl);
//   vint32m1_t __riscv_vmtl_v_i32m1_L4  (const int32_t *base, size_t ld,
//                                        size_t vl);
//   vint32m1_t __riscv_vmtl_v_i32m1_m   (vbool32_t mask, const int32_t *base,
//                                        size_t ld, size_t vl);
//   vint32m1_t __riscv_vmtl_v_i32m1_L4_m(vbool32_t mask, const int32_t *base,
//                                        size_t ld, size_t vl);
//   void __riscv_vmts_v_i32m1     (int32_t *base, size_t ld, vint32m1_t vs3,
//                                  size_t vl);
//   void __riscv_vmts_v_i32m1_m   (vbool32_t mask, int32_t *base, size_t ld,
//                                  vint32m1_t vs3, size_t vl);
//
// The instruction encoding is element-type-agnostic — SEW lives in vtype —
// so every element type maps to the same pseudo with the matching IR
// overload. Fractional LMUL is spec-reserved for the IME family. Tile L/S
// has only one tile, so there's no EMUL_C ≠ LMUL distinction.
//
// The default-lambda form (no `_L{N}`) uses vtype.lambda. The `_L{N}`
// variants request the exact per-instruction Lλ override encoded in the
// instruction's lambda field; the immediate is checked exactly by hardware
// (NOT WARL-clamped) and raises illegal-instruction when the requested
// value is unsupported for the current (VLEN, SEW). The matching IR
// intrinsics keep their lowercase int_riscv_vmtl_l{1,...,64} spelling.
//
// The `_L{N}` spelling must come AFTER the type suffix (spec order), which
// the emitter's Name+auto-Suffix composition cannot produce, so each
// (type, LMUL, lambda) cell is emitted as its own fully-named builtin with
// Suffix="" — the same technique the MAC builtins above use for `_lm{N}`.
// The masked `_m` qualifier is appended last by the standard masked-name
// machinery, giving `_i32m1_L4_m`.
//
// Masking: tile loads/stores follow the spec's masked form with the mask as
// the leading argument and no maskedoff operand. Tile loads always leave
// inactive and tail destination elements undisturbed architecturally, so
// the policy operand of the masked IR intrinsic only constrains compiler
// freedom; we pass the TAMA policy bits to keep the poison-passthru
// convention of the unsuffixed RVV forms.
//===----------------------------------------------------------------------===//

// (TypeRange char, C type token, load prototype, store prototype) rows used
// to spell explicit names. The unsigned rows reuse the signed TypeRange char
// with the `U` modifier in the prototype (same signless IR intrinsic; the
// suffix token differs).
defvar ZvvmTileTypeCells = [
  ["c", "i8",  "vPCez",   "0Pezv"],
  ["s", "i16", "vPCez",   "0Pezv"],
  ["i", "i32", "vPCez",   "0Pezv"],
  ["l", "i64", "vPCez",   "0Pezv"],
  ["x", "f16", "vPCez",   "0Pezv"],
  ["f", "f32", "vPCez",   "0Pezv"],
  ["d", "f64", "vPCez",   "0Pezv"],
  ["c", "u8",  "UvPCUez", "0PUezUv"],
  ["s", "u16", "UvPCUez", "0PUezUv"],
  ["i", "u32", "UvPCUez", "0PUezUv"],
  ["l", "u64", "UvPCUez", "0PUezUv"],
];
defvar ZvvmTileLambdas = [1, 2, 4, 8, 16, 32, 64];

// Tile load: the IR intrinsic threads a passthru through arg 0. For the
// plain (tail-agnostic) form the passthru is poison; the `_tu` policy
// variant supplies it explicitly; the masked `_m` form (mask, ptr, ld, vl)
// maps to (poison, ptr, ld, mask, vl, policy).
let HasMasked = true,
    HasMaskedOffOperand = false,
    MaskedPolicyScheme = NonePolicy,
    UnMaskedPolicyScheme = HasPassthruOperand,
    ManualCodegen = [{
      {
        if (IsMasked) {
          // Builtin: (mask, ptr, ld, vl).
          // Intrinsic: (passthru, ptr, ld, mask, vl, policy).
          std::rotate(Ops.begin(), Ops.begin() + 1, Ops.end() - 1);
          Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));
          Ops.insert(Ops.begin(), llvm::PoisonValue::get(ResultType));
          IntrinsicTypes = {ResultType, Ops[1]->getType(), Ops[4]->getType()};
        } else {
          // TA builtin: (ptr, ld, vl); _tu builtin: (passthru, ptr, ld, vl).
          // Intrinsic: (passthru, ptr, ld, vl).
          if (Ops.size() == 3)
            Ops.insert(Ops.begin(), llvm::PoisonValue::get(ResultType));
          IntrinsicTypes = {ResultType, Ops[1]->getType(), Ops[3]->getType()};
        }
      }
    }] in {
  // Default-lambda forms: one def per mnemonic, auto type suffix.
  let Log2LMUL = [0, 1, 2, 3] in {
    let RequiredFeatures = ["experimental-zvvmtls"],
        Name = "vmtl_v",
        OverloadedName = "vmtl_v",
        IRName = "vmtl",
        MaskedIRName = "vmtl_mask" in
      def vmtl_v : RVVOutOp0Builtin<"v", "vPCez", "csilxfd">;

    let RequiredFeatures = ["experimental-zvvmttls"],
        Name = "vmttl_v",
        OverloadedName = "vmttl_v",
        IRName = "vmttl",
        MaskedIRName = "vmttl_mask" in
      def vmttl_v : RVVOutOp0Builtin<"v", "vPCez", "csilxfd">;

    // Unsigned integer element types (u8/u16/u32/u64): the `Uv` suffix
    // spells the u-token; the IR intrinsic is the same signless one.
    let RequiredFeatures = ["experimental-zvvmtls"],
        Name = "vmtl_v",
        OverloadedName = "vmtl_v",
        IRName = "vmtl",
        MaskedIRName = "vmtl_mask" in
      def vmtl_v_u : RVVOutOp0Builtin<"Uv", "UvPCUez", "csil">;

    let RequiredFeatures = ["experimental-zvvmttls"],
        Name = "vmttl_v",
        OverloadedName = "vmttl_v",
        IRName = "vmttl",
        MaskedIRName = "vmttl_mask" in
      def vmttl_v_u : RVVOutOp0Builtin<"Uv", "UvPCUez", "csil">;
  }

  // Per-lambda `_L{N}` override variants: one fully-named def per
  // (type, LMUL, lambda) cell so `_L{N}` lands after the type suffix.
  foreach tp = ZvvmTileTypeCells in {
    foreach l2 = [0, 1, 2, 3] in {
      defvar tsuf = tp[1] # "m" # !shl(1, l2);
      foreach lam = ZvvmTileLambdas in {
        let Log2LMUL = [l2] in {
          let RequiredFeatures = ["experimental-zvvmtls"],
              Name = "vmtl_v_" # tsuf # "_L" # lam,
              OverloadedName = "vmtl_v_" # tsuf # "_L" # lam,
              IRName = "vmtl_l" # lam,
              MaskedIRName = "vmtl_l" # lam # "_mask" in
            def : RVVOutOp0Builtin<"", tp[2], tp[0]>;

          let RequiredFeatures = ["experimental-zvvmttls"],
              Name = "vmttl_v_" # tsuf # "_L" # lam,
              OverloadedName = "vmttl_v_" # tsuf # "_L" # lam,
              IRName = "vmttl_l" # lam,
              MaskedIRName = "vmttl_l" # lam # "_mask" in
            def : RVVOutOp0Builtin<"", tp[2], tp[0]>;
        }
      }
    }
  }
}

// Tile store: builtin order is (ptr, ld, value, vl) — masked adds the mask
// in front — while the IR intrinsic order is (value, ptr, ld[, mask], vl).
// Rearrange to match the IR shape. The rearrangement is operand-order-only
// and is LMUL-agnostic.
let HasMasked = true,
    HasMaskedOffOperand = false,
    MaskedPolicyScheme = NonePolicy,
    ManualCodegen = [{
      {
        if (IsMasked) {
          // Builtin: (mask, ptr, ld, value, vl).
          // Intrinsic: (value, ptr, ld, mask, vl).
          std::swap(Ops[0], Ops[3]);
          IntrinsicTypes = {Ops[0]->getType(), Ops[1]->getType(),
                            Ops[4]->getType()};
        } else {
          // Builtin: (ptr, ld, value, vl). Intrinsic: (value, ptr, ld, vl).
          std::rotate(Ops.begin(), Ops.begin() + 2, Ops.begin() + 3);
          IntrinsicTypes = {Ops[0]->getType(), Ops[1]->getType(),
                            Ops[3]->getType()};
        }
      }
    }] in {
  // Default-lambda forms.
  let Log2LMUL = [0, 1, 2, 3] in {
    let RequiredFeatures = ["experimental-zvvmtls"],
        Name = "vmts_v",
        OverloadedName = "vmts_v",
        IRName = "vmts",
        MaskedIRName = "vmts_mask" in
      def vmts_v : RVVBuiltin<"v", "0Pezv", "csilxfd">;

    let RequiredFeatures = ["experimental-zvvmttls"],
        Name = "vmtts_v",
        OverloadedName = "vmtts_v",
        IRName = "vmtts",
        MaskedIRName = "vmtts_mask" in
      def vmtts_v : RVVBuiltin<"v", "0Pezv", "csilxfd">;

    // Unsigned integer element types.
    let RequiredFeatures = ["experimental-zvvmtls"],
        Name = "vmts_v",
        OverloadedName = "vmts_v",
        IRName = "vmts",
        MaskedIRName = "vmts_mask" in
      def vmts_v_u : RVVBuiltin<"Uv", "0PUezUv", "csil">;

    let RequiredFeatures = ["experimental-zvvmttls"],
        Name = "vmtts_v",
        OverloadedName = "vmtts_v",
        IRName = "vmtts",
        MaskedIRName = "vmtts_mask" in
      def vmtts_v_u : RVVBuiltin<"Uv", "0PUezUv", "csil">;
  }

  // Per-lambda `_L{N}` override variants for stores.
  foreach tp = ZvvmTileTypeCells in {
    foreach l2 = [0, 1, 2, 3] in {
      defvar tsuf = tp[1] # "m" # !shl(1, l2);
      foreach lam = ZvvmTileLambdas in {
        let Log2LMUL = [l2] in {
          let RequiredFeatures = ["experimental-zvvmtls"],
              Name = "vmts_v_" # tsuf # "_L" # lam,
              OverloadedName = "vmts_v_" # tsuf # "_L" # lam,
              IRName = "vmts_l" # lam,
              MaskedIRName = "vmts_l" # lam # "_mask" in
            def : RVVBuiltin<"", tp[3], tp[0]>;

          let RequiredFeatures = ["experimental-zvvmttls"],
              Name = "vmtts_v_" # tsuf # "_L" # lam,
              OverloadedName = "vmtts_v_" # tsuf # "_L" # lam,
              IRName = "vmtts_l" # lam,
              MaskedIRName = "vmtts_l" # lam # "_mask" in
            def : RVVBuiltin<"", tp[3], tp[0]>;
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Storage-width-qualified tile load/store: `_as_e{S}` (order-preserving pair
// only, per spec §"Alternate-format and storage-width-qualified tile
// load/store intrinsics").
//
// The instruction moves S-bit storage elements (vtype.SEW = S); the logical
// type in the name describes the register-group interpretation only. `ld`
// and `vl` are counted in S-bit storage elements, the base pointer is
// `const uint{S}_t *` / `uint{S}_t *` regardless of the logical type, and
// the transfer is bit-preserving (no numeric conversion). For a masked
// form, each mask bit controls one S-bit storage element, so the mask ratio
// is S/LMUL — e.g. `__riscv_vmtl_v_f16m1_as_e32_m` takes vbool32_t.
//
// Generated C signatures (f16m1_as_e32 shown):
//   vfloat16m1_t __riscv_vmtl_v_f16m1_as_e32   (const uint32_t *base,
//                                               size_t ld, size_t vl);
//   vfloat16m1_t __riscv_vmtl_v_f16m1_as_e32_m (vbool32_t mask,
//                                               const uint32_t *base,
//                                               size_t ld, size_t vl);
//   void __riscv_vmts_v_f16m1_as_e32 (uint32_t *base, size_t ld,
//                                     vfloat16m1_t vs3, size_t vl);
//
// Implementation: each def is BASED ON THE STORAGE integer type (TypeRange =
// storage char, Log2LMUL = the tile's register-group size), which makes the
// auto-generated mask type the spec's S/LMUL ratio and the pointer the
// storage-width unsigned element pointer for free. The logical vector type
// is spelled from the storage base via `(FixedSEW:W)` plus the `U`/`F`
// scalar-kind modifiers (plain `v`/`Uv`/`Fv` when W == S — applyFixedSEW
// rejects a same-SEW request). Codegen calls the tile intrinsic at the
// storage container type and reinterprets to/from the logical type with a
// bit-preserving bitcast (equal register-group size by construction).
//
// When S equals the logical type's natural width the `_as_e{S}` form is the
// spec-mandated alias of the unqualified intrinsic.
//
// No `_tu` variants are emitted for `_as_e{S}` (UnMaskedPolicyScheme =
// NonePolicy): the spec defines none, and tile loads are architecturally
// always tail/inactive-undisturbed anyway.
//
// Non-ISO logical types (bf16, OFP8, OFP4, int4) are deferred with the
// MX/format phase alongside their vector types.
//===----------------------------------------------------------------------===//

// Codegen for `_as_e{S}` loads: call the tile-load intrinsic at the S-bit
// storage container type and bitcast the result to the logical type.
class ZvvmAsELoadCodegen<string sw> {
  code Str = [{
      {
        const unsigned StorageEW = }] # sw # [{;
        auto *LogicalTy = cast<llvm::ScalableVectorType>(ResultType);
        unsigned TotalBits =
            LogicalTy->getElementType()->getScalarSizeInBits() *
            LogicalTy->getMinNumElements();
        auto *StorageTy = llvm::ScalableVectorType::get(
            llvm::IntegerType::get(getLLVMContext(), StorageEW),
            TotalBits / StorageEW);
        if (IsMasked) {
          // Builtin: (mask, ptr, ld, vl).
          // Intrinsic: (passthru, ptr, ld, mask, vl, policy).
          std::rotate(Ops.begin(), Ops.begin() + 1, Ops.end() - 1);
          Ops.push_back(ConstantInt::get(Ops.back()->getType(), PolicyAttrs));
          Ops.insert(Ops.begin(), llvm::PoisonValue::get(StorageTy));
          IntrinsicTypes = {StorageTy, Ops[1]->getType(), Ops[4]->getType()};
        } else {
          // Builtin: (ptr, ld, vl). Intrinsic: (passthru, ptr, ld, vl).
          Ops.insert(Ops.begin(), llvm::PoisonValue::get(StorageTy));
          IntrinsicTypes = {StorageTy, Ops[1]->getType(), Ops[3]->getType()};
        }
        llvm::Value *Load =
            Builder.CreateCall(CGM.getIntrinsic(ID, IntrinsicTypes), Ops, "");
        return Builder.CreateBitCast(Load, ResultType);
      }
  }];
}

// Codegen for `_as_e{S}` stores: bitcast the logical value to the S-bit
// storage container type and call the tile-store intrinsic there.
class ZvvmAsEStoreCodegen<string sw> {
  code Str = [{
      {
        const unsigned StorageEW = }] # sw # [{;
        unsigned ValueIdx = IsMasked ? 3 : 2;
        auto *LogicalTy =
            cast<llvm::ScalableVectorType>(Ops[ValueIdx]->getType());
        unsigned TotalBits =
            LogicalTy->getElementType()->getScalarSizeInBits() *
            LogicalTy->getMinNumElements();
        auto *StorageTy = llvm::ScalableVectorType::get(
            llvm::IntegerType::get(getLLVMContext(), StorageEW),
            TotalBits / StorageEW);
        Ops[ValueIdx] = Builder.CreateBitCast(Ops[ValueIdx], StorageTy);
        if (IsMasked) {
          // Builtin: (mask, ptr, ld, value, vl).
          // Intrinsic: (value, ptr, ld, mask, vl).
          std::swap(Ops[0], Ops[3]);
          IntrinsicTypes = {StorageTy, Ops[1]->getType(), Ops[4]->getType()};
        } else {
          // Builtin: (ptr, ld, value, vl). Intrinsic: (value, ptr, ld, vl).
          std::rotate(Ops.begin(), Ops.begin() + 2, Ops.begin() + 3);
          IntrinsicTypes = {StorageTy, Ops[1]->getType(), Ops[3]->getType()};
        }
      }
  }];
}

// (logical token, logical-vector prototype from the storage base) cells per
// storage width. `F`/`U` swap the scalar kind at the (FixedSEW-adjusted)
// element width; natural-width cells omit FixedSEW.
defvar ZvvmAsECells_e8  = [["i8", "v"], ["u8", "Uv"]];
defvar ZvvmAsECells_e16 = [["i8",  "(FixedSEW:8)v"],  ["u8",  "(FixedSEW:8)Uv"],
                           ["i16", "v"],              ["u16", "Uv"],
                           ["f16", "Fv"]];
defvar ZvvmAsECells_e32 = [["i8",  "(FixedSEW:8)v"],  ["u8",  "(FixedSEW:8)Uv"],
                           ["i16", "(FixedSEW:16)v"], ["u16", "(FixedSEW:16)Uv"],
                           ["f16", "(FixedSEW:16)Fv"],
                           ["i32", "v"],              ["u32", "Uv"],
                           ["f32", "Fv"]];
defvar ZvvmAsECells_e64 = [["i8",  "(FixedSEW:8)v"],  ["u8",  "(FixedSEW:8)Uv"],
                           ["i16", "(FixedSEW:16)v"], ["u16", "(FixedSEW:16)Uv"],
                           ["f16", "(FixedSEW:16)Fv"],
                           ["i32", "(FixedSEW:32)v"], ["u32", "(FixedSEW:32)Uv"],
                           ["f32", "(FixedSEW:32)Fv"],
                           ["i64", "v"],              ["u64", "Uv"],
                           ["f64", "Fv"]];

foreach st = [["c", "8"], ["s", "16"], ["i", "32"], ["l", "64"]] in {
  defvar cells = !cond(!eq(st[1], "8")  : ZvvmAsECells_e8,
                       !eq(st[1], "16") : ZvvmAsECells_e16,
                       !eq(st[1], "32") : ZvvmAsECells_e32,
                       true             : ZvvmAsECells_e64);

  // Loads.
  let HasMasked = true,
      HasMaskedOffOperand = false,
      MaskedPolicyScheme = NonePolicy,
      UnMaskedPolicyScheme = NonePolicy,
      RequiredFeatures = ["experimental-zvvmtls"],
      ManualCodegen = ZvvmAsELoadCodegen<st[1]>.Str in {
    foreach l2 = [0, 1, 2, 3] in {
      defvar lmul = !shl(1, l2);
      foreach cell = cells in {
        defvar tsuf = cell[0] # "m" # lmul # "_as_e" # st[1];
        defvar loadProto = cell[1] # "PCUez";
        let Log2LMUL = [l2] in {
          let Name = "vmtl_v_" # tsuf,
              OverloadedName = "vmtl_v_" # tsuf,
              IRName = "vmtl",
              MaskedIRName = "vmtl_mask" in
            def : RVVBuiltin<"", loadProto, st[0]>;

          foreach lam = ZvvmTileLambdas in {
            let Name = "vmtl_v_" # tsuf # "_L" # lam,
                OverloadedName = "vmtl_v_" # tsuf # "_L" # lam,
                IRName = "vmtl_l" # lam,
                MaskedIRName = "vmtl_l" # lam # "_mask" in
              def : RVVBuiltin<"", loadProto, st[0]>;
          }
        }
      }
    }
  }

  // Stores.
  let HasMasked = true,
      HasMaskedOffOperand = false,
      MaskedPolicyScheme = NonePolicy,
      RequiredFeatures = ["experimental-zvvmtls"],
      ManualCodegen = ZvvmAsEStoreCodegen<st[1]>.Str in {
    foreach l2 = [0, 1, 2, 3] in {
      defvar lmul = !shl(1, l2);
      foreach cell = cells in {
        defvar tsuf = cell[0] # "m" # lmul # "_as_e" # st[1];
        defvar storeProto = "0PUez" # cell[1];
        // Natural-width `_as_e{S}` cells are spec-mandated ALIASES of the
        // plain store (identical C signature), so only the non-natural cells
        // join the overloaded `vmts_v` short-form set — otherwise the set
        // would contain duplicate signatures and every such call would be
        // ambiguous.
        defvar isNaturalW = !or(!eq(cell[1], "v"), !eq(cell[1], "Uv"),
                                !eq(cell[1], "Fv"));
        let Log2LMUL = [l2] in {
          let Name = "vmts_v_" # tsuf,
              OverloadedName = !if(isNaturalW, "vmts_v_" # tsuf, "vmts_v"),
              IRName = "vmts",
              MaskedIRName = "vmts_mask" in
            def : RVVBuiltin<"", storeProto, st[0]>;

          foreach lam = ZvvmTileLambdas in {
            let Name = "vmts_v_" # tsuf # "_L" # lam,
                OverloadedName = "vmts_v_" # tsuf # "_L" # lam,
                IRName = "vmts_l" # lam,
                MaskedIRName = "vmts_l" # lam # "_mask" in
              def : RVVBuiltin<"", storeProto, st[0]>;
          }
        }
      }
    }
  }
}
