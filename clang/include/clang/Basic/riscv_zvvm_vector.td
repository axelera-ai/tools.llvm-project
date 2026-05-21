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
// Zvvmm: integer matrix multiply-accumulate
//===----------------------------------------------------------------------===//

// Matrix multiply-accumulate is never masked (the encoding requires vm=1)
// and exposes no per-instruction policy operand — vta/vma come from vtype.
// vd is read+written, so the builtin's first vector argument is the
// passthrough accumulator.

let RequiredFeatures = ["experimental-zvvmm"],
    HasMasked = false,
    UnMaskedPolicyScheme = NonePolicy,
    Log2LMUL = [0],
    // The IR intrinsic has two independent vector-type overloads: C (the
    // accumulator, == return type, index -1) and A/B (operand 1).
    IntrinsicTypes = [-1, 1],
    IRName = "vmmacc" in {
  // vmmacc.vv — W = 1, A/B and C all have the same SEW.
  //
  // Generated C signature for TypeRange "i", LMUL=1:
  //   vint32m1_t __riscv_vmmacc_vv_i32m1(vint32m1_t vd, vint32m1_t vs1,
  //                                       vint32m1_t vs2, size_t vl);
  def vmmacc_vv : RVVBuiltin<"v", "vvvv", "i">;
}

//===----------------------------------------------------------------------===//
// Zvvfmm: floating-point matrix multiply-accumulate.
//
// Matrix multiply-accumulate is never masked from the C surface: the
// underlying encoding uses vm=1 (Phase 1 does not expose the vm=0 microscaled
// MX* form). vta/vma come from vtype.
//
// Only the non-widening vfmmacc.vv is exposed at the Clang level in Phase 1.
// The widening variants need vd (= EMUL_C) and vs1/vs2 (= LMUL) to be
// independent — same EMUL_C ≠ LMUL gap that blocks the integer widening
// builtins. They land alongside the integer widening builtins when that
// coverage work is done.
//===----------------------------------------------------------------------===//

let RequiredFeatures = ["experimental-zvvfmm"],
    HasMasked = false,
    UnMaskedPolicyScheme = NonePolicy,
    Log2LMUL = [0],
    // The IR intrinsic has two independent vector-type overloads: C (the
    // accumulator, == return type, index -1) and A/B (operand 1).
    IntrinsicTypes = [-1, 1],
    IRName = "vfmmacc" in {
  // vfmmacc.vv — W = 1, A/B and C all have the same SEW.
  //
  // Generated C signature for TypeRange "f", LMUL=1:
  //   vfloat32m1_t __riscv_vfmmacc_vv_f32m1(vfloat32m1_t vd,
  //                                          vfloat32m1_t vs1,
  //                                          vfloat32m1_t vs2,
  //                                          size_t vl);
  def vfmmacc_vv : RVVBuiltin<"v", "vvvv", "f">;
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
let HasMasked = false,
    UnMaskedPolicyScheme = HasPassthruOperand,
    Log2LMUL = [0] in {
  let RequiredFeatures = ["experimental-zvvmtls"],
      Name = "vmtl_v",
      IRName = "vmtl" in
    def vmtl_v : RVVOutOp0Builtin<"v", "vPCez", "csilxfd">;

  let RequiredFeatures = ["experimental-zvvmttls"],
      Name = "vmttl_v",
      IRName = "vmttl" in
    def vmttl_v : RVVOutOp0Builtin<"v", "vPCez", "csilxfd">;
}

// Tile store: builtin order is (ptr, ld, vec, vl), IR intrinsic order is
// (vec, ptr, ld, vl). Rotate to match the IR shape.
let HasMasked = false,
    Log2LMUL = [0],
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
}
