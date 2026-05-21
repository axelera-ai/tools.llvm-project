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
// Zvvmtls (tile load/store), Zvvmttls (transposing tile load/store).
//
// Spec: https://github.com/riscv/integrated-matrix-extension
//
// Current scope:
//   * vsetvl_matrix — full-vtype write (covers lambda / bs / altfmt_A / altfmt_B).
//   * vmmacc_vv     — non-widening integer matrix multiply-accumulate, SEW=32
//                     LMUL=1 only. (Widening variants and broader (SEW, LMUL)
//                     coverage land in follow-up patches.)
//   * vmtl_v / vmts_v — order-preserving tile load/store, default-lambda
//                       unmasked, SEW=32 LMUL=1 only. (Lambda override _L{N},
//                       masked variants, and the transposing vmttl/vmtts
//                       mnemonics land in follow-ups.)
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
// Zvvmtls: order-preserving tile load and store.
//
// Generated C signatures for TypeRange "i", LMUL=1 (default lambda override):
//   vint32m1_t __riscv_vmtl_v_i32m1(const int32_t *base, size_t ld, size_t vl);
//   void       __riscv_vmts_v_i32m1(int32_t *base, size_t ld,
//                                   vint32m1_t vs3, size_t vl);
//
// Per-instruction lambda override (_L{N} variants) and masked variants are
// added in follow-up patches.
//===----------------------------------------------------------------------===//

// Tile load: passthru is poisoned for unmasked builds (standard RVV pattern
// for tail-undisturbed); the IR intrinsic threads it through arg 0.
let RequiredFeatures = ["experimental-zvvmtls"],
    HasMasked = false,
    UnMaskedPolicyScheme = HasPassthruOperand,
    Log2LMUL = [0],
    Name = "vmtl_v",
    IRName = "vmtl" in
  def vmtl_v : RVVOutOp0Builtin<"v", "vPCez", "i">;

// Tile store: builtin order is (ptr, ld, vec, vl), IR intrinsic order is
// (vec, ptr, ld, vl). Rotate to match the IR shape.
let RequiredFeatures = ["experimental-zvvmtls"],
    HasMasked = false,
    Log2LMUL = [0],
    Name = "vmts_v",
    IRName = "vmts",
    ManualCodegen = [{
      // Builtin: (ptr, ld, value, vl). Intrinsic: (value, ptr, ld, vl).
      std::rotate(Ops.begin(), Ops.begin() + 2, Ops.begin() + 3);
      IntrinsicTypes = {Ops[0]->getType(), Ops[1]->getType(),
                        Ops[2]->getType()};
    }] in
  def vmts_v : RVVBuiltin<"v", "0Pezv", "i">;
