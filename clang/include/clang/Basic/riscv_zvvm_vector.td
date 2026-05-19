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
// Phase 1 of the Clang exposure covers only the non-widening integer matrix
// MAC (`vmmacc.vv`). Widening MAC variants and tile load/store builtins are
// added in follow-up patches.
//
//===----------------------------------------------------------------------===//

include "riscv_vector_common.td"

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
