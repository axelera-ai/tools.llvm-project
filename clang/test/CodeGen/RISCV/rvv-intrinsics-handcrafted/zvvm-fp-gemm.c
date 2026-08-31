// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +v \
// RUN:   -target-feature +experimental-zvvmm -target-feature +experimental-zvvfp32mm \
// RUN:   -target-feature +experimental-zvvmtls \
// RUN:   -target-feature +experimental-zvvfmm \
// RUN:   -O2 -emit-llvm %s -o - | FileCheck %s
//
// End-to-end test that the Phase-1 Zvvfmm Clang builtin (vfmmacc_vv) composes
// with the Phase-2 Zvvm builtins (vsetvl_matrix, vmtl_v, vmts_v) into a small
// FP32 GEMM kernel. Geometry: SEW=32, LMUL=1, lambda=2 — the same
// (SEW, LMUL, lambda) tuple the integer GEMM test pins.
//
// Tile load/store now exposes the f32m1 element type natively, so the kernel
// drives __riscv_vmtl_v_f32m1 / __riscv_vmts_v_f32m1 directly without an
// integer-bridge vreinterpret.

#pragma clang riscv intrinsic zvvm_vector

#include <riscv_vector.h>

// -- small FP32 GEMM ---------------------------------------------------------
//
// C[M, N] += A[M, K] x B[N, K]. Same shape as zvvm-gemm.c (the integer
// version): A is row-major (stride lda), B is stored as B^T row-major / B
// column-major (stride ldb), C is row-major (stride ldc). All three tiles
// can therefore be loaded with the order-preserving vmtl.v.

// All four IR intrinsic calls survive -O2 inlining; the optimizer reorders
// the basic blocks, so use CHECK-DAG to match them in any order. The
// .nxv2f32 overloads pin the kernel to the FP path end-to-end, so an
// accidental regression to either the integer vmmacc or an integer tile LS
// path would fail the test.
//
// CHECK-LABEL: define dso_local void @gemm_f32
// CHECK-DAG:     call i64 @llvm.riscv.vsetvl.matrix.i64
// CHECK-DAG:     call <vscale x 2 x float> @llvm.riscv.vmtl.nxv2f32
// CHECK-DAG:     call <vscale x 2 x float> @llvm.riscv.vmtl.nxv2f32
// CHECK-DAG:     call <vscale x 2 x float> @llvm.riscv.vfmmacc.nxv2f32.nxv2f32.i64
// CHECK-DAG:     call void @llvm.riscv.vmts.nxv2f32
void gemm_f32(const float *A, const float *B, float *C,
              size_t M, size_t N, size_t K,
              size_t lda, size_t ldb, size_t ldc) {
  // K_eff = lambda * LMUL = 2 * 1 = 2.
  // M_tile = VLEN / (SEW * lambda) = 128 / (32 * 2) = 2  for VLEN=128.
  const size_t K_eff  = 2;
  const size_t M_tile = 2;
  const size_t LAMBDA = 2; // encoding 2 = L2.

  for (size_t j = 0; j < N;) {
    // Configure vtype (e32 m1 ta ma lambda=2) and ask for the longest column
    // block that fits in the remaining N columns. vl is rounded down to a
    // multiple of lambda * LMUL by the hardware.
    size_t vl = __riscv_vsetvl_matrix(K_eff * (N - j),
                                      /*sew=*/2, /*lmul=*/0,
                                      /*vta=*/1, /*vma=*/1,
                                      /*lambda=*/LAMBDA, /*bs=*/0,
                                      /*altfmt_a=*/0, /*altfmt_b=*/0);
    size_t N_tile = vl / K_eff;
    if (N_tile == 0)
      break;

    for (size_t i = 0; i < M; i += M_tile) {
      // Zero the FP C accumulator tile.
      vfloat32m1_t c = __riscv_vfmv_v_f_f32m1(0.0f, vl);

      // Inner K loop.
      for (size_t k = 0; k < K; k += K_eff) {
        vfloat32m1_t a = __riscv_vmtl_v_f32m1(&A[i * lda + k], lda, vl);
        vfloat32m1_t b = __riscv_vmtl_v_f32m1(&B[j * ldb + k], ldb, vl);
        c = __riscv_vfmmacc_vv_f32m1(c, a, b, vl);
      }

      __riscv_vmts_v_f32m1(&C[i * ldc + j], ldc, c, vl);
    }

    j += N_tile;
  }
}
