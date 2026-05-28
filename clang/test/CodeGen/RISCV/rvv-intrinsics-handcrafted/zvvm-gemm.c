// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +v \
// RUN:   -target-feature +experimental-zvvmm \
// RUN:   -target-feature +experimental-zvvmtls \
// RUN:   -O2 -emit-llvm %s -o - | FileCheck %s
//
// End-to-end test that the Phase-2 Zvvm Clang builtins — vsetvl_matrix,
// vmtl_v, vmts_v, and the previously landed vmmacc_vv — compose into
// a small integer GEMM kernel. Geometry: SEW=32, LMUL=1, lambda=2
// (the only (SEW, LMUL, lambda) combo the current builtins cover).

#pragma clang riscv intrinsic zvvm_vector

#include <riscv_vector.h>

// -- per-builtin smoke tests --------------------------------------------------

// CHECK-LABEL: define dso_local i64 @test_vsetvl_matrix
// CHECK:   call i64 @llvm.riscv.vsetvl.matrix.i64
size_t test_vsetvl_matrix(size_t avl) {
  return __riscv_vsetvl_matrix(avl, /*sew=*/2, /*lmul=*/0,
                               /*vta=*/1, /*vma=*/1,
                               /*lambda=*/2, /*bs=*/0,
                               /*altfmt_a=*/0, /*altfmt_b=*/0);
}

// CHECK-LABEL: define dso_local <vscale x 2 x i32> @test_vmtl_v_i32m1
// CHECK:   call <vscale x 2 x i32> @llvm.riscv.vmtl.nxv2i32
vint32m1_t test_vmtl_v_i32m1(const int32_t *base, size_t ld, size_t vl) {
  return __riscv_vmtl_v_i32m1(base, ld, vl);
}

// CHECK-LABEL: define dso_local void @test_vmts_v_i32m1
// CHECK:   call void @llvm.riscv.vmts.nxv2i32
void test_vmts_v_i32m1(int32_t *base, size_t ld, vint32m1_t v, size_t vl) {
  __riscv_vmts_v_i32m1(base, ld, v, vl);
}

// -- small integer GEMM ------------------------------------------------------
//
// C[M, N] += A[M, K] x B[N, K]. A is row-major (stride lda), B is stored
// as B^T row-major / B column-major (stride ldb), C is row-major (stride
// ldc). All three tiles can therefore be loaded with the order-preserving
// vmtl.v.

// All five intrinsic calls survive -O2 in the inlined body; the optimizer
// reorders the basic blocks, so use CHECK-DAG to match them in any order.
//
// CHECK-LABEL: define dso_local void @gemm_i32
// CHECK-DAG:     call i64 @llvm.riscv.vsetvl.matrix.i64
// CHECK-DAG:     call <vscale x 2 x i32> @llvm.riscv.vmtl.nxv2i32
// CHECK-DAG:     call <vscale x 2 x i32> @llvm.riscv.vmtl.nxv2i32
// CHECK-DAG:     call <vscale x 2 x i32> @llvm.riscv.vmmacc.nxv2i32.nxv2i32.i64
// CHECK-DAG:     call void @llvm.riscv.vmts.nxv2i32
void gemm_i32(const int32_t *A, const int32_t *B, int32_t *C,
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
      // Zero the C accumulator tile.
      vint32m1_t c = __riscv_vmv_v_x_i32m1(0, vl);

      // Inner K loop: accumulate A[i..][k..] x B[j..][k..] into C tile.
      for (size_t k = 0; k < K; k += K_eff) {
        vint32m1_t a = __riscv_vmtl_v_i32m1(&A[i * lda + k], lda, vl);
        vint32m1_t b = __riscv_vmtl_v_i32m1(&B[j * ldb + k], ldb, vl);
        c = __riscv_vmmacc_vv_i32m1_lm1(c, a, b, vl);
      }

      __riscv_vmts_v_i32m1(&C[i * ldc + j], ldc, c, vl);
    }

    j += N_tile;
  }
}
