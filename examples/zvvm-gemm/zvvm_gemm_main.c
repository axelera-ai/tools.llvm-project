// Standalone Zvvm GEMM test program for spike simulation.
//
// Build (RV64, freestanding, assembly):
//   clang -cc1 -triple riscv64 -target-feature +v \
//         -target-feature +experimental-zvvi32mm \
//         -target-feature +experimental-zvvmtls \
//         -O2 -S -nostdsysteminc \
//         -internal-isystem <clang-resource>/include \
//         zvvm_gemm_main.c -o zvvm_gemm.s
//
// Geometry assumed by the kernel: SEW=32, LMUL=1, lambda=2 (encoding L2).
// On a VLEN=128 implementation this gives EMUL_C=1, K_eff=2, M_tile=2.
//
// The matrices are stored as:
//   A: M x K, row-major,    stride lda
//   B: N x K, "column-major" wrt the C tile (vmtl.v loads B^T tiles directly),
//             stride ldb
//   C: M x N, row-major,    stride ldc
//
// main() runs the kernel on a 4x4 all-ones input and returns C[0][0]. Each
// element of C should be K = 4, so main returns 4 on success.

#pragma clang riscv intrinsic zvvm_vector

#include <riscv_vector.h>

void gemm_i32(const int *A, const int *B, int *C,
              unsigned long M, unsigned long N, unsigned long K,
              unsigned long lda, unsigned long ldb, unsigned long ldc) {
  const unsigned long K_eff  = 2;
  const unsigned long M_tile = 2;
  const unsigned long LAMBDA = 2; // L2

  for (unsigned long j = 0; j < N;) {
    unsigned long vl = __riscv_vsetvl_matrix(K_eff * (N - j),
                                             /*sew=*/2, /*lmul=*/0,
                                             /*vta=*/1, /*vma=*/1,
                                             /*lambda=*/LAMBDA, /*bs=*/0,
                                             /*altfmt_a=*/0, /*altfmt_b=*/0);
    unsigned long N_tile = vl / K_eff;
    if (N_tile == 0)
      break;

    for (unsigned long i = 0; i < M; i += M_tile) {
      vint32m1_t c = __riscv_vmv_v_x_i32m1(0, vl);

      for (unsigned long k = 0; k < K; k += K_eff) {
        vint32m1_t a = __riscv_vmtl_v_i32m1(&A[i * lda + k], lda, vl);
        vint32m1_t b = __riscv_vmtl_v_i32m1(&B[j * ldb + k], ldb, vl);
        c = __riscv_vmmacc_vv_i32m1(c, a, b, vl);
      }

      __riscv_vmts_v_i32m1(&C[i * ldc + j], ldc, c, vl);
    }

    j += N_tile;
  }
}

// 4x4 all-ones input; expected C[i][j] = K = 4.
static int A[16] = { 1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1 };
static int B[16] = { 1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1 };
static int C[16];

int main(void) {
  gemm_i32(A, B, C,
           /*M=*/4, /*N=*/4, /*K=*/4,
           /*lda=*/4, /*ldb=*/4, /*ldc=*/4);
  return C[0]; // expect 4
}
