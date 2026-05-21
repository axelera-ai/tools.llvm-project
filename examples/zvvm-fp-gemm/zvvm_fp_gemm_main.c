// Standalone Zvvfmm FP32 GEMM test program for spike simulation.
//
// FP32 analogue of examples/zvvm-gemm — same geometry (SEW=32, LMUL=1,
// lambda=2; K_eff=2, M_tile=2 at VLEN=128, EMUL_C=1), but driving the
// floating-point matrix multiply-accumulate (vfmmacc.vv) and the
// recently-added native f32m1 tile load/store builtins.
//
// Build (RV64, freestanding, assembly):
//   clang -cc1 -triple riscv64 -target-feature +v -target-feature +f
//         -target-feature +d
//         -target-feature +experimental-zvvmm
//         -target-feature +experimental-zvvmtls
//         -target-feature +experimental-zvvfmm
//         -O2 -S -nostdsysteminc
//         -internal-isystem <clang-resource>/include
//         zvvm_fp_gemm_main.c -o zvvm_fp_gemm.s
//
// The matrices are stored as:
//   A: M x K, row-major,    stride lda
//   B: N x K, "column-major" wrt the C tile (vmtl.v loads B^T tiles directly),
//             stride ldb
//   C: M x N, row-major,    stride ldc
//
// main() runs the kernel on a 4x4 input of mixed-sign FP32 values whose
// numerators are small integers and whose denominators are powers of two —
// every product (and partial sum) is exactly representable in fp32, so the
// answer doesn't drift with operand ordering or fused multiply-add. The
// dot product of A[0,:] and B[0,:] (which the kernel writes into C[0,0])
// is exactly 7.0, so main returns 7 on success.

#pragma clang riscv intrinsic zvvm_vector

#include <riscv_vector.h>

void gemm_f32(const float *A, const float *B, float *C,
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
      vfloat32m1_t c = __riscv_vfmv_v_f_f32m1(0.0f, vl);

      for (unsigned long k = 0; k < K; k += K_eff) {
        vfloat32m1_t a = __riscv_vmtl_v_f32m1(&A[i * lda + k], lda, vl);
        vfloat32m1_t b = __riscv_vmtl_v_f32m1(&B[j * ldb + k], ldb, vl);
        c = __riscv_vfmmacc_vv_f32m1(c, a, b, vl);
      }

      __riscv_vmts_v_f32m1(&C[i * ldc + j], ldc, c, vl);
    }

    j += N_tile;
  }
}

// Both matrices are row-major 4x4. Values are mixed-sign fp32 with
// power-of-two denominators (every element exactly representable). The
// kernel computes C = A * B^T, so C[0,0] = A[0,:] . B[0,:].
//
//   A[0,:] . B[0,:] = 1.5*2.0 + 0.5*4.0 + 2.0*1.0 + 0.25*0.0
//                   = 3.0   + 2.0   + 2.0   + 0.0
//                   = 7.0
static float A[16] = {  1.5f,   0.5f,   2.0f,   0.25f,
                       -0.75f,  1.25f,  0.5f,   1.0f,
                        2.0f,   0.25f,  1.5f,   0.5f,
                        0.5f,  -1.0f,  -0.25f,  1.5f  };
static float B[16] = {  2.0f,   4.0f,   1.0f,   0.0f,
                        1.0f,  -0.5f,   0.25f,  2.0f,
                        0.5f,   1.5f,   2.0f,  -0.25f,
                       -0.25f,  0.5f,   1.0f,   0.75f };
static float C[16];

int main(void) {
  gemm_f32(A, B, C,
           /*M=*/4, /*N=*/4, /*K=*/4,
           /*lda=*/4, /*ldb=*/4, /*ldc=*/4);
  return (int)C[0]; // expect 7
}
