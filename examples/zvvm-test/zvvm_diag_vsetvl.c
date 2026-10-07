// Diagnostic 2: matrix vsetvl probe.
// Calls __riscv_vsetvl_matrix(avl=8, sew=e32, lmul=m1, vta=vma=1,
// lambda=2, bs=0, altfmt_a=altfmt_b=0) and returns the established vl.
//
// Expected outcomes when run as `spike ...; echo $?`:
//   8  — vsetvl wrote the full matrix vtype and returned a sensible vl.
//        Subsequent hangs in the full GEMM are about the matrix ops
//        themselves, not vsetvl.
//   0  — vsetvl set vill=1 (lambda=2 not WARL-supported on this spike).
//   no exit / "stuck" — spike's vsetvl handler hangs on the matrix vtype.

#pragma clang riscv intrinsic zvvm_vector
#include <riscv_vector.h>

int main(void) {
  unsigned long vl = __riscv_vsetvl_matrix(/*avl=*/8,
                                           /*sew=*/2,    /*lmul=*/0,
                                           /*vta=*/1,    /*vma=*/1,
                                           /*lambda=*/2, /*bs=*/0,
                                           /*altfmt_a=*/0, /*altfmt_b=*/0);
  return (int)vl;
}
