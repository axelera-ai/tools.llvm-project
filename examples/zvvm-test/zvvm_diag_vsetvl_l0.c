// Diagnostic 3: register-form vsetvl with NO matrix bits set.
//
// __riscv_vsetvl_matrix always lowers to the register-form `vsetvl` (rd, rs1,
// rs2) instruction — but here we pass lambda=0, bs=0, altfmt_*=0, so the
// vtype value is just 0xD0 (the standard vsew/vlmul/vta/vma bits). Bits
// 60..63 are clear.
//
// Outcomes:
//   8  — register-form vsetvl works on your spike; the matrix vtype bits
//        are what hangs it.
//   0  — register-form vsetvl returns with vill=1 even on a standard vtype.
//   stuck — your spike's register-form vsetvl handler hangs unconditionally
//           (whether or not high-end bits are set).

#pragma clang riscv intrinsic zvvm_vector
#include <riscv_vector.h>

int main(void) {
  unsigned long vl = __riscv_vsetvl_matrix(/*avl=*/8,
                                           /*sew=*/2,    /*lmul=*/0,
                                           /*vta=*/1,    /*vma=*/1,
                                           /*lambda=*/0, /*bs=*/0,
                                           /*altfmt_a=*/0, /*altfmt_b=*/0);
  return (int)vl;
}
