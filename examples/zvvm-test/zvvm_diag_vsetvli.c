// Diagnostic 4: standard vsetvli (immediate form) for sanity.
//
// This uses Clang's built-in RVV vsetvl helper, which lowers to vsetvli
// with an 11-bit immediate — NOT the register-form vsetvl that the
// matrix intrinsic uses. If this works but diag_vsetvl_l0 hangs, your
// spike's vsetvl decode/handler for the register form is broken
// (independently of the matrix extension).
//
// Outcomes:
//   8  — standard vsetvli works; the issue is the register form / matrix bits.
//   anything else — even base RVV is having trouble.

#include <riscv_vector.h>

int main(void) {
  size_t vl = __riscv_vsetvl_e32m1(8);
  return (int)vl;
}
