# RUN: not llvm-mc -triple=riscv64 \
# RUN:     --mattr=+experimental-zvvmm,+experimental-zvvfmm,+experimental-zvvxi8fp16mm %s 2>&1 \
# RUN:        | FileCheck %s
# RUN: not llvm-mc -triple=riscv64 --mattr=+experimental-zvvmm,+experimental-zvvfmm \
# RUN:     --defsym=NOFEAT=1 %s 2>&1 | FileCheck %s --check-prefix=CHECK-NOFEAT

.ifndef NOFEAT

# vm=0 of the non-widening MACs is reserved: no v0.scale form exists.
vmmacc.vv v8, v4, v20, v0.scale
# CHECK: [[@LINE-1]]:24: error: invalid operand for instruction

vfmmacc.vv v8, v4, v20, v0.scale
# CHECK: [[@LINE-1]]:25: error: invalid operand for instruction

# Matrix MACs are never vector-masked; v0.t is not a valid trailing operand.
vfwmmacc.vv v8, v4, v20, v0.t
# CHECK: [[@LINE-1]]:26: error: expected '.scale' suffix

# The MX-only mnemonics require the v0.scale operand.
vfwimmacc.vv v8, v4, v20
# CHECK: [[@LINE-1]]:1: error: too few operands for instruction

.else

# Without any Zvvxi8* extension the MX-only mnemonic is unavailable, even
# with both family flags enabled.
vfwimmacc.vv v8, v4, v20, v0.scale
# CHECK-NOFEAT: [[@LINE-1]]:1: error: instruction requires the following: 'Zvvxi8fp16mm' or 'Zvvxi8bf16mm' (MXINT8 Widening Matrix Multiply-Accumulate)

.endif
