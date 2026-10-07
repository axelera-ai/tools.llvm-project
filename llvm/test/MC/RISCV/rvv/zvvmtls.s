# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v,+experimental-zvvmtls %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v,+experimental-zvvmtls %s \
# RUN:        | llvm-objdump -d --mattr=+v,+experimental-zvvmtls --no-print-imm-hex - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v,+experimental-zvvmtls %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

# Zvvmtls: order-preserving tile load/store.
# Encoding: opcode 0x07 (LOAD-FP) for vmtl.v, 0x27 (STORE-FP) for vmts.v;
# funct3 = 0b111 (width), MEW=1, mop=0b00 (order-preserving), lambda is the
# 3-bit immediate at bits[31:29]:
#   omitted/0b000 = use vtype.lambda, 0b001=L1, 0b010=L2, 0b011=L4,
#   0b100=L8, 0b101=L16, 0b110=L32, 0b111=L64.

vmtl.v v8, (a0), a1
# CHECK-INST: vmtl.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x12]
# CHECK-UNKNOWN: 12b57407 <unknown>

vmtl.v v8, (a0), a1, L1
# CHECK-INST: vmtl.v v8, (a0), a1, L1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x32]
# CHECK-UNKNOWN: 32b57407 <unknown>

vmtl.v v8, (a0), a1, L4
# CHECK-INST: vmtl.v v8, (a0), a1, L4
# CHECK-ENCODING: [0x07,0x74,0xb5,0x72]
# CHECK-UNKNOWN: 72b57407 <unknown>

vmtl.v v8, (a0), a1, L64
# CHECK-INST: vmtl.v v8, (a0), a1, L64
# CHECK-ENCODING: [0x07,0x74,0xb5,0xf2]
# CHECK-UNKNOWN: f2b57407 <unknown>

vmtl.v v8, (a0), a1, v0.t
# CHECK-INST: vmtl.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x10]
# CHECK-UNKNOWN: 10b57407 <unknown>

vmtl.v v8, (a0), a1, L4, v0.t
# CHECK-INST: vmtl.v v8, (a0), a1, L4, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x70]
# CHECK-UNKNOWN: 70b57407 <unknown>

vmts.v v8, (a0), a1
# CHECK-INST: vmts.v v8, (a0), a1
# CHECK-ENCODING: [0x27,0x74,0xb5,0x12]
# CHECK-UNKNOWN: 12b57427 <unknown>

vmts.v v8, (a0), a1, L4
# CHECK-INST: vmts.v v8, (a0), a1, L4
# CHECK-ENCODING: [0x27,0x74,0xb5,0x72]
# CHECK-UNKNOWN: 72b57427 <unknown>

vmts.v v8, (a0), a1, v0.t
# CHECK-INST: vmts.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x74,0xb5,0x10]
# CHECK-UNKNOWN: 10b57427 <unknown>

vmts.v v8, (a0), a1, L4, v0.t
# CHECK-INST: vmts.v v8, (a0), a1, L4, v0.t
# CHECK-ENCODING: [0x27,0x74,0xb5,0x70]
# CHECK-UNKNOWN: 70b57427 <unknown>
