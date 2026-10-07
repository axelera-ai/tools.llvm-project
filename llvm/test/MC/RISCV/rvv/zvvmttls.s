# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v,+experimental-zvvmttls %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v,+experimental-zvvmttls %s \
# RUN:        | llvm-objdump -d --mattr=+v,+experimental-zvvmttls --no-print-imm-hex - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v,+experimental-zvvmttls %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

# Zvvmttls: transposing tile load/store.
# Encoding: same as Zvvmtls but with mop=0b01 (transposing).
# Lambda override: omitted = vtype.lambda, "L1".."L64" select explicit values.

vmttl.v v8, (a0), a1
# CHECK-INST: vmttl.v v8, (a0), a1
# CHECK-ENCODING: [0x07,0x74,0xb5,0x16]
# CHECK-UNKNOWN: 16b57407 <unknown>

vmttl.v v8, (a0), a1, L4
# CHECK-INST: vmttl.v v8, (a0), a1, L4
# CHECK-ENCODING: [0x07,0x74,0xb5,0x76]
# CHECK-UNKNOWN: 76b57407 <unknown>

vmttl.v v8, (a0), a1, v0.t
# CHECK-INST: vmttl.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x14]
# CHECK-UNKNOWN: 14b57407 <unknown>

vmttl.v v8, (a0), a1, L4, v0.t
# CHECK-INST: vmttl.v v8, (a0), a1, L4, v0.t
# CHECK-ENCODING: [0x07,0x74,0xb5,0x74]
# CHECK-UNKNOWN: 74b57407 <unknown>

vmtts.v v8, (a0), a1
# CHECK-INST: vmtts.v v8, (a0), a1
# CHECK-ENCODING: [0x27,0x74,0xb5,0x16]
# CHECK-UNKNOWN: 16b57427 <unknown>

vmtts.v v8, (a0), a1, L4
# CHECK-INST: vmtts.v v8, (a0), a1, L4
# CHECK-ENCODING: [0x27,0x74,0xb5,0x76]
# CHECK-UNKNOWN: 76b57427 <unknown>

vmtts.v v8, (a0), a1, v0.t
# CHECK-INST: vmtts.v v8, (a0), a1, v0.t
# CHECK-ENCODING: [0x27,0x74,0xb5,0x14]
# CHECK-UNKNOWN: 14b57427 <unknown>

vmtts.v v8, (a0), a1, L4, v0.t
# CHECK-INST: vmtts.v v8, (a0), a1, L4, v0.t
# CHECK-ENCODING: [0x27,0x74,0xb5,0x74]
# CHECK-UNKNOWN: 74b57427 <unknown>
