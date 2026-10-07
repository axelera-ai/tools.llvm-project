# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-zvvmm %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-zvvmm %s \
# RUN:        | llvm-objdump -d --mattr=+experimental-zvvmm --no-print-imm-hex - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-zvvmm %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

# Encodings: OP-V (0x57) opcode, OPIVV (funct3=0), vm=1, funct6 selects mnemonic.
# Operand layout: vd, vs1, vs2 in asm (note: in encoding bits[24:20]=vs2,
# bits[19:15]=vs1 -- the asm order differs from the bit-field order, matching
# the VMACVV convention).

vmmacc.vv v8, v4, v20
# CHECK-INST: vmmacc.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x42,0xe3]
# CHECK-UNKNOWN: e3420457 <unknown>

vwmmacc.vv v8, v4, v20
# CHECK-INST: vwmmacc.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x42,0xe7]
# CHECK-UNKNOWN: e7420457 <unknown>

vqmmacc.vv v8, v4, v20
# CHECK-INST: vqmmacc.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x42,0xeb]
# CHECK-UNKNOWN: eb420457 <unknown>

v8wmmacc.vv v8, v4, v20
# CHECK-INST: v8wmmacc.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x42,0xef]
# CHECK-UNKNOWN: ef420457 <unknown>
