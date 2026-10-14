# RUN: llvm-mc -triple=riscv64 -show-encoding \
# RUN:     --mattr=+experimental-zvvfmm,+experimental-zvvxi8fp16mm,+experimental-zvvxi8fp32mm,+experimental-zvvxi8fp64mm %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj \
# RUN:     --mattr=+experimental-zvvfmm,+experimental-zvvxi8fp16mm,+experimental-zvvxi8fp32mm,+experimental-zvvxi8fp64mm %s \
# RUN:        | llvm-objdump -d \
# RUN:     --mattr=+experimental-zvvfmm,+experimental-zvvxi8fp16mm,+experimental-zvvxi8fp32mm,+experimental-zvvxi8fp64mm \
# RUN:     --no-print-imm-hex - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj \
# RUN:     --mattr=+experimental-zvvfmm,+experimental-zvvxi8fp16mm,+experimental-zvvxi8fp32mm,+experimental-zvvxi8fp64mm %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

# The BS=16 flags imply their BS=32 siblings, so the integer-input MX forms
# also assemble with only the Zvvxn* flags enabled:
# RUN: llvm-mc -triple=riscv64 -show-encoding \
# RUN:     --mattr=+experimental-zvvfmm,+experimental-zvvxni8fp16mm,+experimental-zvvxni8fp32mm,+experimental-zvvxni8fp64mm %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST

# Microscaled (vm=0) matrix multiply-accumulate encodings. OP-V (0x57)
# opcode; the trailing v0.scale operand encodes as vm=0 and selects the MX
# form reading paired E8M0 block scales from v0 (block size from vtype.bs).
#
#   FP scaled forms (OPFVV, same mnemonics as the unscaled vm=1 forms):
#     vfwmmacc.vv   funct6 = 0x15   vfqmmacc.vv  funct6 = 0x16
#     vf8wmmacc.vv  funct6 = 0x17
#   Integer-input FP-accumulate forms (OPIVV, MX-only mnemonics — the vm=1
#   encodings of these funct6 values are vwmmacc/vqmmacc/v8wmmacc):
#     vfwimmacc.vv  funct6 = 0x39   vfqimmacc.vv funct6 = 0x3a
#     vf8wimmacc.vv funct6 = 0x3b
#   vm=0 of vmmacc.vv (0x38) and vfmmacc.vv (0x14) is reserved.

vfwmmacc.vv v8, v4, v20, v0.scale
# CHECK-INST: vfwmmacc.vv v8, v4, v20, v0.scale
# CHECK-ENCODING: [0x57,0x14,0x42,0x55]
# CHECK-UNKNOWN: 55421457 <unknown>

vfqmmacc.vv v8, v4, v20, v0.scale
# CHECK-INST: vfqmmacc.vv v8, v4, v20, v0.scale
# CHECK-ENCODING: [0x57,0x14,0x42,0x59]
# CHECK-UNKNOWN: 59421457 <unknown>

vf8wmmacc.vv v8, v4, v20, v0.scale
# CHECK-INST: vf8wmmacc.vv v8, v4, v20, v0.scale
# CHECK-ENCODING: [0x57,0x14,0x42,0x5d]
# CHECK-UNKNOWN: 5d421457 <unknown>

# The unscaled (vm=1) form of the same mnemonic still assembles alongside.
vfwmmacc.vv v8, v4, v20
# CHECK-INST: vfwmmacc.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x42,0x57]
# CHECK-UNKNOWN: 57421457 <unknown>

vfwimmacc.vv v8, v4, v20, v0.scale
# CHECK-INST: vfwimmacc.vv v8, v4, v20, v0.scale
# CHECK-ENCODING: [0x57,0x04,0x42,0xe5]
# CHECK-UNKNOWN: e5420457 <unknown>

vfqimmacc.vv v8, v4, v20, v0.scale
# CHECK-INST: vfqimmacc.vv v8, v4, v20, v0.scale
# CHECK-ENCODING: [0x57,0x04,0x42,0xe9]
# CHECK-UNKNOWN: e9420457 <unknown>

vf8wimmacc.vv v8, v4, v20, v0.scale
# CHECK-INST: vf8wimmacc.vv v8, v4, v20, v0.scale
# CHECK-ENCODING: [0x57,0x04,0x42,0xed]
# CHECK-UNKNOWN: ed420457 <unknown>
