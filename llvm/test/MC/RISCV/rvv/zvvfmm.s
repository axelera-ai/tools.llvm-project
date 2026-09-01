# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-zvvfmm %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-zvvfmm %s \
# RUN:        | llvm-objdump -d --mattr=+experimental-zvvfmm --no-print-imm-hex - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-zvvfmm %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

# Encodings: OP-V (0x57) opcode, OPFVV (funct3=1), vm=1, funct6 selects
# mnemonic. Per the IME spec wavedrom blocks:
#   vfmmacc.vv    funct6 = 0x14 (0b010100)
#   vfwmmacc.vv   funct6 = 0x15 (0b010101)
#   vfqmmacc.vv   funct6 = 0x16 (0b010110)
#   vf8wmmacc.vv  funct6 = 0x17 (0b010111)
# Operand layout: vd, vs1, vs2 in asm (note: in encoding bits[24:20]=vs2,
# bits[19:15]=vs1 -- the asm order differs from the bit-field order, matching
# the VMACVV convention shared with the integer Mm-acc instructions).

vfmmacc.vv v8, v4, v20
# CHECK-INST: vfmmacc.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x42,0x53]
# CHECK-UNKNOWN: 53421457 <unknown>

vfwmmacc.vv v8, v4, v20
# CHECK-INST: vfwmmacc.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x42,0x57]
# CHECK-UNKNOWN: 57421457 <unknown>

vfqmmacc.vv v8, v4, v20
# CHECK-INST: vfqmmacc.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x42,0x5b]
# CHECK-UNKNOWN: 5b421457 <unknown>

vf8wmmacc.vv v8, v4, v20
# CHECK-INST: vf8wmmacc.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x42,0x5f]
# CHECK-UNKNOWN: 5f421457 <unknown>

# vm=0 microscaled forms of the widening FP MACs (trailing v0.scale; the
# encodings differ from the vm=1 forms above only in bit 25). Gated on the
# Zvvfmm family flag; the MX encoding-map legality is a vtype-time property.
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
