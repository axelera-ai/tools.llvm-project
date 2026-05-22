//===-- RISCVInstrInfoZvvm.td - RISC-V IME instructions ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V Integrated Matrix Extension (Zvvm family).
// Covers Zvvmm (integer matrix multiply-accumulate), Zvvfmm (floating-point
// matrix multiply-accumulate), Zvvmtls (order-preserving tile load/store),
// and Zvvmttls (transposing tile load/store).
//
// Caller responsibility: vtype must be configured (SEW, LMUL, lambda,
// altfmt_A/altfmt_B) via vsetvl prior to invoking these instructions. The
// RISCVInsertVSETVLI pass is NOT taught about lambda/altfmt_A/altfmt_B yet, so
// it will not place a vsetvli on behalf of these instructions.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Real instruction class (encoded form, used by the assembler / disassembler).
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
// op vd, vs1, vs2
//   vd is read-modified-written (tied operand).
//   vm is hardcoded to 1: for integer Mm-acc the encoding requires it; for
//   FP Mm-acc the spec uses vm=0 to switch to microscaled MX* operation,
//   which Phase 1 does not expose.
// Encoding (OP-V major opcode):
//   [31:26] = funct6, [25] = 1, [24:20] = vs2, [19:15] = vs1,
//   [14:12] = OPIVV (000) for integer / OPFVV (001) for FP, [11:7] = vd,
//   [6:0] = OP-V (0b1010111).
class RVInstVMatrixMACOp<bits<6> funct6, RISCVVFormat opv, string opcodestr>
    : RVInstVV<funct6, opv, (outs VR:$vd_wb),
               (ins VR:$vd, VR:$vs1, VR:$vs2),
               opcodestr, "$vd, $vs1, $vs2"> {
  let vm = 1;
  let Constraints = "$vd = $vd_wb";
}

class RVInstVMatrixMAC<bits<6> funct6, string opcodestr>
    : RVInstVMatrixMACOp<funct6, OPIVV, opcodestr>;

class RVInstVFMatrixMAC<bits<6> funct6, string opcodestr>
    : RVInstVMatrixMACOp<funct6, OPFVV, opcodestr>;
} // hasSideEffects = 0, mayLoad = 0, mayStore = 0

//===----------------------------------------------------------------------===//
// Zvvmm: integer matrix multiply-accumulate instructions.
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZvvmm] in {
def VMMACC_VV   : RVInstVMatrixMAC<0b111000, "vmmacc.vv">;
def VWMMACC_VV  : RVInstVMatrixMAC<0b111001, "vwmmacc.vv">;
def VQMMACC_VV  : RVInstVMatrixMAC<0b111010, "vqmmacc.vv">;
def V8WMMACC_VV : RVInstVMatrixMAC<0b111011, "v8wmmacc.vv">;
} // Predicates = [HasStdExtZvvmm]

//===----------------------------------------------------------------------===//
// Pseudo instructions used by ISel.
//
// Each pseudo carries an extra GPRNoX0:$vl operand from the intrinsic so the
// .td pattern can consume it. The PseudoInstExpansion below rewrites the
// pseudo to the real instruction at MC-lowering time, discarding $vl (the
// real instruction reads VL implicitly from the vl CSR).
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0,
    Uses = [VL, VTYPE], Constraints = "$vd = $vd_wb" in
class PseudoVMatrixMAC<Instruction realInst>
    : Pseudo<(outs VR:$vd_wb),
             (ins VR:$vd, VR:$vs1, VR:$vs2, GPRNoX0:$vl), []>,
      PseudoInstExpansion<(realInst VR:$vd_wb, VR:$vd, VR:$vs1, VR:$vs2)>;

let Predicates = [HasStdExtZvvmm] in {
def PseudoVMMACC_VV   : PseudoVMatrixMAC<VMMACC_VV>;
def PseudoVWMMACC_VV  : PseudoVMatrixMAC<VWMMACC_VV>;
def PseudoVQMMACC_VV  : PseudoVMatrixMAC<VQMMACC_VV>;
def PseudoV8WMMACC_VV : PseudoVMatrixMAC<V8WMMACC_VV>;
} // Predicates = [HasStdExtZvvmm]

//===----------------------------------------------------------------------===//
// ISel patterns: lower intrinsics to pseudos.
//
// Phase 1 emits one pattern per intrinsic for a single representative
// (SEW, LMUL) combination. Additional (SEW, LMUL) coverage is added when we
// scale the multiclass machinery in a later phase.
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZvvmm] in {
// W=1: SEW(C) = SEW(A) = SEW(B) = 32, EMUL_C = LMUL = 1.
def : Pat<(nxv2i32 (int_riscv_vmmacc (nxv2i32 VR:$vd),
                                     (nxv2i32 VR:$vs1),
                                     (nxv2i32 VR:$vs2),
                                     (XLenVT GPRNoX0:$vl))),
          (PseudoVMMACC_VV VR:$vd, VR:$vs1, VR:$vs2, GPRNoX0:$vl)>;

// W=2: SEW(C) = 32, EEW(A) = EEW(B) = 16, EMUL_C = LMUL = 1.
def : Pat<(nxv2i32 (int_riscv_vwmmacc (nxv2i32 VR:$vd),
                                      (nxv4i16 VR:$vs1),
                                      (nxv4i16 VR:$vs2),
                                      (XLenVT GPRNoX0:$vl))),
          (PseudoVWMMACC_VV VR:$vd, VR:$vs1, VR:$vs2, GPRNoX0:$vl)>;

// W=4: SEW(C) = 32, EEW(A) = EEW(B) = 8, EMUL_C = LMUL = 1.
def : Pat<(nxv2i32 (int_riscv_vqmmacc (nxv2i32 VR:$vd),
                                      (nxv8i8 VR:$vs1),
                                      (nxv8i8 VR:$vs2),
                                      (XLenVT GPRNoX0:$vl))),
          (PseudoVQMMACC_VV VR:$vd, VR:$vs1, VR:$vs2, GPRNoX0:$vl)>;

// W=8: SEW(C) = 64, EEW(A) = EEW(B) = 8, EMUL_C = LMUL = 1.
def : Pat<(nxv1i64 (int_riscv_v8wmmacc (nxv1i64 VR:$vd),
                                       (nxv8i8 VR:$vs1),
                                       (nxv8i8 VR:$vs2),
                                       (XLenVT GPRNoX0:$vl))),
          (PseudoV8WMMACC_VV VR:$vd, VR:$vs1, VR:$vs2, GPRNoX0:$vl)>;
} // Predicates = [HasStdExtZvvmm]

//===----------------------------------------------------------------------===//
// Zvvfmm: floating-point matrix multiply-accumulate.
//
// Encoding bits per the IME spec's per-instruction wavedrom blocks:
//   vfmmacc.vv    funct6 = 0b010100 (0x14)
//   vfwmmacc.vv   funct6 = 0b010101 (0x15)
//   vfqmmacc.vv   funct6 = 0b010110 (0x16)
//   vf8wmmacc.vv  funct6 = 0b010111 (0x17)
// All share funct3 = OPFVV (0b001) and have vm hardcoded to 1 (Phase 1 does
// not encode the vm=0 microscaled MX* form).
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZvvfmm] in {
def VFMMACC_VV   : RVInstVFMatrixMAC<0b010100, "vfmmacc.vv">;
def VFWMMACC_VV  : RVInstVFMatrixMAC<0b010101, "vfwmmacc.vv">;
def VFQMMACC_VV  : RVInstVFMatrixMAC<0b010110, "vfqmmacc.vv">;
def VF8WMMACC_VV : RVInstVFMatrixMAC<0b010111, "vf8wmmacc.vv">;
} // Predicates = [HasStdExtZvvfmm]

let Predicates = [HasStdExtZvvfmm] in {
def PseudoVFMMACC_VV   : PseudoVMatrixMAC<VFMMACC_VV>;
def PseudoVFWMMACC_VV  : PseudoVMatrixMAC<VFWMMACC_VV>;
def PseudoVFQMMACC_VV  : PseudoVMatrixMAC<VFQMMACC_VV>;
def PseudoVF8WMMACC_VV : PseudoVMatrixMAC<VF8WMMACC_VV>;
} // Predicates = [HasStdExtZvvfmm]

// ISel patterns for the FP MACs. Phase 1 lowers each intrinsic for one
// representative (SEW, EMUL_C=LMUL=1) shape. Broader (SEW, LMUL) coverage —
// and the architecturally-trickier EMUL_C ≠ LMUL coverage — lands alongside
// the integer-side widening builtins in a follow-up.
//
// Type choices below mirror the integer side's container convention: when the
// IR has no first-class FP type for an element width (OFP8 / E4M3 / E5M2), we
// use the integer container (nxv8i8) and rely on vtype.altfmt to select the
// FP8 interpretation at execution time.
let Predicates = [HasStdExtZvvfmm] in {
// W=1: SEW(C) = SEW(A) = SEW(B) = 32, FP32 accumulator and FP32 inputs.
def : Pat<(nxv2f32 (int_riscv_vfmmacc (nxv2f32 VR:$vd),
                                      (nxv2f32 VR:$vs1),
                                      (nxv2f32 VR:$vs2),
                                      (XLenVT GPRNoX0:$vl))),
          (PseudoVFMMACC_VV VR:$vd, VR:$vs1, VR:$vs2, GPRNoX0:$vl)>;

// W=2: SEW(C) = 32, EEW(A) = EEW(B) = 16, FP16 inputs to FP32 accumulator.
// BF16 input (altfmt_A/B = 1) reuses the same IR type at LMUL=1; format
// disambiguation happens at vsetvl-time, so no additional ISel pattern is
// required.
def : Pat<(nxv2f32 (int_riscv_vfwmmacc (nxv2f32 VR:$vd),
                                       (nxv4f16 VR:$vs1),
                                       (nxv4f16 VR:$vs2),
                                       (XLenVT GPRNoX0:$vl))),
          (PseudoVFWMMACC_VV VR:$vd, VR:$vs1, VR:$vs2, GPRNoX0:$vl)>;

// W=4: SEW(C) = 32, EEW(A) = EEW(B) = 8, OFP8 inputs to FP32 accumulator.
// OFP8 has no first-class IR FP type — use the integer container (nxv8i8).
def : Pat<(nxv2f32 (int_riscv_vfqmmacc (nxv2f32 VR:$vd),
                                       (nxv8i8 VR:$vs1),
                                       (nxv8i8 VR:$vs2),
                                       (XLenVT GPRNoX0:$vl))),
          (PseudoVFQMMACC_VV VR:$vd, VR:$vs1, VR:$vs2, GPRNoX0:$vl)>;

// W=8: SEW(C) = 64, EEW(A) = EEW(B) = 8, OFP8 inputs to FP64 accumulator.
def : Pat<(nxv1f64 (int_riscv_vf8wmmacc (nxv1f64 VR:$vd),
                                        (nxv8i8 VR:$vs1),
                                        (nxv8i8 VR:$vs2),
                                        (XLenVT GPRNoX0:$vl))),
          (PseudoVF8WMMACC_VV VR:$vd, VR:$vs1, VR:$vs2, GPRNoX0:$vl)>;
} // Predicates = [HasStdExtZvvfmm]

//===----------------------------------------------------------------------===//
// Pseudo for the register-form `vsetvl` with a full XLen vtype immediate.
//
// Emitted by:
//   - selectVSETVLMatrix (for the int_riscv_vsetvl_matrix intrinsic).
//   - RISCVInsertVSETVLI (when it needs to write a vtype that includes
//     non-default matrix fields — the standard PseudoVSETVLI immediate
//     cannot express the high-end matrix bits).
//
// Expanded by RISCVExpandPseudoInsts to:
//   movImm $rd, $vtype          ; materialize the constant (LUI/ADDI/...)
//   vsetvl  $rd, $rs1, $rd      ; rs2 is read before rd is written, so $rd
//                               ; can double as scratch.
//===----------------------------------------------------------------------===//

let hasSideEffects = 1, mayLoad = 0, mayStore = 0, Defs = [VL, VTYPE],
    // $rd is used as the scratch register that holds the materialized vtype
    // constant before being overwritten with the new VL — see
    // expandPseudoVSETVLMatrix. earlyclobber prevents the register allocator
    // from coalescing $rd with $rs1, which would otherwise lose the AVL
    // value when the LI sequence writes $rd.
    Constraints = "@earlyclobber $rd" in
def PseudoVSETVL_MATRIX
    : Pseudo<(outs GPR:$rd), (ins GPRNoX0:$rs1, ixlenimm_li:$vtype), []>,
      Sched<[WriteVSETVL, ReadVSETVL, ReadVSETVL]>;

//===----------------------------------------------------------------------===//
// Targeted lambda primitives.
//
// PseudoVSETLAMBDA: read-modify-write `vtype.lambda`. Expanded to a sequence
// that reads current vtype, masks out the lambda bits, ORs in the new
// encoding, reads current vl, emits a register-form `vsetvl` to commit the
// updated vtype while preserving vl, and reads back the post-WARL-clamp
// lambda field into $rd. The `$scratch` output exists only to give the
// register allocator a second scratch register for the read-modify-write
// sequence — its value is dead.
//
// PseudoQUERYLAMBDA: pure read of `vtype`. Extracts `vtype.lambda[2:0]` into
// $rd. No vtype write, no scratch required.
//===----------------------------------------------------------------------===//

let hasSideEffects = 1, mayLoad = 0, mayStore = 0, Defs = [VL, VTYPE],
    Constraints = "@earlyclobber $rd,@earlyclobber $scratch" in
def PseudoVSETLAMBDA
    : Pseudo<(outs GPR:$rd, GPR:$scratch), (ins ixlenimm_li:$encoding), []>,
      Sched<[WriteVSETVL, ReadVSETVL]>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, Uses = [VTYPE] in
def PseudoQUERYLAMBDA
    : Pseudo<(outs GPR:$rd), (ins), []>;

// ISel for int_riscv_vsetlambda / int_riscv_query_lambda is performed
// manually in RISCVDAGToDAGISel::selectVSETLAMBDA / selectQUERYLAMBDA — the
// multi-output PseudoVSETLAMBDA can't be expressed cleanly as a td pattern
// because tblgen cannot infer the scratch output's type independent of $rd.

//===----------------------------------------------------------------------===//
// Tile load/store (Zvvmtls / Zvvmttls).
//
// Encoding (reuses the RVV LD/ST major-opcode space):
//   [31:29] = lambda  (3-bit override; 0b000 = use vtype.lambda, 0b001 = L1,
//                      0b010 = L2, 0b011 = L4, 0b100 = L8, 0b101 = L16,
//                      0b110 = L32, 0b111 = L64)
//   [28]    = 1       (MEW: tile marker)
//   [27:26] = mop     (0b00 = order-preserving, 0b01 = transposing)
//   [25]    = vm
//   [24:20] = rs2     (LD stride in elements; GPR)
//   [19:15] = rs1     (base address; GPR)
//   [14:12] = 0b111   (width)
//   [11:7]  = vd / vs3
//   [6:0]   = 0x07 (LOAD-FP) for loads, 0x27 (STORE-FP) for stores
//
// The Lλ operand is exposed as a single `lambdaarg` immediate (parsed as
// "L1" / "L2" / "L4" / "L8" / "L16" / "L32" / "L64"; omitting it defaults to
// 0 = use vtype.lambda). All eight lambda variants share one Instruction def
// per (mnemonic, masked-or-not), encoded via the operand value.
//===----------------------------------------------------------------------===//

def LambdaArg : AsmOperandClass {
  let Name = "LambdaArg";
  let RenderMethod = "addLambdaArgOperands";
  let ParserMethod = "parseLambdaArg";
  let DiagnosticType = "InvalidLambdaArg";
  let DiagnosticString = "operand must be one of L1, L2, L4, L8, L16, L32, L64";
}

def lambdaarg : Operand<XLenVT> {
  let ParserMatchClass = LambdaArg;
  let PrintMethod = "printLambdaArg";
  let DecoderMethod = "decodeLambdaArg";
  let OperandType = "OPERAND_UIMM3";
  let OperandNamespace = "RISCVOp";
}

// Loose-matching VMaskOp variant used only by InstAliases: returns NoMatch
// (rather than Error) when the next token isn't "v0.t", so the asm matcher
// can backtrack from a no-Lλ alias to the canonical (explicit-Lλ) form.
def LooseVMaskAsmOperand : AsmOperandClass {
  let Name = "LooseVMaskReg";
  let RenderMethod = "addRegOperands";
  let PredicateMethod = "isV0Reg";
  let ParserMethod = "parseLooseMaskReg";
  let IsOptional = 1;
  let DefaultMethod = "defaultMaskRegOp";
  let DiagnosticType = "InvalidVMaskRegister";
}

def LooseVMaskOp : RegisterOperand<VMV0> {
  let ParserMatchClass = LooseVMaskAsmOperand;
  let PrintMethod = "printVMaskReg";
  let EncoderMethod = "getVMaskReg";
  let DecoderMethod = "decodeVMaskReg";
  let OperandNamespace = "RISCVOp";
  let OperandType = "OPERAND_VMASK";
}

// One real tile-load / tile-store class. Lambda is mandatory on the canonical
// form (the asm matcher cannot fall through two consecutive IsOptional
// operands when the first NoMatches on the next token). The VMaskOp operand
// is parsed optionally from ", v0.t". The no-Lλ input form is supplied by
// InstAliases (using LooseVMaskOp so the matcher can backtrack to the
// canonical form when an explicit Lλ is what follows rs2).
let hasSideEffects = 0, mayLoad = 1, mayStore = 0, Uses = [VL, VTYPE],
    RVVConstraint = VMConstraint in
class TileLoad<bits<2> mop, string opcodestr>
    : RVInst<(outs VR:$vd),
             (ins GPRMemZeroOffset:$rs1, GPR:$rs2, lambdaarg:$lambda,
                  VMaskOp:$vm),
             opcodestr, "$vd, ${rs1}, $rs2${lambda}$vm", [], InstFormatR> {
  bits<3> lambda;
  bits<5> rs2;
  bits<5> rs1;
  bits<5> vd;
  bit vm;

  let Inst{31-29} = lambda;
  let Inst{28} = 1;
  let Inst{27-26} = mop;
  let Inst{25} = vm;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = 0b111;
  let Inst{11-7} = vd;
  let Inst{6-0} = OPC_LOAD_FP.Value;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 1, Uses = [VL, VTYPE] in
class TileStore<bits<2> mop, string opcodestr>
    : RVInst<(outs),
             (ins VR:$vs3, GPRMemZeroOffset:$rs1, GPR:$rs2, lambdaarg:$lambda,
                  VMaskOp:$vm),
             opcodestr, "$vs3, ${rs1}, $rs2${lambda}$vm", [], InstFormatR> {
  bits<3> lambda;
  bits<5> rs2;
  bits<5> rs1;
  bits<5> vs3;
  bit vm;

  let Inst{31-29} = lambda;
  let Inst{28} = 1;
  let Inst{27-26} = mop;
  let Inst{25} = vm;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = 0b111;
  let Inst{11-7} = vs3;
  let Inst{6-0} = OPC_STORE_FP.Value;
}

let Predicates = [HasStdExtZvvmtls] in {
  def VMTL_V : TileLoad <0b00, "vmtl.v">;
  def VMTS_V : TileStore<0b00, "vmts.v">;
}
let Predicates = [HasStdExtZvvmttls] in {
  def VMTTL_V : TileLoad <0b01, "vmttl.v">;
  def VMTTS_V : TileStore<0b01, "vmtts.v">;
}

// InstAliases for the no-Lλ input syntax. LooseVMaskOp absorbs an optional
// ", v0.t" without erroring when the next token is something else (e.g. an
// explicit Lλ identifier), letting the matcher fall through to the canonical
// form.
let Predicates = [HasStdExtZvvmtls] in {
  def : InstAlias<"vmtl.v $vd, ${rs1}, $rs2$vm",
                  (VMTL_V VR:$vd, GPRMemZeroOffset:$rs1, GPR:$rs2,
                          0, LooseVMaskOp:$vm)>;
  def : InstAlias<"vmts.v $vs3, ${rs1}, $rs2$vm",
                  (VMTS_V VR:$vs3, GPRMemZeroOffset:$rs1, GPR:$rs2,
                          0, LooseVMaskOp:$vm)>;
}
let Predicates = [HasStdExtZvvmttls] in {
  def : InstAlias<"vmttl.v $vd, ${rs1}, $rs2$vm",
                  (VMTTL_V VR:$vd, GPRMemZeroOffset:$rs1, GPR:$rs2,
                           0, LooseVMaskOp:$vm)>;
  def : InstAlias<"vmtts.v $vs3, ${rs1}, $rs2$vm",
                  (VMTTS_V VR:$vs3, GPRMemZeroOffset:$rs1, GPR:$rs2,
                           0, LooseVMaskOp:$vm)>;
}

//===----------------------------------------------------------------------===//
// Pseudo instructions for tile load/store.
//
// Each pseudo carries the vl operand from the intrinsic. The masked variants
// additionally carry a VMV0:$mask (forced to v0). PseudoInstExpansion rewrites
// the pseudo to the real instruction with a literal vm bit (1 = no mask,
// 0 = masked).
//===----------------------------------------------------------------------===//

// Pseudos carry the vl (and mask) operands the IR intrinsic supplies and
// expand to the matching real instruction. Unmasked pseudos pass zero_reg as
// the VMaskOp value (vm = 1); masked pseudos pass V0 (vm = 0).
class PseudoVMatrixTileLoad<Instruction realInst>
    : Pseudo<(outs VR:$vd),
             (ins VR:$passthru, GPRMemZeroOffset:$rs1, GPR:$rs2,
                  lambdaarg:$lambda, GPRNoX0:$vl), []>,
      PseudoInstExpansion<(realInst VR:$vd, GPRMemZeroOffset:$rs1, GPR:$rs2,
                                    lambdaarg:$lambda, zero_reg)> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Uses = [VL, VTYPE];
  let Constraints = "$vd = $passthru";
}

class PseudoVMatrixTileLoadMask<Instruction realInst>
    : Pseudo<(outs VR:$vd),
             (ins VR:$passthru, GPRMemZeroOffset:$rs1, GPR:$rs2,
                  lambdaarg:$lambda, VMV0:$mask, GPRNoX0:$vl,
                  vec_policy:$policy), []>,
      PseudoInstExpansion<(realInst VR:$vd, GPRMemZeroOffset:$rs1, GPR:$rs2,
                                    lambdaarg:$lambda, V0)> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Uses = [VL, VTYPE, V0];
  let Constraints = "$vd = $passthru";
}

class PseudoVMatrixTileStore<Instruction realInst>
    : Pseudo<(outs),
             (ins VR:$vs3, GPRMemZeroOffset:$rs1, GPR:$rs2,
                  lambdaarg:$lambda, GPRNoX0:$vl), []>,
      PseudoInstExpansion<(realInst VR:$vs3, GPRMemZeroOffset:$rs1, GPR:$rs2,
                                    lambdaarg:$lambda, zero_reg)> {
  let mayLoad = 0;
  let mayStore = 1;
  let hasSideEffects = 0;
  let Uses = [VL, VTYPE];
}

class PseudoVMatrixTileStoreMask<Instruction realInst>
    : Pseudo<(outs),
             (ins VR:$vs3, GPRMemZeroOffset:$rs1, GPR:$rs2,
                  lambdaarg:$lambda, VMV0:$mask, GPRNoX0:$vl), []>,
      PseudoInstExpansion<(realInst VR:$vs3, GPRMemZeroOffset:$rs1, GPR:$rs2,
                                    lambdaarg:$lambda, V0)> {
  let mayLoad = 0;
  let mayStore = 1;
  let hasSideEffects = 0;
  let Uses = [VL, VTYPE, V0];
}

let Predicates = [HasStdExtZvvmtls] in {
  def PseudoVMTL_V       : PseudoVMatrixTileLoad<VMTL_V>;
  def PseudoVMTL_V_MASK  : PseudoVMatrixTileLoadMask<VMTL_V>;
  def PseudoVMTS_V       : PseudoVMatrixTileStore<VMTS_V>;
  def PseudoVMTS_V_MASK  : PseudoVMatrixTileStoreMask<VMTS_V>;
}
let Predicates = [HasStdExtZvvmttls] in {
  def PseudoVMTTL_V      : PseudoVMatrixTileLoad<VMTTL_V>;
  def PseudoVMTTL_V_MASK : PseudoVMatrixTileLoadMask<VMTTL_V>;
  def PseudoVMTTS_V      : PseudoVMatrixTileStore<VMTTS_V>;
  def PseudoVMTTS_V_MASK : PseudoVMatrixTileStoreMask<VMTTS_V>;
}

//===----------------------------------------------------------------------===//
// ISel patterns: intrinsic -> pseudo.
//
// The tile load/store instructions are element-type-agnostic at the encoding
// layer — SEW comes from vtype.vsew and the only thing the IR vector type
// affects in the backend is register-class selection (everything is in VR at
// LMUL=1). We therefore emit one pattern per intrinsic per LMUL=1 element
// type, for both the order-preserving (vmtl/vmts) and transposing (vmttl/
// vmtts) mnemonics.
//===----------------------------------------------------------------------===//

// LMUL=1 vector types the tile load/store builtins expose to Clang. The
// corresponding mask types come from `vti.Mask` so the same class works for
// every element width.
defvar AllZvvmTileLSM1VTIs =
    [VI8M1, VI16M1, VI32M1, VI64M1, VF16M1, VF32M1, VF64M1];

class TileLoadPat<Intrinsic intr, Instruction pseudo, int lambdaVal,
                  VTypeInfo vti>
    : Pat<(vti.Vector (intr (vti.Vector VR:$passthru),
                            GPR:$rs1,
                            (XLenVT GPR:$rs2),
                            (XLenVT GPRNoX0:$vl))),
          (pseudo VR:$passthru, GPR:$rs1, GPR:$rs2, lambdaVal, GPRNoX0:$vl)>;

class TileLoadMaskPat<Intrinsic intr, Instruction pseudo, int lambdaVal,
                      VTypeInfo vti>
    : Pat<(vti.Vector (intr (vti.Vector VR:$passthru),
                            GPR:$rs1,
                            (XLenVT GPR:$rs2),
                            (vti.Mask VMV0:$mask),
                            (XLenVT GPRNoX0:$vl),
                            (XLenVT timm:$policy))),
          (pseudo VR:$passthru, GPR:$rs1, GPR:$rs2, lambdaVal,
                  VMV0:$mask, GPRNoX0:$vl, (XLenVT timm:$policy))>;

class TileStorePat<Intrinsic intr, Instruction pseudo, int lambdaVal,
                   VTypeInfo vti>
    : Pat<(intr (vti.Vector VR:$vs3),
                GPR:$rs1,
                (XLenVT GPR:$rs2),
                (XLenVT GPRNoX0:$vl)),
          (pseudo VR:$vs3, GPR:$rs1, GPR:$rs2, lambdaVal, GPRNoX0:$vl)>;

class TileStoreMaskPat<Intrinsic intr, Instruction pseudo, int lambdaVal,
                       VTypeInfo vti>
    : Pat<(intr (vti.Vector VR:$vs3),
                GPR:$rs1,
                (XLenVT GPR:$rs2),
                (vti.Mask VMV0:$mask),
                (XLenVT GPRNoX0:$vl)),
          (pseudo VR:$vs3, GPR:$rs1, GPR:$rs2, lambdaVal,
                  VMV0:$mask, GPRNoX0:$vl)>;

// Each per-lambda intrinsic lowers to the same pseudo with a different literal
// lambda value in the operand slot. Lambda encoding:
//   "" -> 0 (use vtype.lambda), "_l1" -> 1, "_l2" -> 2, "_l4" -> 3,
//   "_l8" -> 4, "_l16" -> 5, "_l32" -> 6, "_l64" -> 7.
foreach lsuf = ["", "_l1", "_l2", "_l4", "_l8", "_l16", "_l32", "_l64"] in {
  defvar lval = !cond(!eq(lsuf, ""):    0,
                      !eq(lsuf, "_l1"): 1,
                      !eq(lsuf, "_l2"): 2,
                      !eq(lsuf, "_l4"): 3,
                      !eq(lsuf, "_l8"): 4,
                      !eq(lsuf, "_l16"): 5,
                      !eq(lsuf, "_l32"): 6,
                      !eq(lsuf, "_l64"): 7);

  // FP element types pick up the matching scalar-FP+Zve* predicates from
  // GetVTypeMinimalPredicates so the pattern is only available when the IR
  // container type itself is legal.
  foreach vti = AllZvvmTileLSM1VTIs in {
    let Predicates = !listconcat([HasStdExtZvvmtls],
                                 GetVTypeMinimalPredicates<vti>.Predicates) in {
      def : TileLoadPat<
          !cast<Intrinsic>("int_riscv_vmtl"#lsuf), PseudoVMTL_V, lval, vti>;
      def : TileLoadMaskPat<
          !cast<Intrinsic>("int_riscv_vmtl"#lsuf#"_mask"),
          PseudoVMTL_V_MASK, lval, vti>;
      def : TileStorePat<
          !cast<Intrinsic>("int_riscv_vmts"#lsuf), PseudoVMTS_V, lval, vti>;
      def : TileStoreMaskPat<
          !cast<Intrinsic>("int_riscv_vmts"#lsuf#"_mask"),
          PseudoVMTS_V_MASK, lval, vti>;
    }
    let Predicates = !listconcat([HasStdExtZvvmttls],
                                 GetVTypeMinimalPredicates<vti>.Predicates) in {
      def : TileLoadPat<
          !cast<Intrinsic>("int_riscv_vmttl"#lsuf), PseudoVMTTL_V, lval, vti>;
      def : TileLoadMaskPat<
          !cast<Intrinsic>("int_riscv_vmttl"#lsuf#"_mask"),
          PseudoVMTTL_V_MASK, lval, vti>;
      def : TileStorePat<
          !cast<Intrinsic>("int_riscv_vmtts"#lsuf), PseudoVMTTS_V, lval, vti>;
      def : TileStoreMaskPat<
          !cast<Intrinsic>("int_riscv_vmtts"#lsuf#"_mask"),
          PseudoVMTTS_V_MASK, lval, vti>;
    }
  }
}
