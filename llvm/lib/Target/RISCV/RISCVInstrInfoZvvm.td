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
//
// Matrix multiply-accumulate has two independent register-group axes:
//   * vd  uses EMUL_C = VLEN / (SEW(C) * lambda^2), independent of LMUL.
//   * vs1 / vs2 use vtype.LMUL.
// PseudoVMatrixMACPair is parameterized over both; PseudoVMatrixMAC is the
// degenerate single-LMUL form kept for the widening MACs that still ride on
// the EMUL_C = LMUL = 1 representative pattern (deferred follow-up).
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0,
    Uses = [VL, VTYPE], Constraints = "$vd = $vd_wb" in
class PseudoVMatrixMAC<Instruction realInst>
    : Pseudo<(outs VR:$vd_wb),
             (ins VR:$vd, VR:$vs1, VR:$vs2, GPRNoX0:$vl), []>,
      PseudoInstExpansion<(realInst VR:$vd_wb, VR:$vd, VR:$vs1, VR:$vs2)>;

// Marker base class for all Zvvm matrix pseudos that need MC lowering via the
// dedicated helper (lowerRISCVMatrixPseudo in RISCVAsmPrinter.cpp). The helper
// looks up the entry in RISCVMatrixPseudosTable, dispatches on `Kind`, and
// handles VRM{2,4,8} → VR subregister extraction since the single real
// instruction always has VR operands. PseudoInstExpansion is unsuitable
// because it requires strict operand-type matching at tablegen time.
//
// We don't inherit from RISCVVPseudo deliberately: matrix instructions
// configure their own vtype via vsetvl_matrix (including the matrix-only
// fields lambda / bs / altfmt_A / altfmt_B), and we don't want
// RISCVInsertVSETVLI to inject a standard vsetvli ahead of them.
//
// Kind values:
//   0 = MAC                  — pseudo (vd_wb, vd, vs1, vs2, vl) → real (vd_wb, vd, vs1, vs2)
//   1 = Tile load (unmasked) — pseudo (vd, passthru, rs1, rs2, lambda, vl)
//                              → real (vd, rs1, rs2, lambda, zero_reg)
//   2 = Tile load masked     — pseudo (vd, passthru, rs1, rs2, lambda, V0, vl, policy)
//                              → real (vd, rs1, rs2, lambda, V0)
//   3 = Tile store unmasked  — pseudo (vs3, rs1, rs2, lambda, vl)
//                              → real (vs3, rs1, rs2, lambda, zero_reg)
//   4 = Tile store masked    — pseudo (vs3, rs1, rs2, lambda, V0, vl)
//                              → real (vs3, rs1, rs2, lambda, V0)
class RISCVMatrixPseudo<Instruction realInst, int kind> {
  Pseudo Pseudo = !cast<Pseudo>(NAME);
  Instruction BaseInstr = realInst;
  bits<3> Kind = kind;
}

def RISCVMatrixPseudosTable : GenericTable {
  let FilterClass = "RISCVMatrixPseudo";
  let CppTypeName = "MatrixPseudoInfo";
  let Fields = [ "Pseudo", "BaseInstr", "Kind" ];
  let PrimaryKey = [ "Pseudo" ];
  let PrimaryKeyName = "getMatrixPseudoInfo";
  let PrimaryKeyEarlyOut = true;
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0,
    Uses = [VL, VTYPE], Constraints = "$vd = $vd_wb" in
class PseudoVMatrixMACPair<Instruction realInst,
                           VTypeInfo cti, VTypeInfo abti>
    : Pseudo<(outs cti.RegClass:$vd_wb),
             (ins cti.RegClass:$vd, abti.RegClass:$vs1,
                  abti.RegClass:$vs2, GPRNoX0:$vl), []>,
      RISCVMatrixPseudo<realInst, /*MAC=*/0>;

// VTI lists for the non-widening MAC (EMUL_C, LMUL) cross-product.
// EMUL_C ∈ {1,2,4,8} (spec also permits 16; capped at 8 because upstream RVV
// has no VRM16 register class). LMUL ∈ {1,2,4,8}; fractional LMUL is spec-
// reserved for the IME family.
defvar AllZvvmMACVTIs_I32 = [VI32M1, VI32M2, VI32M4, VI32M8];
defvar AllZvvmMACVTIs_F32 = [VF32M1, VF32M2, VF32M4, VF32M8];

let Predicates = [HasStdExtZvvmm] in {
// Pseudo name: PseudoVMMACC_VV_M{EMUL_C}_M{LMUL} (in that order).
foreach cti = AllZvvmMACVTIs_I32 in
  foreach abti = AllZvvmMACVTIs_I32 in
    def "PseudoVMMACC_VV_"#cti.LMul.MX#"_"#abti.LMul.MX
        : PseudoVMatrixMACPair<VMMACC_VV, cti, abti>;

def PseudoVWMMACC_VV  : PseudoVMatrixMAC<VWMMACC_VV>;
def PseudoVQMMACC_VV  : PseudoVMatrixMAC<VQMMACC_VV>;
def PseudoV8WMMACC_VV : PseudoVMatrixMAC<V8WMMACC_VV>;
} // Predicates = [HasStdExtZvvmm]

//===----------------------------------------------------------------------===//
// ISel patterns: lower intrinsics to pseudos.
//
// Non-widening MAC (vmmacc.vv): EMUL_C ∈ {1,2,4,8} × LMUL ∈ {1,2,4,8} cross
// product, since EMUL_C and LMUL are independent. The widening MACs still
// emit one representative pattern per intrinsic at EMUL_C = LMUL = 1; the
// (SEW(A) ≠ SEW(C), EMUL_C ≠ LMUL) coverage for them is deferred.
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZvvmm] in {
// W=1: SEW(C) = SEW(A) = SEW(B) = 32. Independent (EMUL_C, LMUL).
foreach cti = AllZvvmMACVTIs_I32 in
  foreach abti = AllZvvmMACVTIs_I32 in
    def : Pat<(cti.Vector (int_riscv_vmmacc (cti.Vector cti.RegClass:$vd),
                                            (abti.Vector abti.RegClass:$vs1),
                                            (abti.Vector abti.RegClass:$vs2),
                                            (XLenVT GPRNoX0:$vl))),
              (!cast<Instruction>("PseudoVMMACC_VV_"#cti.LMul.MX#"_"#abti.LMul.MX)
                  cti.RegClass:$vd, abti.RegClass:$vs1, abti.RegClass:$vs2,
                  GPRNoX0:$vl)>;

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
// Pseudo name: PseudoVFMMACC_VV_M{EMUL_C}_M{LMUL} (in that order).
foreach cti = AllZvvmMACVTIs_F32 in
  foreach abti = AllZvvmMACVTIs_F32 in
    def "PseudoVFMMACC_VV_"#cti.LMul.MX#"_"#abti.LMul.MX
        : PseudoVMatrixMACPair<VFMMACC_VV, cti, abti>;

def PseudoVFWMMACC_VV  : PseudoVMatrixMAC<VFWMMACC_VV>;
def PseudoVFQMMACC_VV  : PseudoVMatrixMAC<VFQMMACC_VV>;
def PseudoVF8WMMACC_VV : PseudoVMatrixMAC<VF8WMMACC_VV>;
} // Predicates = [HasStdExtZvvfmm]

// ISel patterns for the FP MACs. The non-widening vfmmacc.vv covers the
// independent (EMUL_C, LMUL) ∈ {1,2,4,8}^2 grid; the widening variants still
// emit one representative pattern per intrinsic at EMUL_C = LMUL = 1, since
// they additionally need varying SEW(A) ≠ SEW(C) coverage that ships in a
// follow-up alongside the integer-widening builtins.
//
// Type choices below mirror the integer side's container convention: when the
// IR has no first-class FP type for an element width (OFP8 / E4M3 / E5M2), we
// use the integer container (nxv8i8) and rely on vtype.altfmt to select the
// FP8 interpretation at execution time.
let Predicates = [HasStdExtZvvfmm] in {
// W=1: SEW(C) = SEW(A) = SEW(B) = 32. Independent (EMUL_C, LMUL).
foreach cti = AllZvvmMACVTIs_F32 in
  foreach abti = AllZvvmMACVTIs_F32 in
    def : Pat<(cti.Vector (int_riscv_vfmmacc (cti.Vector cti.RegClass:$vd),
                                             (abti.Vector abti.RegClass:$vs1),
                                             (abti.Vector abti.RegClass:$vs2),
                                             (XLenVT GPRNoX0:$vl))),
              (!cast<Instruction>("PseudoVFMMACC_VV_"#cti.LMul.MX#"_"#abti.LMul.MX)
                  cti.RegClass:$vd, abti.RegClass:$vs1, abti.RegClass:$vs2,
                  GPRNoX0:$vl)>;

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
// PseudoVSETLAMBDA / PseudoVSETLAMBDA_REG: read-modify-write `vtype.lambda`
// with a compile-time (immediate) or runtime (GPR) encoding. Expanded to a
// sequence that reads current vtype, masks out the lambda bits, ORs in the
// new encoding (an encoding of 0 requests the spec's preserve-or-initialize
// behavior in hardware), reads current vl, emits a register-form `vsetvl` to
// commit the updated vtype while preserving vl, and reads back the
// established (post-WARL) lambda field into $rd. The `$scratch` output
// exists only to give the register allocator a second scratch register for
// the read-modify-write sequence — its value is dead.
//
// PseudoQUERYLAMBDA: pure read of `vtype`. Extracts `vtype.lambda[2:0]` into
// $rd. No vtype write, no scratch required.
//===----------------------------------------------------------------------===//

let hasSideEffects = 1, mayLoad = 0, mayStore = 0, Defs = [VL, VTYPE],
    Constraints = "@earlyclobber $rd,@earlyclobber $scratch" in {
def PseudoVSETLAMBDA
    : Pseudo<(outs GPR:$rd, GPR:$scratch), (ins ixlenimm_li:$encoding), []>,
      Sched<[WriteVSETVL, ReadVSETVL]>;

// Register form for a runtime lambda encoding (spec: __riscv_vsetlambda
// accepts a runtime request). Same expansion shape as PseudoVSETLAMBDA except
// the new-encoding step masks $encoding to 3 bits (andi + slli) instead of
// materializing an immediate. The @earlyclobber on both outputs guarantees
// $encoding stays live (in a distinct register) until it is read mid-sequence.
def PseudoVSETLAMBDA_REG
    : Pseudo<(outs GPR:$rd, GPR:$scratch), (ins GPR:$encoding), []>,
      Sched<[WriteVSETVL, ReadVSETVL]>;
} // hasSideEffects = 1, ..., Constraints = "@earlyclobber ..."

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
// Each pseudo is generated per (mnemonic, LMUL ∈ {1,2,4,8}) so its register
// operand uses the LMUL-appropriate VR / VRM2 / VRM4 / VRM8 class. The single
// underlying real instruction (VMTL_V / VMTS_V / VMTTL_V / VMTTS_V) always
// encodes via the base VR register number; the LMUL grouping is established
// in vtype by a prior vsetvl_matrix. MC lowering (RISCVMatrixPseudo path)
// drops $passthru / $vl / $policy and adds zero_reg or V0 for $vm depending
// on masked-ness, and converts VRM{2,4,8} → VR via sub_vrm1_0.
//===----------------------------------------------------------------------===//

class PseudoVMatrixTileLoad<Instruction realInst, LMULInfo m>
    : Pseudo<(outs m.vrclass:$vd),
             (ins m.vrclass:$passthru, GPRMemZeroOffset:$rs1, GPR:$rs2,
                  lambdaarg:$lambda, GPRNoX0:$vl), []>,
      RISCVMatrixPseudo<realInst, /*TileLoad=*/1> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Uses = [VL, VTYPE];
  let Constraints = "$vd = $passthru";
}

class PseudoVMatrixTileLoadMask<Instruction realInst, LMULInfo m>
    : Pseudo<(outs m.vrclass:$vd),
             (ins m.vrclass:$passthru, GPRMemZeroOffset:$rs1, GPR:$rs2,
                  lambdaarg:$lambda, VMV0:$mask, GPRNoX0:$vl,
                  vec_policy:$policy), []>,
      RISCVMatrixPseudo<realInst, /*TileLoadMask=*/2> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasSideEffects = 0;
  let Uses = [VL, VTYPE, V0];
  let Constraints = "$vd = $passthru";
}

class PseudoVMatrixTileStore<Instruction realInst, LMULInfo m>
    : Pseudo<(outs),
             (ins m.vrclass:$vs3, GPRMemZeroOffset:$rs1, GPR:$rs2,
                  lambdaarg:$lambda, GPRNoX0:$vl), []>,
      RISCVMatrixPseudo<realInst, /*TileStore=*/3> {
  let mayLoad = 0;
  let mayStore = 1;
  let hasSideEffects = 0;
  let Uses = [VL, VTYPE];
}

class PseudoVMatrixTileStoreMask<Instruction realInst, LMULInfo m>
    : Pseudo<(outs),
             (ins m.vrclass:$vs3, GPRMemZeroOffset:$rs1, GPR:$rs2,
                  lambdaarg:$lambda, VMV0:$mask, GPRNoX0:$vl), []>,
      RISCVMatrixPseudo<realInst, /*TileStoreMask=*/4> {
  let mayLoad = 0;
  let mayStore = 1;
  let hasSideEffects = 0;
  let Uses = [VL, VTYPE, V0];
}

defvar AllZvvmTileLSLMULs = [V_M1, V_M2, V_M4, V_M8];

let Predicates = [HasStdExtZvvmtls] in
foreach m = AllZvvmTileLSLMULs in {
  def "PseudoVMTL_V_"#m.MX      : PseudoVMatrixTileLoad<VMTL_V, m>;
  def "PseudoVMTL_V_"#m.MX#"_MASK"  : PseudoVMatrixTileLoadMask<VMTL_V, m>;
  def "PseudoVMTS_V_"#m.MX      : PseudoVMatrixTileStore<VMTS_V, m>;
  def "PseudoVMTS_V_"#m.MX#"_MASK"  : PseudoVMatrixTileStoreMask<VMTS_V, m>;
}
let Predicates = [HasStdExtZvvmttls] in
foreach m = AllZvvmTileLSLMULs in {
  def "PseudoVMTTL_V_"#m.MX     : PseudoVMatrixTileLoad<VMTTL_V, m>;
  def "PseudoVMTTL_V_"#m.MX#"_MASK" : PseudoVMatrixTileLoadMask<VMTTL_V, m>;
  def "PseudoVMTTS_V_"#m.MX     : PseudoVMatrixTileStore<VMTTS_V, m>;
  def "PseudoVMTTS_V_"#m.MX#"_MASK" : PseudoVMatrixTileStoreMask<VMTTS_V, m>;
}

//===----------------------------------------------------------------------===//
// ISel patterns: intrinsic -> pseudo.
//
// The tile load/store instructions are element-type-agnostic at the encoding
// layer — SEW comes from vtype.vsew and the only thing the IR vector type
// affects in the backend is register-class selection. We emit one pattern per
// (intrinsic, lambda, element-type, LMUL) and dispatch to the matching
// per-LMUL pseudo (PseudoVMTL_V_M{1,2,4,8}, etc.) so VR / VRM2 / VRM4 / VRM8
// is picked up correctly via vti.RegClass.
//===----------------------------------------------------------------------===//

// Vector types the tile load/store builtins are ISel-pattern-matched over:
// every element type at LMUL ∈ {1, 2, 4, 8}. The corresponding mask type
// comes from `vti.Mask` and the register class from `vti.RegClass`, so the
// same generic Pat<> class works for every element width and LMUL.
defvar AllZvvmTileLSVTIs =
    [VI8M1,  VI8M2,  VI8M4,  VI8M8,
     VI16M1, VI16M2, VI16M4, VI16M8,
     VI32M1, VI32M2, VI32M4, VI32M8,
     VI64M1, VI64M2, VI64M4, VI64M8,
     VF16M1, VF16M2, VF16M4, VF16M8,
     VF32M1, VF32M2, VF32M4, VF32M8,
     VF64M1, VF64M2, VF64M4, VF64M8];

class TileLoadPat<Intrinsic intr, Instruction pseudo, int lambdaVal,
                  VTypeInfo vti>
    : Pat<(vti.Vector (intr (vti.Vector vti.RegClass:$passthru),
                            GPR:$rs1,
                            (XLenVT GPR:$rs2),
                            (XLenVT GPRNoX0:$vl))),
          (pseudo vti.RegClass:$passthru, GPR:$rs1, GPR:$rs2, lambdaVal,
                  GPRNoX0:$vl)>;

class TileLoadMaskPat<Intrinsic intr, Instruction pseudo, int lambdaVal,
                      VTypeInfo vti>
    : Pat<(vti.Vector (intr (vti.Vector vti.RegClass:$passthru),
                            GPR:$rs1,
                            (XLenVT GPR:$rs2),
                            (vti.Mask VMV0:$mask),
                            (XLenVT GPRNoX0:$vl),
                            (XLenVT timm:$policy))),
          (pseudo vti.RegClass:$passthru, GPR:$rs1, GPR:$rs2, lambdaVal,
                  VMV0:$mask, GPRNoX0:$vl, (XLenVT timm:$policy))>;

class TileStorePat<Intrinsic intr, Instruction pseudo, int lambdaVal,
                   VTypeInfo vti>
    : Pat<(intr (vti.Vector vti.RegClass:$vs3),
                GPR:$rs1,
                (XLenVT GPR:$rs2),
                (XLenVT GPRNoX0:$vl)),
          (pseudo vti.RegClass:$vs3, GPR:$rs1, GPR:$rs2, lambdaVal,
                  GPRNoX0:$vl)>;

class TileStoreMaskPat<Intrinsic intr, Instruction pseudo, int lambdaVal,
                       VTypeInfo vti>
    : Pat<(intr (vti.Vector vti.RegClass:$vs3),
                GPR:$rs1,
                (XLenVT GPR:$rs2),
                (vti.Mask VMV0:$mask),
                (XLenVT GPRNoX0:$vl)),
          (pseudo vti.RegClass:$vs3, GPR:$rs1, GPR:$rs2, lambdaVal,
                  VMV0:$mask, GPRNoX0:$vl)>;

// Each per-lambda intrinsic lowers to the same pseudo (for its (mnemonic,
// LMUL)) with a different literal lambda value in the operand slot. Lambda
// encoding:
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
  foreach vti = AllZvvmTileLSVTIs in {
    defvar mx = vti.LMul.MX;
    let Predicates = !listconcat([HasStdExtZvvmtls],
                                 GetVTypeMinimalPredicates<vti>.Predicates) in {
      def : TileLoadPat<
          !cast<Intrinsic>("int_riscv_vmtl"#lsuf),
          !cast<Instruction>("PseudoVMTL_V_"#mx), lval, vti>;
      def : TileLoadMaskPat<
          !cast<Intrinsic>("int_riscv_vmtl"#lsuf#"_mask"),
          !cast<Instruction>("PseudoVMTL_V_"#mx#"_MASK"), lval, vti>;
      def : TileStorePat<
          !cast<Intrinsic>("int_riscv_vmts"#lsuf),
          !cast<Instruction>("PseudoVMTS_V_"#mx), lval, vti>;
      def : TileStoreMaskPat<
          !cast<Intrinsic>("int_riscv_vmts"#lsuf#"_mask"),
          !cast<Instruction>("PseudoVMTS_V_"#mx#"_MASK"), lval, vti>;
    }
    let Predicates = !listconcat([HasStdExtZvvmttls],
                                 GetVTypeMinimalPredicates<vti>.Predicates) in {
      def : TileLoadPat<
          !cast<Intrinsic>("int_riscv_vmttl"#lsuf),
          !cast<Instruction>("PseudoVMTTL_V_"#mx), lval, vti>;
      def : TileLoadMaskPat<
          !cast<Intrinsic>("int_riscv_vmttl"#lsuf#"_mask"),
          !cast<Instruction>("PseudoVMTTL_V_"#mx#"_MASK"), lval, vti>;
      def : TileStorePat<
          !cast<Intrinsic>("int_riscv_vmtts"#lsuf),
          !cast<Instruction>("PseudoVMTTS_V_"#mx), lval, vti>;
      def : TileStoreMaskPat<
          !cast<Intrinsic>("int_riscv_vmtts"#lsuf#"_mask"),
          !cast<Instruction>("PseudoVMTTS_V_"#mx#"_MASK"), lval, vti>;
    }
  }
}
