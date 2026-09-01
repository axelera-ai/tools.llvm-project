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

// The `v0.scale` operand of the microscaled (vm=0) MAC forms. Parsed from
// the literal token "v0.scale" and always the V0 register; encoded via the
// vm bit exactly like a mask operand (V0 -> vm=0). Mirrors VMaskOp /
// VMaskCarryInOp from RISCVInstrInfoV.td.
def VScaleAsmOperand : AsmOperandClass {
  let Name = "RVVScaleRegOpOperand";
  let RenderMethod = "addRegOperands";
  let PredicateMethod = "isV0Reg";
  let ParserMethod = "parseVScaleReg";
  let DiagnosticType = "InvalidVScaleRegister";
  let DiagnosticString = "operand must be v0.scale";
}

def VScaleOp : RegisterOperand<VMV0> {
  let ParserMatchClass = VScaleAsmOperand;
  let PrintMethod = "printVScaleReg";
  let EncoderMethod = "getVMaskReg";
  // Deliberately no DecoderMethod: the default VMV0 register-class decoder
  // accepts only encoding 0 (= V0, vm=0). A vm=1 word therefore fails to
  // decode as a scaled form and falls through to the unscaled (vm=1
  // hardcoded) sibling def, keeping disassembly deterministic.
}

let hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
// op vd, vs1, vs2
//   vd is read-modified-written (tied operand).
//   vm is hardcoded to 1 (the unscaled form). The vm=0 encodings are the
//   microscaled MX forms and are separate instruction defs below: for the
//   FP widening MACs vm=0 is the same mnemonic with a trailing `v0.scale`
//   operand; for the integer widening MAC opcodes vm=0 encodes the distinct
//   integer-input FP-accumulate mnemonics vfwimmacc.vv / vfqimmacc.vv /
//   vf8wimmacc.vv. vm=0 remains reserved (undefined) for vmmacc.vv and
//   vfmmacc.vv.
// Encoding (OP-V major opcode):
//   [31:26] = funct6, [25] = vm, [24:20] = vs2, [19:15] = vs1,
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

// Microscaled (vm=0) MAC form: op vd, vs1, vs2, v0.scale
//   The mandatory trailing VScaleOp operand is always V0 and drives the vm
//   bit through getVMaskReg (V0 -> 0), following the VALUmVV (vadc.vvm)
//   carry-in precedent. Paired E8M0 block scales are read from v0; the
//   block size comes from vtype.bs.
class RVInstVMatrixMACScaleOp<bits<6> funct6, RISCVVFormat opv,
                              string opcodestr>
    : RVInstVV<funct6, opv, (outs VR:$vd_wb),
               (ins VR:$vd, VR:$vs1, VR:$vs2, VScaleOp:$vm),
               opcodestr, "$vd, $vs1, $vs2$vm"> {
  let Constraints = "$vd = $vd_wb";
}
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
// .td pattern can consume it. MC lowering rewrites the pseudo to the real
// instruction, discarding $vl (the real instruction reads VL implicitly from
// the vl CSR).
//
// Matrix multiply-accumulate has two independent register-group axes:
//   * vd  uses EMUL_C = VLEN / (SEW(C) * lambda^2), independent of LMUL.
//   * vs1 / vs2 use vtype.LMUL.
// PseudoVMatrixMACPair is parameterized over both. All eight MAC mnemonics
// (non-widening and widening, integer and FP) use the same pseudo shape: the
// pseudo is register-class-typed only, so one 16-cell pseudo grid per
// mnemonic serves every SEW combination; the per-SEW distinction lives in
// the ISel patterns.
//===----------------------------------------------------------------------===//

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
//   5 = MAC scaled (vm=0)    — pseudo (vd_wb, vd, vs1, vs2, scale, vl)
//                              → real (vd_wb, vd, vs1, vs2, V0)
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

// EMUL_C = 16 variant: vd is the IME m16 accumulator — a 16-register-aligned
// pair of M8 groups (VRN2M8, IR type riscv_nxv64i8x2). Same operand shape and
// MC lowering (Kind = MAC); baseVRRegister peels the tuple to its base VR.
let hasSideEffects = 0, mayLoad = 0, mayStore = 0,
    Uses = [VL, VTYPE], Constraints = "$vd = $vd_wb" in
class PseudoVMatrixMACPairM16<Instruction realInst, VTypeInfo abti>
    : Pseudo<(outs VRN2M8:$vd_wb),
             (ins VRN2M8:$vd, abti.RegClass:$vs1,
                  abti.RegClass:$vs2, GPRNoX0:$vl), []>,
      RISCVMatrixPseudo<realInst, /*MAC=*/0>;

// VTI lists for the MAC (EMUL_C, LMUL) cross-product. The pseudos only care
// about the register classes (VR / VRM2 / VRM4 / VRM8), so one grid per
// mnemonic covers every SEW combination.
// EMUL_C ∈ {1,2,4,8} (spec also permits 16; capped at 8 because upstream RVV
// has no VRM16 register class). LMUL ∈ {1,2,4,8}; fractional LMUL is spec-
// reserved for the IME family.
defvar AllZvvmMACVTIs_I32 = [VI32M1, VI32M2, VI32M4, VI32M8];
defvar AllZvvmMACVTIs_F32 = [VF32M1, VF32M2, VF32M4, VF32M8];

let Predicates = [HasStdExtZvvmm] in {
// Pseudo name: PseudoV{,W,Q,8W}MMACC_VV_M{EMUL_C}_M{LMUL} (in that order).
foreach cti = AllZvvmMACVTIs_I32 in
  foreach abti = AllZvvmMACVTIs_I32 in {
    def "PseudoVMMACC_VV_"#cti.LMul.MX#"_"#abti.LMul.MX
        : PseudoVMatrixMACPair<VMMACC_VV, cti, abti>;
    def "PseudoVWMMACC_VV_"#cti.LMul.MX#"_"#abti.LMul.MX
        : PseudoVMatrixMACPair<VWMMACC_VV, cti, abti>;
    def "PseudoVQMMACC_VV_"#cti.LMul.MX#"_"#abti.LMul.MX
        : PseudoVMatrixMACPair<VQMMACC_VV, cti, abti>;
    def "PseudoV8WMMACC_VV_"#cti.LMul.MX#"_"#abti.LMul.MX
        : PseudoVMatrixMACPair<V8WMMACC_VV, cti, abti>;
  }

// EMUL_C = 16 column: one M16 pseudo per (mnemonic, LMUL); the VRN2M8 vd
// class is SEW-agnostic, so — as with the M1..M8 grid — one pseudo serves
// every SEW combination.
foreach abti = AllZvvmMACVTIs_I32 in {
  def "PseudoVMMACC_VV_M16_"#abti.LMul.MX
      : PseudoVMatrixMACPairM16<VMMACC_VV, abti>;
  def "PseudoVWMMACC_VV_M16_"#abti.LMul.MX
      : PseudoVMatrixMACPairM16<VWMMACC_VV, abti>;
  def "PseudoVQMMACC_VV_M16_"#abti.LMul.MX
      : PseudoVMatrixMACPairM16<VQMMACC_VV, abti>;
  def "PseudoV8WMMACC_VV_M16_"#abti.LMul.MX
      : PseudoVMatrixMACPairM16<V8WMMACC_VV, abti>;
}
} // Predicates = [HasStdExtZvvmm]

//===----------------------------------------------------------------------===//
// ISel patterns: lower intrinsics to pseudos.
//
// Every MAC covers the full EMUL_C ∈ {1,2,4,8} × LMUL ∈ {1,2,4,8} cross
// product, since EMUL_C and LMUL are independent. For the widening MACs an
// additional axis is the (SEW(C), EEW(A/B)) combination — one pattern set is
// emitted per combination, all mapping to the same register-class-typed
// pseudo grid. Cells with 4-bit inputs (Int4 / OFP4) have no scalable IR
// element type and are deferred to the MX/format follow-up.
//
// Shared pattern class: one cell of a MAC's (EMUL_C, LMUL) grid at a fixed
// (C-type, A/B-type) combination.
//===----------------------------------------------------------------------===//

class ZvvmMACPat<Intrinsic intr, string pseudoBase,
                 VTypeInfo cti, VTypeInfo abti>
    : Pat<(cti.Vector (intr (cti.Vector cti.RegClass:$vd),
                            (abti.Vector abti.RegClass:$vs1),
                            (abti.Vector abti.RegClass:$vs2),
                            (XLenVT GPRNoX0:$vl))),
          (!cast<Instruction>("Pseudo"#pseudoBase#"_VV_"#cti.LMul.MX#"_"#abti.LMul.MX)
              cti.RegClass:$vd, abti.RegClass:$vs1, abti.RegClass:$vs2,
              GPRNoX0:$vl)>;

// EMUL_C = 16 cell: the C accumulator is the byte-typed m16 tuple
// (riscv_nxv64i8x2); the C-side SEW is erased at the IR level (it lives in
// vtype), so the (intrinsic, A/B type) pair alone identifies the spec row.
class ZvvmMACM16Pat<Intrinsic intr, string pseudoBase, VTypeInfo abti>
    : Pat<(riscv_nxv64i8x2 (intr (riscv_nxv64i8x2 VRN2M8:$vd),
                                 (abti.Vector abti.RegClass:$vs1),
                                 (abti.Vector abti.RegClass:$vs2),
                                 (XLenVT GPRNoX0:$vl))),
          (!cast<Instruction>("Pseudo"#pseudoBase#"_VV_M16_"#abti.LMul.MX)
              VRN2M8:$vd, abti.RegClass:$vs1, abti.RegClass:$vs2,
              GPRNoX0:$vl)>;

// Integer MACs. Rows: [pseudo base, C-type prefix, A/B-type prefix,
// per-type IME extension (spec tbl-extensions / tbl-int-encoding-map)].
// Patterns are gated on the per-type extension, which implies the Zvvmm
// family flag (the family flag alone enables only the MC instructions).
//   vmmacc   (W=1): SEW(C) = EEW(A/B) ∈ {8, 16, 32, 64}
//   vwmmacc  (W=2): (i16, i8), (i32, i16), (i64, i32)
//   vqmmacc  (W=4): (i32, i8), (i64, i16)   — (i16, i4) deferred (no i4 IR)
//   v8wmmacc (W=8): (i64, i8)               — (i32, i4) deferred (no i4 IR)
defvar ZvvmIntMACRows = [
  ["VMMACC",   "I8",  "I8",  "Zvvi8mm"],
  ["VMMACC",   "I16", "I16", "Zvvi16mm"],
  ["VMMACC",   "I32", "I32", "Zvvi32mm"],
  ["VMMACC",   "I64", "I64", "Zvvi64mm"],
  ["VWMMACC",  "I16", "I8",  "Zvvi8i16mm"],
  ["VWMMACC",  "I32", "I16", "Zvvi16i32mm"],
  ["VWMMACC",  "I64", "I32", "Zvvi32i64mm"],
  ["VQMMACC",  "I32", "I8",  "Zvvi8i32mm"],
  ["VQMMACC",  "I64", "I16", "Zvvi16i64mm"],
  ["V8WMMACC", "I64", "I8",  "Zvvi8i64mm"],
];

foreach row = ZvvmIntMACRows in {
  foreach cMX = ["M1", "M2", "M4", "M8"] in {
    foreach abMX = ["M1", "M2", "M4", "M8"] in {
      defvar cti  = !cast<VTypeInfo>("V" # row[1] # cMX);
      defvar abti = !cast<VTypeInfo>("V" # row[2] # abMX);
      let Predicates = !listconcat(
              [!cast<Predicate>("HasStdExt" # row[3])],
              GetVTypeMinimalPredicates<cti>.Predicates) in
      def : ZvvmMACPat<!cast<Intrinsic>("int_riscv_" # !tolower(row[0])),
                       row[0], cti, abti>;
    }
  }
}

// EMUL_C = 16 column of the integer grid. The C SEW is carried by vtype, not
// the IR, so only the A/B type distinguishes rows here; the row's per-type
// extension still gates the pattern.
foreach row = ZvvmIntMACRows in {
  foreach abMX = ["M1", "M2", "M4", "M8"] in {
    defvar abti = !cast<VTypeInfo>("V" # row[2] # abMX);
    let Predicates = !listconcat(
            [!cast<Predicate>("HasStdExt" # row[3])],
            GetVTypeMinimalPredicates<abti>.Predicates) in
    def : ZvvmMACM16Pat<!cast<Intrinsic>("int_riscv_" # !tolower(row[0]) # "_m16"),
                        row[0], abti>;
  }
}

//===----------------------------------------------------------------------===//
// Zvvfmm: floating-point matrix multiply-accumulate.
//
// Encoding bits per the IME spec's per-instruction wavedrom blocks:
//   vfmmacc.vv    funct6 = 0b010100 (0x14)
//   vfwmmacc.vv   funct6 = 0b010101 (0x15)
//   vfqmmacc.vv   funct6 = 0b010110 (0x16)
//   vf8wmmacc.vv  funct6 = 0b010111 (0x17)
// All share funct3 = OPFVV (0b001). vm=1 is the normal unscaled form; vm=0
// selects the microscaled MX form with paired E8M0 block scales in v0
// (trailing `v0.scale` operand). vm=0 of vfmmacc.vv is reserved.
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZvvfmm] in {
def VFMMACC_VV   : RVInstVFMatrixMAC<0b010100, "vfmmacc.vv">;
def VFWMMACC_VV  : RVInstVFMatrixMAC<0b010101, "vfwmmacc.vv">;
def VFQMMACC_VV  : RVInstVFMatrixMAC<0b010110, "vfqmmacc.vv">;
def VF8WMMACC_VV : RVInstVFMatrixMAC<0b010111, "vf8wmmacc.vv">;

// vm=0 microscaled siblings (same mnemonics, trailing v0.scale). Gated on
// the family flag: the encodings belong to the same instructions (the spec's
// "Included in" section lists Zvvfmm); whether a given (input format,
// accumulator format, bs) combination is legal is an encoding-map / vtype
// property checked at execution time, not at assembly time.
def VFWMMACC_VV_SCALE  : RVInstVMatrixMACScaleOp<0b010101, OPFVV,
                                                 "vfwmmacc.vv">;
def VFQMMACC_VV_SCALE  : RVInstVMatrixMACScaleOp<0b010110, OPFVV,
                                                 "vfqmmacc.vv">;
def VF8WMMACC_VV_SCALE : RVInstVMatrixMACScaleOp<0b010111, OPFVV,
                                                 "vf8wmmacc.vv">;
} // Predicates = [HasStdExtZvvfmm]

//===----------------------------------------------------------------------===//
// Integer-input FP-accumulate microscaled MACs (MX-only instructions).
//
// These are the vm=0 encodings of the integer widening MAC opcodes:
//   vfwimmacc.vv   funct6 = 0b111001 (vm=0 of vwmmacc.vv)
//   vfqimmacc.vv   funct6 = 0b111010 (vm=0 of vqmmacc.vv)
//   vf8wimmacc.vv  funct6 = 0b111011 (vm=0 of v8wmmacc.vv)
// funct3 = OPIVV. MXINT inputs are always signed (altfmt_A/altfmt_B must be
// 0); the accumulator is FP with the format selected by vtype.altfmt. The
// mandatory trailing v0.scale operand supplies the paired E8M0 block scales.
// vm=0 of vmmacc.vv (funct6 0b111000) remains reserved.
//===----------------------------------------------------------------------===//

let Predicates = [HasZvvmMXVfwimmacc] in
def VFWIMMACC_VV  : RVInstVMatrixMACScaleOp<0b111001, OPIVV, "vfwimmacc.vv">;
let Predicates = [HasZvvmMXVfqimmacc] in
def VFQIMMACC_VV  : RVInstVMatrixMACScaleOp<0b111010, OPIVV, "vfqimmacc.vv">;
let Predicates = [HasZvvmMXVf8wimmacc] in
def VF8WIMMACC_VV : RVInstVMatrixMACScaleOp<0b111011, OPIVV, "vf8wimmacc.vv">;

let Predicates = [HasStdExtZvvfmm] in {
// Pseudo name: PseudoVF{,W,Q,8W}MMACC_VV_M{EMUL_C}_M{LMUL} (in that order).
foreach cti = AllZvvmMACVTIs_F32 in
  foreach abti = AllZvvmMACVTIs_F32 in {
    def "PseudoVFMMACC_VV_"#cti.LMul.MX#"_"#abti.LMul.MX
        : PseudoVMatrixMACPair<VFMMACC_VV, cti, abti>;
    def "PseudoVFWMMACC_VV_"#cti.LMul.MX#"_"#abti.LMul.MX
        : PseudoVMatrixMACPair<VFWMMACC_VV, cti, abti>;
    def "PseudoVFQMMACC_VV_"#cti.LMul.MX#"_"#abti.LMul.MX
        : PseudoVMatrixMACPair<VFQMMACC_VV, cti, abti>;
    def "PseudoVF8WMMACC_VV_"#cti.LMul.MX#"_"#abti.LMul.MX
        : PseudoVMatrixMACPair<VF8WMMACC_VV, cti, abti>;
  }

// EMUL_C = 16 column (see the integer grid above).
foreach abti = AllZvvmMACVTIs_F32 in {
  def "PseudoVFMMACC_VV_M16_"#abti.LMul.MX
      : PseudoVMatrixMACPairM16<VFMMACC_VV, abti>;
  def "PseudoVFWMMACC_VV_M16_"#abti.LMul.MX
      : PseudoVMatrixMACPairM16<VFWMMACC_VV, abti>;
  def "PseudoVFQMMACC_VV_M16_"#abti.LMul.MX
      : PseudoVMatrixMACPairM16<VFQMMACC_VV, abti>;
  def "PseudoVF8WMMACC_VV_M16_"#abti.LMul.MX
      : PseudoVMatrixMACPairM16<VF8WMMACC_VV, abti>;
}
} // Predicates = [HasStdExtZvvfmm]

// ISel patterns for the FP MACs: full (EMUL_C, LMUL) ∈ {1,2,4,8}^2 grid per
// (SEW(C), EEW(A/B)) combination.
//
// Type choices mirror the integer side's container convention: when the IR
// has no first-class FP type for an element width (OFP8 / E4M3 / E5M2), we
// use the integer container (I8) and rely on vtype.altfmt_A/altfmt_B to
// select the FP8 interpretation at execution time. BF16 inputs
// (altfmt_A/B = 1 at 16-bit EEW) reuse the F16 IR container; format
// disambiguation happens at vsetvl-time, so no additional patterns are
// required. OFP4 inputs are deferred (no 4-bit IR element type).
//
// Rows: [pseudo base, C-type prefix, A/B-type prefix].
//   vfmmacc   (W=1): (f16, f16), (f32, f32), (f64, f64), and the SEW=8 row
//                    (i8, i8) — an OFP8 accumulator with OFP8 inputs
//                    (Zvvofp8mm); OFP8 has no first-class IR FP type, so
//                    both sides use the I8 container and vtype.altfmt /
//                    altfmt_A/altfmt_B select the E4M3/E5M2 interpretation.
//   vfwmmacc  (W=2): (f16, i8=OFP8), (f32, f16), (f64, f32)
//   vfqmmacc  (W=4): (f32, i8=OFP8), (f64, f16)
//   vf8wmmacc (W=8): (f64, i8=OFP8)
// Rows: [pseudo base, C-type prefix, A/B-type prefix, per-type IME
// extension (spec tbl-extensions / tbl-fp-encoding-map)]. Patterns are
// gated on the per-type extension, which implies the Zvvfmm family flag.
// BF16-format rows (Zvvbf16mm etc.) share the F16 IR container with the
// FP16 rows — the per-type gating for BF16 is a Clang/vtype concern until
// distinct BF16 rows exist.
defvar ZvvmFPMACRows = [
  ["VFMMACC",   "I8",  "I8",  "Zvvofp8mm"],
  ["VFMMACC",   "F16", "F16", "Zvvfp16mm"],
  ["VFMMACC",   "F32", "F32", "Zvvfp32mm"],
  ["VFMMACC",   "F64", "F64", "Zvvfp64mm"],
  ["VFWMMACC",  "F16", "I8",  "Zvvofp8fp16mm"],
  ["VFWMMACC",  "F32", "F16", "Zvvfp16fp32mm"],
  ["VFWMMACC",  "F64", "F32", "Zvvfp32fp64mm"],
  ["VFQMMACC",  "F32", "I8",  "Zvvofp8fp32mm"],
  ["VFQMMACC",  "F64", "F16", "Zvvfp16fp64mm"],
  ["VF8WMMACC", "F64", "I8",  "Zvvofp8fp64mm"],
];

foreach row = ZvvmFPMACRows in {
  foreach cMX = ["M1", "M2", "M4", "M8"] in {
    foreach abMX = ["M1", "M2", "M4", "M8"] in {
      defvar cti  = !cast<VTypeInfo>("V" # row[1] # cMX);
      defvar abti = !cast<VTypeInfo>("V" # row[2] # abMX);
      let Predicates = !listconcat(
              [!cast<Predicate>("HasStdExt" # row[3])],
              GetVTypeMinimalPredicates<cti>.Predicates,
              GetVTypeMinimalPredicates<abti>.Predicates) in
      def : ZvvmMACPat<!cast<Intrinsic>("int_riscv_" # !tolower(row[0])),
                       row[0], cti, abti>;
    }
  }
}

// EMUL_C = 16 column of the FP grid. Caveat vs. the M1..M8 grid: two FP rows
// of the same mnemonic may share an A/B IR container while differing only in
// the (vtype-carried) C SEW — e.g. vfmmacc F16 could be the FP16 row or a
// half of another. In the current row set every FP mnemonic's rows have
// distinct A/B containers, so the (intrinsic, A/B type) mapping stays
// unambiguous; a duplicate-pattern tablegen error will flag any future
// violation.
foreach row = ZvvmFPMACRows in {
  foreach abMX = ["M1", "M2", "M4", "M8"] in {
    defvar abti = !cast<VTypeInfo>("V" # row[2] # abMX);
    let Predicates = !listconcat(
            [!cast<Predicate>("HasStdExt" # row[3])],
            GetVTypeMinimalPredicates<abti>.Predicates) in
    def : ZvvmMACM16Pat<!cast<Intrinsic>("int_riscv_" # !tolower(row[0]) # "_m16"),
                        row[0], abti>;
  }
}

//===----------------------------------------------------------------------===//
// Microscaled (vm=0, `v0.scale`) MAC pseudos and ISel patterns.
//
// The scale operand is a full register of paired E8M0 block scales pinned to
// v0 through the VMV0Pair register class (nxv4i16 = vuint16m1_t). The `bs`
// immarg of the *_scaled / vf*immacc intrinsics is a semantic record of the
// block size the caller compiled against and is dropped here — the
// architectural block size is vtype.bs, programmed by the caller via
// vsetvl_matrix (exactly as altfmt_A/altfmt_B carry signedness/format).
//
// Pattern rows follow the MX columns of the spec encoding maps at EEW >= 8
// (4-bit MXINT4 / MXFP4 input rows are deferred — no 4-bit IR element type):
//   vfwmmacc.vv  vm=0: (F16 accum, I8 = MXFP8 inputs)   Zvvxofp8fp16mm
//   vfqmmacc.vv  vm=0: (F32 accum, I8 = MXFP8 inputs)   Zvvxofp8fp32mm
//   vf8wmmacc.vv vm=0: (F64 accum, I8 = MXFP8 inputs)   Zvvxofp8fp64mm
//   vfwimmacc.vv:      (F16 accum, I8 = MXINT8 inputs)  Zvvxi8fp16mm
//   vfqimmacc.vv:      (F32 accum, I8 = MXINT8 inputs)  Zvvxi8fp32mm
//   vf8wimmacc.vv:     (F64 accum, I8 = MXINT8 inputs)  Zvvxi8fp64mm
// BF16-accumulator MX rows share the F16 IR container (vtype.altfmt selects
// the format); BS=16 rows share the BS=32 patterns (bs is not encoded).
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 0, mayStore = 0,
    Uses = [VL, VTYPE], Constraints = "$vd = $vd_wb" in {
class PseudoVMatrixMACPairScaled<Instruction realInst,
                                 VTypeInfo cti, VTypeInfo abti>
    : Pseudo<(outs cti.RegClass:$vd_wb),
             (ins cti.RegClass:$vd, abti.RegClass:$vs1,
                  abti.RegClass:$vs2, VMV0Pair:$scale, GPRNoX0:$vl), []>,
      RISCVMatrixPseudo<realInst, /*MACScaled=*/5>;

class PseudoVMatrixMACPairScaledM16<Instruction realInst, VTypeInfo abti>
    : Pseudo<(outs VRN2M8:$vd_wb),
             (ins VRN2M8:$vd, abti.RegClass:$vs1,
                  abti.RegClass:$vs2, VMV0Pair:$scale, GPRNoX0:$vl), []>,
      RISCVMatrixPseudo<realInst, /*MACScaled=*/5>;
} // hasSideEffects = 0, ..., Constraints

// FP scaled pseudo grids (vm=0 forms of the FP widening mnemonics).
let Predicates = [HasStdExtZvvfmm] in {
foreach cti = AllZvvmMACVTIs_F32 in
  foreach abti = AllZvvmMACVTIs_F32 in {
    def "PseudoVFWMMACC_VV_"#cti.LMul.MX#"_"#abti.LMul.MX#"_SCALED"
        : PseudoVMatrixMACPairScaled<VFWMMACC_VV_SCALE, cti, abti>;
    def "PseudoVFQMMACC_VV_"#cti.LMul.MX#"_"#abti.LMul.MX#"_SCALED"
        : PseudoVMatrixMACPairScaled<VFQMMACC_VV_SCALE, cti, abti>;
    def "PseudoVF8WMMACC_VV_"#cti.LMul.MX#"_"#abti.LMul.MX#"_SCALED"
        : PseudoVMatrixMACPairScaled<VF8WMMACC_VV_SCALE, cti, abti>;
  }
foreach abti = AllZvvmMACVTIs_F32 in {
  def "PseudoVFWMMACC_VV_M16_"#abti.LMul.MX#"_SCALED"
      : PseudoVMatrixMACPairScaledM16<VFWMMACC_VV_SCALE, abti>;
  def "PseudoVFQMMACC_VV_M16_"#abti.LMul.MX#"_SCALED"
      : PseudoVMatrixMACPairScaledM16<VFQMMACC_VV_SCALE, abti>;
  def "PseudoVF8WMMACC_VV_M16_"#abti.LMul.MX#"_SCALED"
      : PseudoVMatrixMACPairScaledM16<VF8WMMACC_VV_SCALE, abti>;
}
} // Predicates = [HasStdExtZvvfmm]

// Integer-input (MXINT) pseudo grids: MX-only instructions.
let Predicates = [HasZvvmMXVfwimmacc] in {
foreach cti = AllZvvmMACVTIs_F32 in
  foreach abti = AllZvvmMACVTIs_F32 in
    def "PseudoVFWIMMACC_VV_"#cti.LMul.MX#"_"#abti.LMul.MX
        : PseudoVMatrixMACPairScaled<VFWIMMACC_VV, cti, abti>;
foreach abti = AllZvvmMACVTIs_F32 in
  def "PseudoVFWIMMACC_VV_M16_"#abti.LMul.MX
      : PseudoVMatrixMACPairScaledM16<VFWIMMACC_VV, abti>;
}
let Predicates = [HasZvvmMXVfqimmacc] in {
foreach cti = AllZvvmMACVTIs_F32 in
  foreach abti = AllZvvmMACVTIs_F32 in
    def "PseudoVFQIMMACC_VV_"#cti.LMul.MX#"_"#abti.LMul.MX
        : PseudoVMatrixMACPairScaled<VFQIMMACC_VV, cti, abti>;
foreach abti = AllZvvmMACVTIs_F32 in
  def "PseudoVFQIMMACC_VV_M16_"#abti.LMul.MX
      : PseudoVMatrixMACPairScaledM16<VFQIMMACC_VV, abti>;
}
let Predicates = [HasZvvmMXVf8wimmacc] in {
foreach cti = AllZvvmMACVTIs_F32 in
  foreach abti = AllZvvmMACVTIs_F32 in
    def "PseudoVF8WIMMACC_VV_"#cti.LMul.MX#"_"#abti.LMul.MX
        : PseudoVMatrixMACPairScaled<VF8WIMMACC_VV, cti, abti>;
foreach abti = AllZvvmMACVTIs_F32 in
  def "PseudoVF8WIMMACC_VV_M16_"#abti.LMul.MX
      : PseudoVMatrixMACPairScaledM16<VF8WIMMACC_VV, abti>;
}

class ZvvmMACScaledPat<Intrinsic intr, string pseudoBase, string suffix,
                       VTypeInfo cti, VTypeInfo abti>
    : Pat<(cti.Vector (intr (cti.Vector cti.RegClass:$vd),
                            (abti.Vector abti.RegClass:$vs1),
                            (abti.Vector abti.RegClass:$vs2),
                            (nxv4i16 VMV0Pair:$scale),
                            (XLenVT timm),
                            (XLenVT GPRNoX0:$vl))),
          (!cast<Instruction>("Pseudo"#pseudoBase#"_VV_"#cti.LMul.MX#"_"#
                              abti.LMul.MX#suffix)
              cti.RegClass:$vd, abti.RegClass:$vs1, abti.RegClass:$vs2,
              VMV0Pair:$scale, GPRNoX0:$vl)>;

class ZvvmMACScaledM16Pat<Intrinsic intr, string pseudoBase, string suffix,
                          VTypeInfo abti>
    : Pat<(riscv_nxv64i8x2 (intr (riscv_nxv64i8x2 VRN2M8:$vd),
                                 (abti.Vector abti.RegClass:$vs1),
                                 (abti.Vector abti.RegClass:$vs2),
                                 (nxv4i16 VMV0Pair:$scale),
                                 (XLenVT timm),
                                 (XLenVT GPRNoX0:$vl))),
          (!cast<Instruction>("Pseudo"#pseudoBase#"_VV_M16_"#abti.LMul.MX#
                              suffix)
              VRN2M8:$vd, abti.RegClass:$vs1, abti.RegClass:$vs2,
              VMV0Pair:$scale, GPRNoX0:$vl)>;

// Rows: [pseudo base, pseudo suffix, intrinsic, C-type prefix, A/B-type
// prefix, MX extension].
defvar ZvvmMXMACRows = [
  ["VFWMMACC",   "_SCALED", "vfwmmacc_scaled",  "F16", "I8", "Zvvxofp8fp16mm"],
  ["VFQMMACC",   "_SCALED", "vfqmmacc_scaled",  "F32", "I8", "Zvvxofp8fp32mm"],
  ["VF8WMMACC",  "_SCALED", "vf8wmmacc_scaled", "F64", "I8", "Zvvxofp8fp64mm"],
  ["VFWIMMACC",  "",        "vfwimmacc",        "F16", "I8", "Zvvxi8fp16mm"],
  ["VFQIMMACC",  "",        "vfqimmacc",        "F32", "I8", "Zvvxi8fp32mm"],
  ["VF8WIMMACC", "",        "vf8wimmacc",       "F64", "I8", "Zvvxi8fp64mm"],
];

foreach row = ZvvmMXMACRows in {
  foreach cMX = ["M1", "M2", "M4", "M8"] in {
    foreach abMX = ["M1", "M2", "M4", "M8"] in {
      defvar cti  = !cast<VTypeInfo>("V" # row[3] # cMX);
      defvar abti = !cast<VTypeInfo>("V" # row[4] # abMX);
      let Predicates = !listconcat(
              [!cast<Predicate>("HasStdExt" # row[5])],
              GetVTypeMinimalPredicates<cti>.Predicates,
              GetVTypeMinimalPredicates<abti>.Predicates) in
      def : ZvvmMACScaledPat<!cast<Intrinsic>("int_riscv_" # row[2]),
                             row[0], row[1], cti, abti>;
    }
  }
  foreach abMX = ["M1", "M2", "M4", "M8"] in {
    defvar abti = !cast<VTypeInfo>("V" # row[4] # abMX);
    let Predicates = !listconcat(
            [!cast<Predicate>("HasStdExt" # row[5])],
            GetVTypeMinimalPredicates<abti>.Predicates) in
    def : ZvvmMACScaledM16Pat<!cast<Intrinsic>("int_riscv_" # row[2] # "_m16"),
                              row[0], row[1], abti>;
  }
}

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
