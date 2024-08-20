//==- RISCVSchedCVA6V.td - CVA6V Scheduling Definitions ------*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//

/// c is true if mx has the worst case behavior compared to LMULs in MxList. The worst case LMUL is
/// the Largest LMUL and the worst case SEW is the smallest SEW for that LMUL.
class CVA6VIsWorstCaseMX<string mx, list<string> MxList> {
  string LLMUL = LargestLMUL<MxList>.r;
  bit c = !eq(mx, LLMUL);
}

class CVA6VIsWorstCaseMXSEW<string mx, int sew, list<string> MxList, bit isF = 0> {
  string LLMUL = LargestLMUL<MxList>.r;
  int SSEW = SmallestSEW<mx, isF>.r;
  bit c = !and(!eq(mx, LLMUL), !eq(sew, SSEW));
}

// VLSU cycles for dense accesses, assuming total memory bandwidth is 1/4 of VLEN
class CVA6VGetLMulCyclesVLSUdense<string mx> {
  int c = !cond(
    !eq(mx, "M1") : 4,
    !eq(mx, "M2") : 8,
    !eq(mx, "M4") : 16,
    !eq(mx, "M8") : 32,
    !eq(mx, "MF2") : 2,
    !eq(mx, "MF4") : 1,
    !eq(mx, "MF8") : 1
  );
}

// VLSU cycles for sparse accesses, assuming the number of memory interfaces is VLEN/512
class CVA6VGetLMulCyclesVLSUsparse<string mx, int sew> {
  int c = !cond(
    !eq(mx, "M1") : !div(512, sew),
    !eq(mx, "M2") : !div(1024, sew),
    !eq(mx, "M4") : !div(2048, sew),
    !eq(mx, "M8") : !div(4096, sew),
    !eq(mx, "MF2") : !div(256, sew),
    !eq(mx, "MF4") : !div(128, sew),
    !eq(mx, "MF8") : !div(64, sew)
  );
}

// VALU cycles assuming VALU datapath is 1/4 of VLEN
class CVA6VGetLMulCyclesVALU<string mx> {
  int c = !cond(
    !eq(mx, "M1") : 4,
    !eq(mx, "M2") : 8,
    !eq(mx, "M4") : 16,
    !eq(mx, "M8") : 32,
    !eq(mx, "MF2") : 2,
    !eq(mx, "MF4") : 1,
    !eq(mx, "MF8") : 1
  );
}

// VSLD cycles assuming VSLD datapath is 1/16 of VLEN
class CVA6VGetLMulCyclesVSLD<string mx> {
  int c = !cond(
    !eq(mx, "M1") : 16,
    !eq(mx, "M2") : 32,
    !eq(mx, "M4") : 64,
    !eq(mx, "M8") : 128,
    !eq(mx, "MF2") : 8,
    !eq(mx, "MF4") : 4,
    !eq(mx, "MF8") : 2
  );
}

// ELEM unit cycles assuming VLEN is 4096
class CVA6VGetLMulCyclesELEM<string mx, int sew> {
  defvar VLEN = 4096;
  int c = !cond(
    !eq(mx, "M1") : !div(VLEN, sew),
    !eq(mx, "M2") : !div(!mul(VLEN, 2), sew),
    !eq(mx, "M4") : !div(!mul(VLEN, 4), sew),
    !eq(mx, "M8") : !div(!mul(VLEN, 8), sew),
    !eq(mx, "MF2") : !div(!div(VLEN, 2), sew),
    !eq(mx, "MF4") : !div(!div(VLEN, 4), sew),
    !eq(mx, "MF8") : !div(!div(VLEN, 8), sew),
  );
}

defvar CVA6VLatencyVLSU    = !add(6, 8); // 6 cycles in VLSU, 8 cycles memory latency
defvar CVA6VLatencyVALUint = 7;
defvar CVA6VLatencyVALUfp  = 8;
defvar CVA6VLatencyVSLD    = 3;
defvar CVA6VLatencyELEM    = 5;

// CVA6V machine model for scheduling and other instruction cost heuristics.
def CVA6VModel : SchedMachineModel {
  let MicroOpBufferSize = 0; // CVA6 is in-order.
  let IssueWidth = 1;        // 1 micro-op is dispatched per cycle.
  let LoadLatency = 3;       // D$ access has latency of 3 cycles on a hit.
  let MispredictPenalty = 5; // Misprediction is detected in 5th stage (EX).
  let CompleteModel = false;
  let UnsupportedFeatures = [HasStdExtZbkb, HasStdExtZbkc, HasStdExtZbkx,
                             HasStdExtZcmt, HasStdExtZknd, HasStdExtZkne,
                             HasStdExtZknh, HasStdExtZksed, HasStdExtZksh,
                             HasStdExtZkr, HasVInstructionsI64];
}

//===----------------------------------------------------------------------===//
// Define each kind of processor resource and number available.

// Modeling each unit as a ProcResource with BufferSize = 1 since CVA6 executes
// out-of-order.

let BufferSize = 1 in {
def CVA6VUnitLSU         : ProcResource<1>; // CVA6 Load/Store Unit
def CVA6VUnitALU         : ProcResource<1>; // CVA6 Int ALU
def CVA6VUnitCSR         : ProcResource<1>; // CVA6 CSR Buffer
def CVA6VUnitFPU_FMA     : ProcResource<1>; // CVA6 FPU multiplier and adder
def CVA6VUnitFPU_DIVSQRT : ProcResource<1>; // CVA6 FPU division and square root
def CVA6VUnitFPU_NONCOMP : ProcResource<1>; // CVA6 FPU non-computational operations
def CVA6VUnitFPU_CONV    : ProcResource<1>; // CVA6 FPU conversions
def CVA6VUnitIMulDiv     : ProcResource<1>; // CVA6 Int Multiply and Divide
def CVA6VUnitBranch      : ProcResource<1>; // CVA6 Branch Unit
def CVA6VUnitVCFG        : ProcResource<1>; // Raptor Vector Configuration Pseudo-Unit
def CVA6VUnitVLSU        : ProcResource<1>; // Raptor Vector Load/Store Unit
def CVA6VUnitVALU        : ProcResource<1>; // Raptor Vector ALU
def CVA6VUnitVSLD        : ProcResource<1>; // Raptor Vector Slide Unit
}

//===----------------------------------------------------------------------===//

// Note: The scheduling model for CVA6V includes definitions for the D extension (double precision
// floating-point) and the full V extension (incl 64-bit elements) because scheduling models for
// 64-bit machines need to include those.
let SchedModel = CVA6VModel in {

// Branching
def : WriteRes<WriteJmp, [CVA6VUnitBranch]>;
def : WriteRes<WriteJal, [CVA6VUnitBranch]>;
def : WriteRes<WriteJalr, [CVA6VUnitBranch]>;

// Integer arithmetic and logic
def : WriteRes<WriteIALU32, [CVA6VUnitALU]>;
def : WriteRes<WriteIALU, [CVA6VUnitALU]>;
def : WriteRes<WriteShiftImm32, [CVA6VUnitALU]>;
def : WriteRes<WriteShiftImm, [CVA6VUnitALU]>;
def : WriteRes<WriteShiftReg32, [CVA6VUnitALU]>;
def : WriteRes<WriteShiftReg, [CVA6VUnitALU]>;

// CSR access
def : WriteRes<WriteCSR, [CVA6VUnitCSR]>;

// Integer multiplication (performed in two cycles, fully pipelined)
let Latency = 2 in {
def : WriteRes<WriteIMul, [CVA6VUnitIMulDiv]>;
def : WriteRes<WriteIMul32, [CVA6VUnitIMulDiv]>;
}

// Integer division and remainder (worst-case as many cycles as bits in the operands)
def : WriteRes<WriteIDiv32, [CVA6VUnitIMulDiv]> {
  let Latency = 32;
  let ReleaseAtCycles = [32];
}
def : WriteRes<WriteIDiv, [CVA6VUnitIMulDiv]> {
  let Latency = 64;
  let ReleaseAtCycles = [64];
}

/////////////////
// Memory access

def : WriteRes<WriteSTB, [CVA6VUnitLSU]>;
def : WriteRes<WriteSTH, [CVA6VUnitLSU]>;
def : WriteRes<WriteSTW, [CVA6VUnitLSU]>;
def : WriteRes<WriteSTD, [CVA6VUnitLSU]>;
def : WriteRes<WriteFST16, [CVA6VUnitLSU]>;
def : WriteRes<WriteFST32, [CVA6VUnitLSU]>;
def : WriteRes<WriteFST64, [CVA6VUnitLSU]>;

// Load latency is 3 cycles on a D$ hit
let Latency = 3 in {
def : WriteRes<WriteLDB, [CVA6VUnitLSU]>;
def : WriteRes<WriteLDH, [CVA6VUnitLSU]>;
def : WriteRes<WriteLDW, [CVA6VUnitLSU]>;
def : WriteRes<WriteLDD, [CVA6VUnitLSU]>;
def : WriteRes<WriteFLD16, [CVA6VUnitLSU]>;
def : WriteRes<WriteFLD32, [CVA6VUnitLSU]>;
def : WriteRes<WriteFLD64, [CVA6VUnitLSU]>;

// Atomic memory
def : WriteRes<WriteAtomicW, [CVA6VUnitLSU]>;
def : WriteRes<WriteAtomicD, [CVA6VUnitLSU]>;

def : WriteRes<WriteAtomicLDW, [CVA6VUnitLSU]>;
def : WriteRes<WriteAtomicLDD, [CVA6VUnitLSU]>;
}

def : WriteRes<WriteAtomicSTW, [CVA6VUnitLSU]>;
def : WriteRes<WriteAtomicSTD, [CVA6VUnitLSU]>;

/////////////////////////////
// Floating-point operations

// Half precision FMA
let Latency = 1 in {
def : WriteRes<WriteFAdd16, [CVA6VUnitFPU_FMA]>;
def : WriteRes<WriteFMul16, [CVA6VUnitFPU_FMA]>;
def : WriteRes<WriteFMA16, [CVA6VUnitFPU_FMA]>;
}

// Single precision FMA
let Latency = 2 in {
def : WriteRes<WriteFAdd32, [CVA6VUnitFPU_FMA]>;
def : WriteRes<WriteFMul32, [CVA6VUnitFPU_FMA]>;
def : WriteRes<WriteFMA32, [CVA6VUnitFPU_FMA]>;
}

// Double precision FMA
let Latency = 3 in {
def : WriteRes<WriteFAdd64, [CVA6VUnitFPU_FMA]>;
def : WriteRes<WriteFMul64, [CVA6VUnitFPU_FMA]>;
def : WriteRes<WriteFMA64, [CVA6VUnitFPU_FMA]>;
}

// FP division and square-root, worst-case takes as many cycles as there are
// bits in the significand, with an extra 2 cycles of latency
let Latency = 26, ReleaseAtCycles = [24] in {
def : WriteRes<WriteFDiv16, [CVA6VUnitFPU_DIVSQRT]>;
def : WriteRes<WriteFSqrt16, [CVA6VUnitFPU_DIVSQRT]>;
def : WriteRes<WriteFDiv32, [CVA6VUnitFPU_DIVSQRT]>;
def : WriteRes<WriteFSqrt32, [CVA6VUnitFPU_DIVSQRT]>;
}
let Latency = 55, ReleaseAtCycles = [53] in {
def : WriteRes<WriteFDiv64, [CVA6VUnitFPU_DIVSQRT]>;
def : WriteRes<WriteFSqrt64, [CVA6VUnitFPU_DIVSQRT]>;
}

// FP non-comp
let Latency = 1 in {
def : WriteRes<WriteFSGNJ16, [CVA6VUnitFPU_NONCOMP]>;
def : WriteRes<WriteFSGNJ32, [CVA6VUnitFPU_NONCOMP]>;
def : WriteRes<WriteFSGNJ64, [CVA6VUnitFPU_NONCOMP]>;
def : WriteRes<WriteFMinMax16, [CVA6VUnitFPU_NONCOMP]>;
def : WriteRes<WriteFMinMax32, [CVA6VUnitFPU_NONCOMP]>;
def : WriteRes<WriteFMinMax64, [CVA6VUnitFPU_NONCOMP]>;
def : WriteRes<WriteFClass16, [CVA6VUnitFPU_NONCOMP]>;
def : WriteRes<WriteFClass32, [CVA6VUnitFPU_NONCOMP]>;
def : WriteRes<WriteFClass64, [CVA6VUnitFPU_NONCOMP]>;
def : WriteRes<WriteFCmp16, [CVA6VUnitFPU_NONCOMP]>;
def : WriteRes<WriteFCmp32, [CVA6VUnitFPU_NONCOMP]>;
def : WriteRes<WriteFCmp64, [CVA6VUnitFPU_NONCOMP]>;
}

// Conversions
let Latency = 2 in {
def : WriteRes<WriteFCvtI32ToF16, [CVA6VUnitFPU_CONV]>;
def : WriteRes<WriteFCvtI32ToF32, [CVA6VUnitFPU_CONV]>;
def : WriteRes<WriteFCvtI32ToF64, [CVA6VUnitFPU_CONV]>;
def : WriteRes<WriteFCvtI64ToF16, [CVA6VUnitFPU_CONV]>;
def : WriteRes<WriteFCvtI64ToF32, [CVA6VUnitFPU_CONV]>;
def : WriteRes<WriteFCvtI64ToF64, [CVA6VUnitFPU_CONV]>;
def : WriteRes<WriteFCvtF16ToI32, [CVA6VUnitFPU_CONV]>;
def : WriteRes<WriteFCvtF16ToI64, [CVA6VUnitFPU_CONV]>;
def : WriteRes<WriteFCvtF32ToI32, [CVA6VUnitFPU_CONV]>;
def : WriteRes<WriteFCvtF32ToI64, [CVA6VUnitFPU_CONV]>;
def : WriteRes<WriteFCvtF64ToI32, [CVA6VUnitFPU_CONV]>;
def : WriteRes<WriteFCvtF64ToI64, [CVA6VUnitFPU_CONV]>;
def : WriteRes<WriteFCvtF32ToF64, [CVA6VUnitFPU_CONV]>;
def : WriteRes<WriteFCvtF64ToF32, [CVA6VUnitFPU_CONV]>;
def : WriteRes<WriteFCvtF16ToF32, [CVA6VUnitFPU_CONV]>;
def : WriteRes<WriteFCvtF32ToF16, [CVA6VUnitFPU_CONV]>;
def : WriteRes<WriteFCvtF16ToF64, [CVA6VUnitFPU_CONV]>;
def : WriteRes<WriteFCvtF64ToF16, [CVA6VUnitFPU_CONV]>;
def : WriteRes<WriteFMovF16ToI16, [CVA6VUnitFPU_CONV]>;
def : WriteRes<WriteFMovI16ToF16, [CVA6VUnitFPU_CONV]>;
def : WriteRes<WriteFMovF32ToI32, [CVA6VUnitFPU_CONV]>;
def : WriteRes<WriteFMovI32ToF32, [CVA6VUnitFPU_CONV]>;
def : WriteRes<WriteFMovF64ToI64, [CVA6VUnitFPU_CONV]>;
def : WriteRes<WriteFMovI64ToF64, [CVA6VUnitFPU_CONV]>;
}

/////////////////////
// Vector Operations

// Configuration-Setting Instructions
def : WriteRes<WriteVSETVLI, [CVA6VUnitVCFG]>;
def : WriteRes<WriteVSETIVLI, [CVA6VUnitVCFG]>;
def : WriteRes<WriteVSETVL, [CVA6VUnitVCFG]>;

// Dense Vector Loads and Stores
foreach mx = SchedMxList in {
  defvar LMulCycles = CVA6VGetLMulCyclesVLSUdense<mx>.c;
  defvar IsWorstCase = CVA6VIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = CVA6VLatencyVLSU, ReleaseAtCycles = [LMulCycles] in {
    defm "" : LMULWriteResMX<"WriteVLDE",    [CVA6VUnitVLSU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDM",    [CVA6VUnitVLSU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDFF",   [CVA6VUnitVLSU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDS8",   [CVA6VUnitVLSU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDS16",  [CVA6VUnitVLSU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDS32",  [CVA6VUnitVLSU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVLDS64",  [CVA6VUnitVLSU], mx, IsWorstCase>;
  }
  let Latency = CVA6VLatencyVLSU, ReleaseAtCycles = [LMulCycles] in {
    defm "" : LMULWriteResMX<"WriteVSTE",    [CVA6VUnitVLSU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTM",    [CVA6VUnitVLSU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTS8",   [CVA6VUnitVLSU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTS16",  [CVA6VUnitVLSU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTS32",  [CVA6VUnitVLSU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSTS64",  [CVA6VUnitVLSU], mx, IsWorstCase>;
  }
  foreach eew = [8, 16, 32, 64] in {
    foreach nf=2-8 in {
      let Latency = CVA6VLatencyVLSU, ReleaseAtCycles = [!mul(nf, LMulCycles)] in {
        defm "" : LMULWriteResMX<"WriteVLSEG" # nf # "e" # eew,   [CVA6VUnitVLSU], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVLSEGFF" # nf # "e" # eew, [CVA6VUnitVLSU], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVLSSEG" # nf # "e" # eew,  [CVA6VUnitVLSU], mx, IsWorstCase>;
      }
      let Latency = CVA6VLatencyVLSU, ReleaseAtCycles = [!mul(nf, LMulCycles)] in {
        defm "" : LMULWriteResMX<"WriteVSSEG" # nf # "e" # eew,   [CVA6VUnitVLSU], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVSSSEG" # nf # "e" # eew,  [CVA6VUnitVLSU], mx, IsWorstCase>;
      }
    }
  }
}

// Sparse Vector Loads and Stores
foreach mx = SchedMxList in {
  foreach eew = [8, 16, 32, 64] in {
    defvar LMulCycles = CVA6VGetLMulCyclesVLSUsparse<mx, eew>.c;
    // note: a dedicated worst case is defined for each EEW, hence we do not CVA6VIsWorstCaseMXSEW
    defvar IsWorstCase = CVA6VIsWorstCaseMX<mx, SchedMxList>.c;
    let Latency = CVA6VLatencyVLSU, ReleaseAtCycles = [LMulCycles] in {
      defm "" : LMULWriteResMX<"WriteVLDUX" # eew,  [CVA6VUnitVLSU], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVLDOX" # eew,  [CVA6VUnitVLSU], mx, IsWorstCase>;
    }
    let Latency = CVA6VLatencyVLSU, ReleaseAtCycles = [LMulCycles] in {
      defm "" : LMULWriteResMX<"WriteVSTUX" # eew,  [CVA6VUnitVLSU], mx, IsWorstCase>;
      defm "" : LMULWriteResMX<"WriteVSTOX" # eew,  [CVA6VUnitVLSU], mx, IsWorstCase>;
    }
    foreach nf=2-8 in {
      let Latency = CVA6VLatencyVLSU, ReleaseAtCycles = [!mul(nf, LMulCycles)] in {
        defm "" : LMULWriteResMX<"WriteVLUXSEG" # nf # "e" # eew, [CVA6VUnitVLSU], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVLOXSEG" # nf # "e" # eew, [CVA6VUnitVLSU], mx, IsWorstCase>;
      }
      let Latency = CVA6VLatencyVLSU, ReleaseAtCycles = [!mul(nf, LMulCycles)] in {
        defm "" : LMULWriteResMX<"WriteVSUXSEG" # nf # "e" # eew, [CVA6VUnitVLSU], mx, IsWorstCase>;
        defm "" : LMULWriteResMX<"WriteVSOXSEG" # nf # "e" # eew, [CVA6VUnitVLSU], mx, IsWorstCase>;
      }
    }
  }
}

// Whole register move/load/store
let Latency = CVA6VLatencyVALUint, ReleaseAtCycles = [!mul(4, 1)] in
  def : WriteRes<WriteVMov1V, [CVA6VUnitVALU]>;
let Latency = CVA6VLatencyVALUint, ReleaseAtCycles = [!mul(4, 2)] in
  def : WriteRes<WriteVMov2V, [CVA6VUnitVALU]>;
let Latency = CVA6VLatencyVALUint, ReleaseAtCycles = [!mul(4, 4)] in
  def : WriteRes<WriteVMov4V, [CVA6VUnitVALU]>;
let Latency = CVA6VLatencyVALUint, ReleaseAtCycles = [!mul(4, 8)] in
  def : WriteRes<WriteVMov8V, [CVA6VUnitVALU]>;
let Latency = CVA6VLatencyVLSU, ReleaseAtCycles = [!mul(4, 1)] in
  def : WriteRes<WriteVLD1R,  [CVA6VUnitVLSU]>;
let Latency = CVA6VLatencyVLSU, ReleaseAtCycles = [!mul(4, 2)] in
  def : WriteRes<WriteVLD2R,  [CVA6VUnitVLSU]>;
let Latency = CVA6VLatencyVLSU, ReleaseAtCycles = [!mul(4, 4)] in
  def : WriteRes<WriteVLD4R,  [CVA6VUnitVLSU]>;
let Latency = CVA6VLatencyVLSU, ReleaseAtCycles = [!mul(4, 8)] in
  def : WriteRes<WriteVLD8R,  [CVA6VUnitVLSU]>;
let Latency = CVA6VLatencyVLSU, ReleaseAtCycles = [!mul(4, 1)] in
  def : WriteRes<WriteVST1R,  [CVA6VUnitVLSU]>;
let Latency = CVA6VLatencyVLSU, ReleaseAtCycles = [!mul(4, 2)] in
  def : WriteRes<WriteVST2R,  [CVA6VUnitVLSU]>;
let Latency = CVA6VLatencyVLSU, ReleaseAtCycles = [!mul(4, 4)] in
  def : WriteRes<WriteVST4R,  [CVA6VUnitVLSU]>;
let Latency = CVA6VLatencyVLSU, ReleaseAtCycles = [!mul(4, 8)] in
  def : WriteRes<WriteVST8R,  [CVA6VUnitVLSU]>;

// 11. Vector Integer Arithmetic Instructions
foreach mx = SchedMxList in {
  defvar LMulCycles = CVA6VGetLMulCyclesVALU<mx>.c;
  defvar IsWorstCase = CVA6VIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = CVA6VLatencyVALUint, ReleaseAtCycles = [LMulCycles] in {
    defm "" : LMULWriteResMX<"WriteVIALUV",   [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIALUX",   [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIALUI",   [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVExtV",    [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICALUV",  [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICALUX",  [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICALUI",  [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICmpV",   [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICmpX",   [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVICmpI",   [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMergeV", [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMergeX", [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMergeI", [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMovV",   [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMovX",   [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMovI",   [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVShiftV",   [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVShiftX",   [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVShiftI",   [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMinMaxV", [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMinMaxX", [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMulV",    [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMulX",    [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIMulAddX", [CVA6VUnitVALU], mx, IsWorstCase>;
  }
  // 3 operands
  let Latency = !add(CVA6VLatencyVALUint, 1), ReleaseAtCycles = [!mul(2, LMulCycles)] in {
    defm "" : LMULWriteResMX<"WriteVIMulAddV", [CVA6VUnitVALU], mx, IsWorstCase>;
  }
}
// Widening
foreach mx = SchedMxListW in {
  defvar LMulCycles = CVA6VGetLMulCyclesVALU<mx>.c;
  defvar IsWorstCase = CVA6VIsWorstCaseMX<mx, SchedMxListW>.c;
  let Latency = !add(CVA6VLatencyVALUint, 1), ReleaseAtCycles = [!mul(2, LMulCycles)] in {
    defm "" : LMULWriteResMX<"WriteVIWALUV",    [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWALUX",    [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWALUI",    [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulV",    [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulX",    [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulAddV", [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVIWMulAddX", [CVA6VUnitVALU], mx, IsWorstCase>;
  }
}
// Narrowing Shift and Clips
foreach mx = SchedMxListW in {
  defvar LMulCycles = CVA6VGetLMulCyclesVALU<mx>.c;
  defvar IsWorstCase = CVA6VIsWorstCaseMX<mx, SchedMxListW>.c;
  let Latency = CVA6VLatencyVALUint, ReleaseAtCycles = [!mul(2, LMulCycles)] in {
    defm "" : LMULWriteResMX<"WriteVNShiftV", [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNShiftX", [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNShiftI", [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNClipV",  [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNClipX",  [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVNClipI",  [CVA6VUnitVALU], mx, IsWorstCase>;
  }
}

// 12. Vector Fixed-Point Arithmetic Instructions
foreach mx = SchedMxList in {
  defvar LMulCycles = CVA6VGetLMulCyclesVALU<mx>.c;
  defvar IsWorstCase = CVA6VIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = CVA6VLatencyVALUint, ReleaseAtCycles = [LMulCycles] in {
    defm "" : LMULWriteResMX<"WriteVSALUV",   [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSALUX",   [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSALUI",   [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVAALUV",   [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVAALUX",   [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSMulV",   [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSMulX",   [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSShiftV", [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSShiftX", [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVSShiftI", [CVA6VUnitVALU], mx, IsWorstCase>;
  }
}

// 13. Vector Floating-Point Instructions
foreach mx = SchedMxListF in {
  defvar LMulCycles = CVA6VGetLMulCyclesVALU<mx>.c;
  defvar IsWorstCase = CVA6VIsWorstCaseMX<mx, SchedMxListF>.c;
  let Latency = CVA6VLatencyVALUfp, ReleaseAtCycles = [LMulCycles] in {
    defm "" : LMULWriteResMX<"WriteVFALUV",     [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFALUF",     [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFMulV",     [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFMulF",     [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFMulAddF",  [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFRecpV",    [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFMinMaxV",  [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFMinMaxF",  [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFSgnjV",    [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFSgnjF",    [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFCmpV",     [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFCmpF",     [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFClassV",   [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFMergeV",   [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFMovV",     [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFCvtIToFV", [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFCvtFToIV", [CVA6VUnitVALU], mx, IsWorstCase>;
  }
  // 3 operands
  let Latency = !add(CVA6VLatencyVALUfp, 1), ReleaseAtCycles = [!mul(2, LMulCycles)] in {
    defm "" : LMULWriteResMX<"WriteVFMulAddV", [CVA6VUnitVALU], mx, IsWorstCase>;
  }
}
// Widening
foreach mx = SchedMxListFW in {
  defvar LMulCycles = CVA6VGetLMulCyclesVALU<mx>.c;
  defvar IsWorstCase = CVA6VIsWorstCaseMX<mx, SchedMxListFW>.c;
  let Latency = !add(CVA6VLatencyVALUfp, 1), ReleaseAtCycles = [!mul(2, LMulCycles)] in {
    defm "" : LMULWriteResMX<"WriteVFWALUV",     [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFWALUF",     [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFWMulV",     [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFWMulF",     [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFWMulAddV",  [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFWMulAddF",  [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFWCvtFToFV", [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFWCvtFToIV", [CVA6VUnitVALU], mx, IsWorstCase>;
  }
}
foreach mx = SchedMxListW in {
  defvar LMulCycles = CVA6VGetLMulCyclesVALU<mx>.c;
  defvar IsWorstCase = CVA6VIsWorstCaseMX<mx, SchedMxListW>.c;
  let Latency = !add(CVA6VLatencyVALUfp, 1), ReleaseAtCycles = [!mul(2, LMulCycles)] in {
    defm "" : LMULWriteResMX<"WriteVFWCvtIToFV", [CVA6VUnitVALU], mx, IsWorstCase>;
  }
}
// Narrowing
foreach mx = SchedMxListFW in {
  defvar LMulCycles = CVA6VGetLMulCyclesVALU<mx>.c;
  defvar IsWorstCase = CVA6VIsWorstCaseMX<mx, SchedMxListFW>.c;
  let Latency = CVA6VLatencyVALUfp, ReleaseAtCycles = [!mul(2, LMulCycles)] in {
    defm "" : LMULWriteResMX<"WriteVFNCvtFToFV", [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFNCvtIToFV", [CVA6VUnitVALU], mx, IsWorstCase>;
  }
}
foreach mx = SchedMxListW in {
  defvar LMulCycles = CVA6VGetLMulCyclesVALU<mx>.c;
  defvar IsWorstCase = CVA6VIsWorstCaseMX<mx, SchedMxListW>.c;
  let Latency = CVA6VLatencyVALUfp, ReleaseAtCycles = [!mul(2, LMulCycles)] in {
    defm "" : LMULWriteResMX<"WriteVFNCvtFToIV", [CVA6VUnitVALU], mx, IsWorstCase>;
  }
}
// Worst case needs 8 cycles for each subword that is 1/8 of VALU datapath => 64 cycles per word
foreach mx = SchedMxListF in {
  foreach sew = SchedSEWSet<mx, isF=1>.val in {
    defvar LMulCycles = CVA6VGetLMulCyclesVALU<mx>.c;
    defvar IsWorstCase = CVA6VIsWorstCaseMXSEW<mx, sew, SchedMxListF, isF=1>.c;
    let Latency = !add(CVA6VLatencyVALUfp, 8), ReleaseAtCycles = [!mul(64, LMulCycles)] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFDivV",  [CVA6VUnitVALU], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFDivF",  [CVA6VUnitVALU], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFSqrtV", [CVA6VUnitVALU], mx, sew, IsWorstCase>;
    }
  }
}
// Integer division done on FP div/sqrt unit
foreach mx = SchedMxList in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar LMulCycles = CVA6VGetLMulCyclesVALU<mx>.c;
    defvar IsWorstCase = CVA6VIsWorstCaseMXSEW<mx, sew, SchedMxList>.c;
    let Latency = !add(CVA6VLatencyVALUfp, 8), ReleaseAtCycles = [!mul(64, LMulCycles)] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVIDivV", [CVA6VUnitVALU], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVIDivX", [CVA6VUnitVALU], mx, sew, IsWorstCase>;
    }
  }
}

// 15. Vector Mask Instructions
// - executed on VALU
foreach mx = SchedMxList in {
  defvar LMulCycles = CVA6VGetLMulCyclesVALU<mx>.c;
  defvar IsWorstCase = CVA6VIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = CVA6VLatencyVALUint, ReleaseAtCycles = [LMulCycles] in {
    defm "" : LMULWriteResMX<"WriteVMALUV", [CVA6VUnitVALU], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVMIdxV", [CVA6VUnitVALU], mx, IsWorstCase>;
  }
}
// - executed on ELEM unit
foreach mx = SchedMxList in {
  defvar LMulCycles = CVA6VGetLMulCyclesELEM<mx, 8>.c;
  defvar IsWorstCase = CVA6VIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = CVA6VLatencyELEM, ReleaseAtCycles = [LMulCycles] in {
    defm "" : LMULWriteResMX<"WriteVMPopV", [CVA6VUnitVSLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVMFFSV", [CVA6VUnitVSLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVMSFSV", [CVA6VUnitVSLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVMIotV", [CVA6VUnitVALU], mx, IsWorstCase>;

  }
}

// 14. Vector Reduction Operations
foreach mx = SchedMxList in {
  foreach sew = SchedSEWSet<mx>.val in {
    defvar LMulCycles = CVA6VGetLMulCyclesELEM<mx, sew>.c;
    defvar IsWorstCase = CVA6VIsWorstCaseMXSEW<mx, sew, SchedMxList>.c;
    let Latency = !add(CVA6VLatencyELEM, LMulCycles), ReleaseAtCycles = [LMulCycles] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVIRedV_From",       [CVA6VUnitVSLD], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVIRedMinMaxV_From", [CVA6VUnitVSLD], mx, sew, IsWorstCase>;
    }
  }
}
foreach mx = SchedMxListWRed in {
  foreach sew = SchedSEWSet<mx, isF=0, isWidening=1>.val in {
    defvar LMulCycles = CVA6VGetLMulCyclesELEM<mx, sew>.c;
    defvar IsWorstCase = CVA6VIsWorstCaseMXSEW<mx, sew, SchedMxListWRed>.c;
    let Latency = !add(CVA6VLatencyELEM, LMulCycles), ReleaseAtCycles = [LMulCycles] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVIWRedV_From", [CVA6VUnitVSLD], mx, sew, IsWorstCase>;
    }
  }
}
foreach mx = SchedMxListF in {
  foreach sew = SchedSEWSet<mx, isF=1>.val in {
    defvar LMulCycles = CVA6VGetLMulCyclesELEM<mx, sew>.c;
    defvar IsWorstCase = CVA6VIsWorstCaseMXSEW<mx, sew, SchedMxListF, isF=1>.c;
    let Latency = !add(CVA6VLatencyELEM, LMulCycles), ReleaseAtCycles = [LMulCycles] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFRedV_From",       [CVA6VUnitVSLD], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFRedMinMaxV_From", [CVA6VUnitVSLD], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFRedOV_From",      [CVA6VUnitVSLD], mx, sew, IsWorstCase>;
    }
  }
}
foreach mx = SchedMxListFWRed in {
  foreach sew = SchedSEWSet<mx, isF=1, isWidening=1>.val in {
    defvar LMulCycles = CVA6VGetLMulCyclesELEM<mx, sew>.c;
    defvar IsWorstCase = CVA6VIsWorstCaseMXSEW<mx, sew, SchedMxListFWRed, isF=1>.c;
    let Latency = !add(CVA6VLatencyELEM, LMulCycles), ReleaseAtCycles = [LMulCycles] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWRedV_From",  [CVA6VUnitVSLD], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVFWRedOV_From", [CVA6VUnitVSLD], mx, sew, IsWorstCase>;
    }
  }
}

// 16. Vector Permutation Instructions
// Slides
foreach mx = SchedMxList in {
  defvar LMulCycles = CVA6VGetLMulCyclesVSLD<mx>.c;
  defvar IsWorstCase = CVA6VIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = CVA6VLatencyVSLD, ReleaseAtCycles = [LMulCycles] in {
    defm "" : LMULWriteResMX<"WriteVISlideI",  [CVA6VUnitVSLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVISlideX",  [CVA6VUnitVSLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVISlide1X", [CVA6VUnitVSLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVFSlide1F", [CVA6VUnitVSLD], mx, IsWorstCase>;
  }
}
// Scalar move
let Latency = CVA6VLatencyELEM, ReleaseAtCycles = [1] in {
  def : WriteRes<WriteVIMovVX, [CVA6VUnitVSLD]>;
  def : WriteRes<WriteVIMovXV, [CVA6VUnitVSLD]>;
  def : WriteRes<WriteVFMovVF, [CVA6VUnitVSLD]>;
  def : WriteRes<WriteVFMovFV, [CVA6VUnitVSLD]>;
}
// Gather and Compress
foreach mx = SchedMxList in {
  //foreach sew = SchedSEWSet<mx>.val in {
    //defvar LMulCycles = CVA6VGetLMulCyclesELEM<mx, sew>.c;
    //defvar IsWorstCase = CVA6VIsWorstCaseMXSEW<mx, sew, SchedMxList>.c;
  defvar LMulCycles = CVA6VGetLMulCyclesELEM<mx, 8>.c;
  defvar IsWorstCase = CVA6VIsWorstCaseMX<mx, SchedMxList>.c;
  let Latency = CVA6VLatencyELEM, ReleaseAtCycles = [LMulCycles] in {
    defm "" : LMULWriteResMX<"WriteVRGatherVI", [CVA6VUnitVSLD], mx, IsWorstCase>;
    defm "" : LMULWriteResMX<"WriteVRGatherVX", [CVA6VUnitVSLD], mx, IsWorstCase>;
  }
  foreach sew = SchedSEWSet<mx>.val in {
    defvar LMulCycles = CVA6VGetLMulCyclesELEM<mx, sew>.c;
    defvar IsWorstCase = CVA6VIsWorstCaseMXSEW<mx, sew, SchedMxList>.c;
    let Latency = CVA6VLatencyELEM, ReleaseAtCycles = [LMulCycles] in {
      defm "" : LMULSEWWriteResMXSEW<"WriteVRGatherVV", [CVA6VUnitVSLD], mx, sew, IsWorstCase>;
      defm "" : LMULSEWWriteResMXSEW<"WriteVCompressV", [CVA6VUnitVSLD], mx, sew, IsWorstCase>;
    }
  }
}

// Others
def : WriteRes<WriteNop, []>;
def : WriteRes<WriteRdVLENB, [CVA6VUnitVCFG]>;

def : InstRW<[WriteIALU], (instrs COPY)>;

//===----------------------------------------------------------------------===//
// Bypass and advance
def : ReadAdvance<ReadJmp, 0>;
def : ReadAdvance<ReadJalr, 0>;
def : ReadAdvance<ReadCSR, 0>;
def : ReadAdvance<ReadStoreData, 0>;
def : ReadAdvance<ReadMemBase, 0>;
def : ReadAdvance<ReadIALU, 0>;
def : ReadAdvance<ReadIALU32, 0>;
def : ReadAdvance<ReadShiftImm, 0>;
def : ReadAdvance<ReadShiftImm32, 0>;
def : ReadAdvance<ReadShiftReg, 0>;
def : ReadAdvance<ReadShiftReg32, 0>;
def : ReadAdvance<ReadIDiv, 0>;
def : ReadAdvance<ReadIDiv32, 0>;
def : ReadAdvance<ReadIMul, 0>;
def : ReadAdvance<ReadIMul32, 0>;
def : ReadAdvance<ReadAtomicWA, 0>;
def : ReadAdvance<ReadAtomicWD, 0>;
def : ReadAdvance<ReadAtomicDA, 0>;
def : ReadAdvance<ReadAtomicDD, 0>;
def : ReadAdvance<ReadAtomicLDW, 0>;
def : ReadAdvance<ReadAtomicLDD, 0>;
def : ReadAdvance<ReadAtomicSTW, 0>;
def : ReadAdvance<ReadAtomicSTD, 0>;
def : ReadAdvance<ReadFStoreData, 0>;
def : ReadAdvance<ReadFMemBase, 0>;
def : ReadAdvance<ReadFAdd16, 0>;
def : ReadAdvance<ReadFAdd32, 0>;
def : ReadAdvance<ReadFAdd64, 0>;
def : ReadAdvance<ReadFMul16, 0>;
def : ReadAdvance<ReadFMul32, 0>;
def : ReadAdvance<ReadFMul64, 0>;
def : ReadAdvance<ReadFMA16, 0>;
def : ReadAdvance<ReadFMA16Addend, 0>;
def : ReadAdvance<ReadFMA32, 0>;
def : ReadAdvance<ReadFMA32Addend, 0>;
def : ReadAdvance<ReadFMA64, 0>;
def : ReadAdvance<ReadFMA64Addend, 0>;
def : ReadAdvance<ReadFDiv16, 0>;
def : ReadAdvance<ReadFDiv32, 0>;
def : ReadAdvance<ReadFDiv64, 0>;
def : ReadAdvance<ReadFSqrt16, 0>;
def : ReadAdvance<ReadFSqrt32, 0>;
def : ReadAdvance<ReadFSqrt64, 0>;
def : ReadAdvance<ReadFCmp16, 0>;
def : ReadAdvance<ReadFCmp32, 0>;
def : ReadAdvance<ReadFCmp64, 0>;
def : ReadAdvance<ReadFSGNJ16, 0>;
def : ReadAdvance<ReadFSGNJ32, 0>;
def : ReadAdvance<ReadFSGNJ64, 0>;
def : ReadAdvance<ReadFMinMax16, 0>;
def : ReadAdvance<ReadFMinMax32, 0>;
def : ReadAdvance<ReadFMinMax64, 0>;
def : ReadAdvance<ReadFCvtF16ToI32, 0>;
def : ReadAdvance<ReadFCvtF16ToI64, 0>;
def : ReadAdvance<ReadFCvtF32ToI32, 0>;
def : ReadAdvance<ReadFCvtF32ToI64, 0>;
def : ReadAdvance<ReadFCvtF64ToI32, 0>;
def : ReadAdvance<ReadFCvtF64ToI64, 0>;
def : ReadAdvance<ReadFCvtI32ToF16, 0>;
def : ReadAdvance<ReadFCvtI32ToF32, 0>;
def : ReadAdvance<ReadFCvtI32ToF64, 0>;
def : ReadAdvance<ReadFCvtI64ToF16, 0>;
def : ReadAdvance<ReadFCvtI64ToF32, 0>;
def : ReadAdvance<ReadFCvtI64ToF64, 0>;
def : ReadAdvance<ReadFCvtF32ToF64, 0>;
def : ReadAdvance<ReadFCvtF64ToF32, 0>;
def : ReadAdvance<ReadFCvtF16ToF32, 0>;
def : ReadAdvance<ReadFCvtF32ToF16, 0>;
def : ReadAdvance<ReadFCvtF16ToF64, 0>;
def : ReadAdvance<ReadFCvtF64ToF16, 0>;
def : ReadAdvance<ReadFMovF16ToI16, 0>;
def : ReadAdvance<ReadFMovI16ToF16, 0>;
def : ReadAdvance<ReadFMovF32ToI32, 0>;
def : ReadAdvance<ReadFMovI32ToF32, 0>;
def : ReadAdvance<ReadFMovF64ToI64, 0>;
def : ReadAdvance<ReadFMovI64ToF64, 0>;
def : ReadAdvance<ReadFClass16, 0>;
def : ReadAdvance<ReadFClass32, 0>;
def : ReadAdvance<ReadFClass64, 0>;

// 6. Configuration-Setting Instructions
def : ReadAdvance<ReadVSETVLI, 2>;
def : ReadAdvance<ReadVSETVL, 2>;

// 7. Vector Loads and Stores
def : ReadAdvance<ReadVLDX, 0>;
def : ReadAdvance<ReadVSTX, 0>;
defm "" : LMULReadAdvance<"ReadVSTEV", 0>;
defm "" : LMULReadAdvance<"ReadVSTM", 0>;
def : ReadAdvance<ReadVLDSX, 0>;
def : ReadAdvance<ReadVSTSX, 0>;
defm "" : LMULReadAdvance<"ReadVSTS8V", 0>;
defm "" : LMULReadAdvance<"ReadVSTS16V", 0>;
defm "" : LMULReadAdvance<"ReadVSTS32V", 0>;
defm "" : LMULReadAdvance<"ReadVSTS64V", 0>;
defm "" : LMULReadAdvance<"ReadVLDUXV", 0>;
defm "" : LMULReadAdvance<"ReadVLDOXV", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX8", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX16", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX32", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX64", 0>;
defm "" : LMULReadAdvance<"ReadVSTUXV", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX8V", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX16V", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX32V", 0>;
defm "" : LMULReadAdvance<"ReadVSTUX64V", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX8", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX16", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX32", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX64", 0>;
defm "" : LMULReadAdvance<"ReadVSTOXV", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX8V", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX16V", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX32V", 0>;
defm "" : LMULReadAdvance<"ReadVSTOX64V", 0>;
// LMUL Aware
def : ReadAdvance<ReadVST1R, 0>;
def : ReadAdvance<ReadVST2R, 0>;
def : ReadAdvance<ReadVST4R, 0>;
def : ReadAdvance<ReadVST8R, 0>;

// 12. Vector Integer Arithmetic Instructions
defm : LMULReadAdvance<"ReadVIALUV", 0>;
defm : LMULReadAdvance<"ReadVIALUX", 0>;
defm : LMULReadAdvanceW<"ReadVIWALUV", 0>;
defm : LMULReadAdvanceW<"ReadVIWALUX", 0>;
defm : LMULReadAdvance<"ReadVExtV", 0>;
defm : LMULReadAdvance<"ReadVICALUV", 0>;
defm : LMULReadAdvance<"ReadVICALUX", 0>;
defm : LMULReadAdvance<"ReadVShiftV", 0>;
defm : LMULReadAdvance<"ReadVShiftX", 0>;
defm : LMULReadAdvanceW<"ReadVNShiftV", 0>;
defm : LMULReadAdvanceW<"ReadVNShiftX", 0>;
defm : LMULReadAdvance<"ReadVICmpV", 0>;
defm : LMULReadAdvance<"ReadVICmpX", 0>;
defm : LMULReadAdvance<"ReadVIMinMaxV", 0>;
defm : LMULReadAdvance<"ReadVIMinMaxX", 0>;
defm : LMULReadAdvance<"ReadVIMulV", 0>;
defm : LMULReadAdvance<"ReadVIMulX", 0>;
defm : LMULSEWReadAdvance<"ReadVIDivV", 0>;
defm : LMULSEWReadAdvance<"ReadVIDivX", 0>;
defm : LMULReadAdvanceW<"ReadVIWMulV", 0>;
defm : LMULReadAdvanceW<"ReadVIWMulX", 0>;
defm : LMULReadAdvance<"ReadVIMulAddV", 0>;
defm : LMULReadAdvance<"ReadVIMulAddX", 0>;
defm : LMULReadAdvanceW<"ReadVIWMulAddV", 0>;
defm : LMULReadAdvanceW<"ReadVIWMulAddX", 0>;
defm : LMULReadAdvance<"ReadVIMergeV", 0>;
defm : LMULReadAdvance<"ReadVIMergeX", 0>;
defm : LMULReadAdvance<"ReadVIMovV", 0>;
defm : LMULReadAdvance<"ReadVIMovX", 0>;

// 13. Vector Fixed-Point Arithmetic Instructions
defm "" : LMULReadAdvance<"ReadVSALUV", 0>;
defm "" : LMULReadAdvance<"ReadVSALUX", 0>;
defm "" : LMULReadAdvance<"ReadVAALUV", 0>;
defm "" : LMULReadAdvance<"ReadVAALUX", 0>;
defm "" : LMULReadAdvance<"ReadVSMulV", 0>;
defm "" : LMULReadAdvance<"ReadVSMulX", 0>;
defm "" : LMULReadAdvance<"ReadVSShiftV", 0>;
defm "" : LMULReadAdvance<"ReadVSShiftX", 0>;
defm "" : LMULReadAdvanceW<"ReadVNClipV", 0>;
defm "" : LMULReadAdvanceW<"ReadVNClipX", 0>;

// 14. Vector Floating-Point Instructions
defm "" : LMULReadAdvance<"ReadVFALUV", 0>;
defm "" : LMULReadAdvance<"ReadVFALUF", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWALUV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWALUF", 0>;
defm "" : LMULReadAdvance<"ReadVFMulV", 0>;
defm "" : LMULReadAdvance<"ReadVFMulF", 0>;
defm "" : LMULSEWReadAdvanceF<"ReadVFDivV", 0>;
defm "" : LMULSEWReadAdvanceF<"ReadVFDivF", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWMulV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWMulF", 0>;
defm "" : LMULReadAdvance<"ReadVFMulAddV", 0>;
defm "" : LMULReadAdvance<"ReadVFMulAddF", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWMulAddV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWMulAddF", 0>;
defm "" : LMULSEWReadAdvanceF<"ReadVFSqrtV", 0>;
defm "" : LMULReadAdvance<"ReadVFRecpV", 0>;
defm "" : LMULReadAdvance<"ReadVFMinMaxV", 0>;
defm "" : LMULReadAdvance<"ReadVFMinMaxF", 0>;
defm "" : LMULReadAdvance<"ReadVFSgnjV", 0>;
defm "" : LMULReadAdvance<"ReadVFSgnjF", 0>;
defm "" : LMULReadAdvance<"ReadVFCmpV", 0>;
defm "" : LMULReadAdvance<"ReadVFCmpF", 0>;
defm "" : LMULReadAdvance<"ReadVFClassV", 0>;
defm "" : LMULReadAdvance<"ReadVFMergeV", 0>;
defm "" : LMULReadAdvance<"ReadVFMergeF", 0>;
defm "" : LMULReadAdvance<"ReadVFMovF", 0>;
defm "" : LMULReadAdvance<"ReadVFCvtIToFV", 0>;
defm "" : LMULReadAdvance<"ReadVFCvtFToIV", 0>;
defm "" : LMULReadAdvanceW<"ReadVFWCvtIToFV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWCvtFToIV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFWCvtFToFV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFNCvtIToFV", 0>;
defm "" : LMULReadAdvanceW<"ReadVFNCvtFToIV", 0>;
defm "" : LMULReadAdvanceFW<"ReadVFNCvtFToFV", 0>;

// 15. Vector Reduction Operations
def : ReadAdvance<ReadVIRedV, 0>;
def : ReadAdvance<ReadVIRedV0, 0>;
def : ReadAdvance<ReadVIWRedV, 0>;
def : ReadAdvance<ReadVIWRedV0, 0>;
def : ReadAdvance<ReadVFRedV, 0>;
def : ReadAdvance<ReadVFRedV0, 0>;
def : ReadAdvance<ReadVFRedOV, 0>;
def : ReadAdvance<ReadVFRedOV0, 0>;
def : ReadAdvance<ReadVFWRedV, 0>;
def : ReadAdvance<ReadVFWRedV0, 0>;
def : ReadAdvance<ReadVFWRedOV, 0>;
def : ReadAdvance<ReadVFWRedOV0, 0>;

// 16. Vector Mask Instructions
defm "" : LMULReadAdvance<"ReadVMALUV", 0>;
defm "" : LMULReadAdvance<"ReadVMPopV", 0>;
defm "" : LMULReadAdvance<"ReadVMFFSV", 0>;
defm "" : LMULReadAdvance<"ReadVMSFSV", 0>;
defm "" : LMULReadAdvance<"ReadVMIotV", 0>;

// 17. Vector Permutation Instructions
def : ReadAdvance<ReadVIMovVX, 0>;
def : ReadAdvance<ReadVIMovXV, 0>;
def : ReadAdvance<ReadVIMovXX, 0>;
def : ReadAdvance<ReadVFMovVF, 0>;
def : ReadAdvance<ReadVFMovFV, 0>;
def : ReadAdvance<ReadVFMovFX, 0>;
defm "" : LMULReadAdvance<"ReadVISlideV", 0>;
defm "" : LMULReadAdvance<"ReadVISlideX", 0>;
defm "" : LMULReadAdvance<"ReadVFSlideV", 0>;
defm "" : LMULReadAdvance<"ReadVFSlideF", 0>;
defm "" : LMULSEWReadAdvance<"ReadVRGatherVV_data", 0>;
defm "" : LMULSEWReadAdvance<"ReadVRGatherVV_index", 0>;
defm "" : LMULReadAdvance<"ReadVRGatherVX_data", 0>;
defm "" : LMULReadAdvance<"ReadVRGatherVX_index", 0>;
defm "" : LMULReadAdvance<"ReadVRGatherVI_data", 0>;
defm "" : LMULSEWReadAdvance<"ReadVCompressV", 0>;
// LMUL Aware
def : ReadAdvance<ReadVMov1V, 0>;
def : ReadAdvance<ReadVMov2V, 0>;
def : ReadAdvance<ReadVMov4V, 0>;
def : ReadAdvance<ReadVMov8V, 0>;

// Others
def : ReadAdvance<ReadVMask, 0>;
def : ReadAdvance<ReadVMergeOp_WorstCase, 0>;
foreach mx = SchedMxList in {
  def : ReadAdvance<!cast<SchedRead>("ReadVMergeOp_" # mx), 0>;
  foreach sew = SchedSEWSet<mx>.val in
    def : ReadAdvance<!cast<SchedRead>("ReadVMergeOp_" # mx  # "_E" # sew), 0>;
}

//===----------------------------------------------------------------------===//
// Unsupported extensions
defm : UnsupportedSchedZba;
defm : UnsupportedSchedZbb;
defm : UnsupportedSchedZbc;
defm : UnsupportedSchedZbs;
defm : UnsupportedSchedZbkb;
defm : UnsupportedSchedZbkx;
defm : UnsupportedSchedZfa;
defm : UnsupportedSchedSFB;
//defm : UnsupportedSchedZvbb;
//defm : UnsupportedSchedZvbc;
//defm : UnsupportedSchedZvkb;
//defm : UnsupportedSchedZvkg;
//defm : UnsupportedSchedZvknha;
//defm : UnsupportedSchedZvknhb;
//defm : UnsupportedSchedZvkned;
//defm : UnsupportedSchedZvksed;
}
