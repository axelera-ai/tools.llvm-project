# Zvvm FP32 GEMM example

FP32 counterpart of `examples/zvvm-gemm`. A minimal bare-metal RV64 program
that exercises the Zvvfmm floating-point matrix multiply-accumulate
(`vfmmacc.vv`) together with the Zvvm tile load/store and full-vtype
`vsetvl.matrix`. The kernel performs `C = A * B^T` for a 4x4 FP32 matrix
of mixed-sign power-of-two-denominator values (each product and partial
sum is exact in fp32) and returns `(int)C[0][0]`, which evaluates to `7`.

The build target is bare-metal spike via the HTIF `tohost` exit
convention — no `pk`, no libc, no sysroot required.

## Files

| File | Purpose |
|------|---------|
| `zvvm_fp_gemm_main.c` | The FP32 GEMM kernel and a small `main` with 4x4 all-ones data. |
| `zvvm_start.S`        | Bare-metal `_start` plus `tohost` / `fromhost` HTIF mailbox (copy of the integer example's startup — `mstatus.VS`/`FS` are set to Dirty so vector + FP instructions don't trap). |
| `zvvm_link.ld`        | Linker script: code at `0x80000000`, tohost page-aligned. |

## Geometry

The kernel is pinned to SEW=32, LMUL=1, lambda=2 — the same (SEW, LMUL,
lambda) tuple as the integer GEMM example, and the only combination the
Phase-2 Zvvfmm builtin covers:

* `K_eff = lambda * LMUL = 2`
* On a `VLEN=128` implementation, `EMUL_C = VLEN / (SEW * lambda^2) = 1`,
  which matches the IR ISel patterns. Different VLEN / lambda choices
  will need broader builtin coverage (see the deferred Phase-2 tasks).

## Build

From the `build/` directory of this branch's checkout:

```sh
EXAMPLE=../examples/zvvm-fp-gemm

bin/clang -cc1 -triple riscv64 \
  -target-feature +m -target-feature +a -target-feature +f -target-feature +d \
  -target-feature +c -target-feature +v \
  -target-feature +experimental-zvvmm \
  -target-feature +experimental-zvvmtls \
  -target-feature +experimental-zvvfmm \
  -target-feature +zvl128b \
  -O2 -emit-obj -nostdsysteminc \
  -internal-isystem lib/clang/22/include \
  $EXAMPLE/zvvm_fp_gemm_main.c -o zvvm_fp_gemm.o

bin/clang -cc1as -triple riscv64 \
  -target-feature +m -target-feature +a -target-feature +f -target-feature +d \
  -target-feature +c -target-feature +v \
  -target-feature +experimental-zvvmm \
  -target-feature +experimental-zvvmtls \
  -target-feature +experimental-zvvfmm \
  -target-feature +zvl128b \
  -filetype obj $EXAMPLE/zvvm_start.S -o zvvm_fp_start.o

bin/ld.lld -T $EXAMPLE/zvvm_link.ld --no-relax \
  zvvm_fp_start.o zvvm_fp_gemm.o -o zvvm_fp_gemm.elf
```

The resulting `zvvm_fp_gemm.elf` (~10 KB) is fully self-contained: all
Zvvm / Zvvfmm instructions are natively encoded in the `.text` section,
so no Zvvm-aware toolchain is needed at simulation time.

## Run on a Zvvm-patched spike

```sh
spike --isa=rv64gcv_zvvmm_zvvmtls_zvvfmm zvvm_fp_gemm.elf
echo $?     # expect 7
```

Adjust `--isa=` to whatever exact extension names your spike build
accepts.

## Vtype that gets programmed

Same shape as the integer example: `vtype = 0x2000_0000_0000_00D0`, which
decodes as:

| Field         | Value | Meaning      |
|---------------|-------|--------------|
| `lambda[2:0]` | `0b010` (bits 62:60) | L2 |
| `bs`          | 0     | n/a (no microscaling) |
| `altfmt_A`    | 0     | A is fp32 (not the alternate 16-bit format) |
| `altfmt_B`    | 0     | B is fp32 |
| `vma`         | 1     | mask-agnostic |
| `vta`         | 1     | tail-agnostic |
| `vsew`        | 0b010 | e32 |
| `vlmul`       | 0b000 | m1 |

## Symbol map

| Symbol         | Address    |
|----------------|------------|
| `_start`       | `0x80000000` |
| `gemm_f32`     | `0x80000034` |
| `main`         | `0x800000d2` |
| `tohost`       | `0x80001000` |
| `fromhost`     | `0x80001008` |
| `__stack_top`  | `0x800050d0` |
