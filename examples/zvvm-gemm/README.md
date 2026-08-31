# Zvvm GEMM example

A minimal end-to-end RV64 program that exercises the RISC-V Integrated
Matrix Extension (Zvvm) Clang builtins added on this branch. The kernel
performs `C = A * B` for a 4x4 integer matrix using `vsetvl.matrix`,
`vmtl.v`, `vmmacc.vv`, and `vmts.v`. `main` runs the kernel on all-ones
input and returns `C[0][0]`, which should be `K = 4`.

The build target is bare-metal spike via the HTIF `tohost` exit
convention — no `pk`, no libc, no sysroot required.

## Files

| File | Purpose |
|------|---------|
| `zvvm_gemm_main.c` | The GEMM kernel and a small `main` with 4x4 all-ones data. |
| `zvvm_start.S`     | Bare-metal `_start` plus `tohost` / `fromhost` HTIF mailbox. |
| `zvvm_link.ld`     | Linker script: code at `0x80000000`, tohost page-aligned. |

## Geometry

The kernel is currently pinned to SEW=32, LMUL=1, lambda=2 — the only
combination the Phase-2 builtins cover. With those settings:

* `K_eff = lambda * LMUL = 2`
* On a `VLEN=128` implementation, `EMUL_C = VLEN / (SEW * lambda^2) = 1`,
  which matches the IR ISel patterns. Different VLEN / lambda choices
  will need broader builtin coverage (see the deferred Phase-2 tasks).

## Build

From the `build/` directory of this branch's checkout:

```sh
EXAMPLE=../examples/zvvm-gemm

bin/clang -cc1 -triple riscv64 \
  -target-feature +m -target-feature +a -target-feature +f -target-feature +d \
  -target-feature +c -target-feature +v \
  -target-feature +experimental-zvvi32mm -target-feature +experimental-zvvmtls \
  -target-feature +zvl128b \
  -O2 -emit-obj -nostdsysteminc \
  -internal-isystem lib/clang/22/include \
  $EXAMPLE/zvvm_gemm_main.c -o zvvm_gemm.o

bin/clang -cc1as -triple riscv64 \
  -target-feature +m -target-feature +a -target-feature +f -target-feature +d \
  -target-feature +c -target-feature +v \
  -target-feature +experimental-zvvi32mm -target-feature +experimental-zvvmtls \
  -target-feature +zvl128b \
  -filetype obj $EXAMPLE/zvvm_start.S -o zvvm_start.o

bin/ld.lld -T $EXAMPLE/zvvm_link.ld --no-relax \
  zvvm_start.o zvvm_gemm.o -o zvvm_gemm.elf
```

The resulting `zvvm_gemm.elf` (~11 KB) is fully self-contained: all
Zvvm instructions are natively encoded in the `.text` section, so no
Zvvm-aware toolchain is needed at simulation time.

## Run on a Zvvm-patched spike

```sh
spike --isa=rv64gcv_zvvmm_zvvi32mm_zvvmtls zvvm_gemm.elf
echo $?     # expect 4
```

Adjust `--isa=` to whatever exact extension names your spike build
accepts.

## Vtype that gets programmed

The matrix vsetvl materializes `vtype = 0x2000_0000_0000_00D0`, which
decodes as:

| Field         | Value | Meaning      |
|---------------|-------|--------------|
| `lambda[2:0]` | `0b010` (bits 62:60) | L2 |
| `bs`          | 0     | n/a (no microscaling) |
| `altfmt_A`    | 0     | signed A |
| `altfmt_B`    | 0     | signed B |
| `vma`         | 1     | mask-agnostic |
| `vta`         | 1     | tail-agnostic |
| `vsew`        | 0b010 | e32 |
| `vlmul`       | 0b000 | m1 |

## Symbol map

| Symbol         | Address    |
|----------------|------------|
| `_start`       | `0x80000000` |
| `gemm_i32`     | `0x8000002a` |
| `main`         | `0x800000c8` |
| `tohost`       | `0x80001000` |
| `fromhost`     | `0x80001008` |
| `__stack_top`  | `0x800050d0` |
