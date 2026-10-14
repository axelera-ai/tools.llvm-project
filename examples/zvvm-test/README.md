# Zvvm bring-up diagnostics

Minimal probes for isolating where a Zvvm-aware spike fails. Each `.c` is
intentionally tiny: it does one thing, returns one value, and exits via
the shared HTIF `tohost` mailbox.

These were originally written to debug a real bring-up: a "spike that
supports Zvvm" turned out to hang on `vsetvl` not because of the matrix
extension at all, but because `mstatus.VS` is `Off` at reset, so every
vector instruction trapped illegal and looped at `mtvec = 0`. The fix
(enabling VS in `_start`) lives in `zvvm_start.S` and is shared with the
GEMM example.

## Files

| File | Purpose |
|------|---------|
| `zvvm_diag_return.c`    | `return 42;` — no vector ops. Confirms the runtime plumbing. |
| `zvvm_diag_vsetvli.c`   | Standard `vsetivli a0, 8, e32, m1, ta, ma`. Confirms V is enabled. |
| `zvvm_diag_vsetvl_l0.c` | Register-form `vsetvl` with vtype=`0xD0` (no matrix bits). Isolates the register-form path. |
| `zvvm_diag_vsetvl.c`    | Matrix `vsetvl` with `lambda=2`. Confirms the Zvvm vtype write actually lands. |
| `zvvm_start.S`          | Bare-metal `_start`: stack, gp, `mstatus.VS=Dirty`, then `main`, then exit-via-tohost. |
| `zvvm_link.ld`          | Linker script: code at `0x80000000`, `tohost` page-aligned at `0x80001000`. |

## Build

From the `build/` directory of this branch's checkout:

```sh
EXAMPLE=../examples/zvvm-test

# Assemble the shared startup once.
bin/clang -cc1as -triple riscv64 \
  -target-feature +m -target-feature +a -target-feature +f -target-feature +d \
  -target-feature +c -target-feature +v \
  -target-feature +experimental-zvvmm -target-feature +experimental-zvvmtls \
  -target-feature +zvl128b \
  -filetype obj $EXAMPLE/zvvm_start.S -o zvvm_start.o

# Build each diagnostic.
for stem in zvvm_diag_return zvvm_diag_vsetvli zvvm_diag_vsetvl_l0 zvvm_diag_vsetvl; do
  bin/clang -cc1 -triple riscv64 \
    -target-feature +m -target-feature +a -target-feature +f -target-feature +d \
    -target-feature +c -target-feature +v \
    -target-feature +experimental-zvvmm -target-feature +experimental-zvvmtls \
    -target-feature +zvl128b \
    -O2 -emit-obj -nostdsysteminc \
    -internal-isystem lib/clang/22/include \
    $EXAMPLE/${stem}.c -o ${stem}.o

  bin/ld.lld -T $EXAMPLE/zvvm_link.ld --no-relax \
    zvvm_start.o ${stem}.o -o ${stem}.elf
done
```

## Expected behaviour

On a correctly-functioning Zvvm-patched spike with `VLEN=128`:

```sh
spike --isa=rv64gcv_zvvmm_zvvmtls zvvm_diag_return.elf;    echo $?  # 42
spike --isa=rv64gcv_zvvmm_zvvmtls zvvm_diag_vsetvli.elf;   echo $?  # 8
spike --isa=rv64gcv_zvvmm_zvvmtls zvvm_diag_vsetvl_l0.elf; echo $?  # 8 (vsetvl with vtype=0xD0)
spike --isa=rv64gcv_zvvmm_zvvmtls zvvm_diag_vsetvl.elf;    echo $?  # 4 (vl clamps to VLMAX = VLEN/SEW = 4)
```

Spike's `*** FAILED *** (tohost = N)` line is cosmetic — it labels any
nonzero exit as "failed" by riscv-tests convention. The actual return
code is what matters.

## Failure modes and what they mean

| Diagnostic that misbehaves | Likely cause |
|----------------------------|--------------|
| `diag_return` doesn't exit / returns garbage | Runtime broken: `tohost` symbol type/section wrong, `_start` not setting sp, or linker script placement off. |
| `diag_vsetvli` hangs | Either V is not enabled in spike's build, **or** `mstatus.VS=Off` and `_start` isn't writing it (see the VS-enable block in `zvvm_start.S`). |
| `diag_vsetvl_l0` hangs but `diag_vsetvli` works | Spike's register-form `vsetvl` handler is broken or not wired up. |
| `diag_vsetvl` hangs but `diag_vsetvl_l0` works | Spike's vsetvl accepts standard vtype but chokes on the matrix bits — usually an unterminated WARL clamp loop, or the high-end vtype bits being masked off. |
| `diag_vsetvl` returns 0 | The matrix `vsetvl` set `vill=1`. Lambda value isn't WARL-supported on this spike (try 1, 4, 8). |
| `diag_vsetvl` returns 4 | Working as intended — `vl = min(avl, VLMAX) = min(8, 4) = 4` for `VLEN=128, SEW=e32, LMUL=1`. |

If all four diagnostics pass, the matching `zvvm-gemm` example should also work end-to-end.
