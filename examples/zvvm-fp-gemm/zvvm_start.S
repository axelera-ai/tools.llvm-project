// Bare-metal entry for spike (HTIF / tohost exit convention).
// spike --isa=rv64gcv_zvvmm_zvvmtls_zvvfmm_zvvfp32mm zvvm_gemm.elf
//
// Spike with HTIF polls the ELF symbol `tohost`. Writing a nonzero value
// terminates the simulation; bit 0 must be 1, the exit code is bits[16:1].
// We pack main()'s i32 return value into bits[16:1].

    .section .text.init, "ax"
    .globl _start
_start:
    // Set up stack and global-pointer. Spike's reset state leaves all gprs
    // zero. Use `lla` (load local address) rather than `la` — the latter
    // resolves to a GOT load (auipc+ld) under cc1as defaults, which would
    // read zero into sp/gp/a1 because the .got isn't populated in a static
    // bare-metal link.
    lla     sp, __stack_top
    .option push
    .option norelax
    lla     gp, __global_pointer$
    .option pop

    // Enable the vector context. mstatus.VS (bits [10:9]) is Off after
    // reset, so every vector instruction — including vsetvli — raises
    // illegal-instruction. We also enable FS (bits [14:13]) since we
    // compile with +f/+d enabled and any spill of an FP register would
    // otherwise trap. Both set to Dirty (0b11) so subsequent state
    // modifications don't get optimised away by the implementation.
    //
    // mstatus.VS = 0b11 ⇒ bits [10:9] = 3   ⇒ 3 << 9  = 0x0600
    // mstatus.FS = 0b11 ⇒ bits [14:13] = 3  ⇒ 3 << 13 = 0x6000
    li      t0, 0x6600
    csrs    mstatus, t0

    call    main

    // Exit: tohost = (main_return << 1) | 1.
    slli    a0, a0, 1
    ori     a0, a0, 1
    lla     a1, tohost
    sd      a0, 0(a1)
1:  j       1b

    // HTIF mailbox. Spike resolves these by ELF symbol; they must be
    // OBJECT symbols of size 8 (NOTYPE / size-0 labels don't satisfy the
    // "tohost and fromhost symbols not in ELF" check).
    .section .tohost, "aw", @progbits
    .align 6
    .globl tohost
    .type tohost, @object
    .size tohost, 8
tohost: .dword 0

    .globl fromhost
    .type fromhost, @object
    .size fromhost, 8
fromhost: .dword 0
