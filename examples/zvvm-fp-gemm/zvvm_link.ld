/* Minimal linker script for bare-metal spike (HTIF tohost exit). */

OUTPUT_ARCH(riscv)
ENTRY(_start)

SECTIONS {
  . = 0x80000000;

  .text.init : { *(.text.init) }
  .text      : { *(.text*) }
  .rodata    : ALIGN(8) { *(.rodata*) *(.srodata*) }

  /* Spike's HTIF looks up the `tohost` / `fromhost` ELF symbols; the
     mailbox section must be page-aligned so spike can mark it MMIO.
     The actual tohost / fromhost OBJECT symbols are defined in start.S
     so they have the type and size spike expects. */
  .tohost ALIGN(0x1000) : { *(.tohost) }

  .data : ALIGN(8) {
    __global_pointer$ = . + 0x800;
    *(.data*) *(.sdata*)
  }
  .bss : ALIGN(8) {
    *(.bss*) *(.sbss*) *(COMMON)
  }

  . = ALIGN(16);
  . = . + 0x4000;     /* 16 KiB stack */
  __stack_top = .;
}
