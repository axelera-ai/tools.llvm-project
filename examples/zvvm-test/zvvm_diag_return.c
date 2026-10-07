// Diagnostic 1: bare runtime smoke test.
// If `spike --isa=... zvvm_diag_return.elf; echo $?` prints 42 then the
// _start / tohost / linker-script plumbing is fine and any further hang
// is a Zvvm semantics issue, not a build issue.
int main(void) { return 42; }
