// RUN: llvm-tblgen -gen-intrinsic-impl -I %p/../../include -DTEST_INTRINSICS_SUPPRESS_DEFS %s | FileCheck %s

// IntrWrite<[ArgMem]> + IntrRead<[InaccessibleMem]> on top of
// IntrInaccessibleMemOrArgMemOnly gives "argmem: write, inaccessiblemem: read"
// (e.g. a store that also reads hidden target state).

include "llvm/IR/Intrinsics.td"

def int_test_store_reads_state
    : DefaultAttrsIntrinsic<[], [llvm_ptr_ty],
                            [IntrInaccessibleMemOrArgMemOnly,
                             IntrRead<[InaccessibleMem]>,
                             IntrWrite<[ArgMem]>]>;

// CHECK: case 0: // llvm.test.store.reads.state
// CHECK: // ArgMem: Mod, InaccessibleMem: Ref, ErrnoMem: NoModRef, Other: NoModRef, TargetMem0: NoModRef, TargetMem1: NoModRef
