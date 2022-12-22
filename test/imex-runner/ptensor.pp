builtin.module(
    convert-ptensor-to-linalg
    convert-shape-to-std
    arith-expand
    arith-bufferize
    func-bufferize
    func.func(
        empty-tensor-to-alloc-tensor
        scf-bufferize
        tensor-bufferize
        linalg-bufferize
        bufferization-bufferize
        linalg-detensorize
        tensor-bufferize
        finalizing-bufferize
        convert-linalg-to-parallel-loops)
    canonicalize
    fold-memref-alias-ops
    expand-strided-metadata
    lower-affine
    convert-scf-to-cf
    convert-memref-to-llvm
    convert-func-to-llvm
    reconcile-unrealized-casts)
