# RFC for XeTile and XeGPU Dialect

## Summary
Lowering GEMM (General matrix multiplication) to an efficient nested loop is a complicated task, with multiple factors determining the efficiency. After decomposing the task into workgroup and further down to subgroup, each subgroup executes a GEMM operation on submatrices. Generating efficient code for GEMM requires a good decomposition that creates enough subtasks to drive high core utilization and large enough subgroup-level submatrix size for code efficiency. On top of this, each hardware has its own recipe for the best code sequence for sub-group level GEMM, which contains both common target-independent techniques and target-specific optimizations. 

This RFC propose XeTile and XeGPU Dialect to support effecitient code generation for Xe GPU. 

## Motivation

To facilitate efficient code generation for GEMM, we introduce two new dialects, XeTile and XeGPU dialects. XeTile dialect supports tile-based programming model and decomposing the GEMM kernel to a large enough tile size at the subgroup level.  User can use the XeTile dialect to build a subgroup-level microkernel that implements batch-reduced gemm, using the best-known recipe for a specific hardware. The recipe at this level includes target-independent optimizations like cooperative prefetch, cooperative load, K-slicing, and software pipelining. Users can further perform optimization like fusing with neighbour operations.  

The XeTile dialect works as the lowest abstraction layer which hides the matrix hardware difference between different GPU micro-architectures through working at tiles with larger size than the underneath hardware support. With the XeTile dialect, the lowering pass can set up hardware 2d block loader to autopad the out-of-boundary access.  XeTile dialect also abstracts out the hard limitations so that it can support arbitrary input matrix sizes.  When the input matrix sizes don’t meet 2d block load requirements, the lowering pass implements 2d tile load using 1d load and scalar load with target-specific recipes. 

XeGPU dialect provides 1:1 mapping to match Xe instructions like DPAS and 2D block load. The matrix size being processed at this level exactly match the hardware instructions or the intrinsic supported by the lower-level GPU compiler.  All the optimization built on top of XeGPU dialect are target-specific. One optimization is to decompose a large contiguous 2D tile to a number of smaller tiles, to help lower-level compiler to provide better register allocation. After lowering to XeGPU dialect, user could estimate the total register size used by the sub-group level GEMM assuming the lower-level compiler does proper register allocation to facilitate the GEMM decomposition. 

## Proposal
A Full and detailed description of proposal.


  ```mlir 
  %tile = XeTile.init_tile %memref,  %base, %offset, %sizes:2, %strides:2:
     memref<64x64xbf16>, index, index, index, index, index
     into tile<64x64xbf16, index, index, index, index, index>
  ```

## Alternative
Proposal needs to mention any alternative that were being considered with pros and cons. Instead of creating a new dialect to introduce intel specific ops, consider alternative  to add specific ops in the existing upstream dialects as extension. For optimization passes, describe alternative optimization approaches.

## Questions
Mention open questions here.
