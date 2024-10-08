// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @gemm attributes {gpu.container_module} {
  func.func @test(%arg0: memref<8x32xf16>) -> memref<16x32xf16> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  host_shared () : memref<8x32xf16>
    memref.copy %arg0, %memref : memref<8x32xf16> to memref<8x32xf16>
    %memref_1 = gpu.alloc  host_shared () : memref<16x32xf16>
    gpu.launch_func  @test_kernel::@test_copy blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) args(%memref : memref<8x32xf16>, %memref_1 : memref<16x32xf16>)

    gpu.dealloc  %memref : memref<8x32xf16>
    return %memref_1 : memref<16x32xf16>
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_copy(%arg0: memref<8x32xf16>, %arg1: memref<16x32xf16>) kernel attributes {VectorComputeFunctionINTEL, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %0 = xegpu.create_nd_tdesc %arg0[0, 0] : memref<8x32xf16> -> !xegpu.tensor_desc<8x16xf16>
      %1 = xegpu.load_nd %0 {packed} : !xegpu.tensor_desc<8x16xf16> -> vector<4x16x2xf16>
      %2 = vector.shape_cast %1 : vector<4x16x2xf16> to vector<4x32xf16>
      %3 = xegpu.create_nd_tdesc %arg1[0, 0] : memref<16x32xf16> -> !xegpu.tensor_desc<4x32xf16>
      xegpu.store_nd %2, %3 : vector<4x32xf16>, !xegpu.tensor_desc<4x32xf16>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %0 = memref.alloc() : memref<8x32xf16>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index

    scf.for %i = %c0 to %c8 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        %m = arith.muli %i, %c16 : index
        %a = arith.addi %m, %j : index
        %t = index.castu %a : index to i16
        %val = arith.uitofp %t : i16 to f16
        memref.store %val, %0[%i, %j] : memref<8x32xf16>
      }
    }


    %2 = call @test(%0) : (memref<8x32xf16>) -> memref<16x32xf16>

    %cast = memref.cast %2: memref<16x32xf16> to memref<*xf16>

    //CHECK: [0,   16,   1,   17,   2,   18,   3,   19,   4,   20,   5,   21,   6,   22,   7,   23,   8,   24,   9,   25,   10,   26,   11,   27,   12,   28,   13,   29,   14,   30,   15,   31]
    //CHECK: [32,   48,   33,   49,   34,   50,   35,   51,   36,   52,   37,   53,   38,   54,   39,   55,   40,   56,   41,   57,   42,   58,   43,   59,   44,   60,   45,   61,   46,   62,   47,   63]
    //CHECK: [64,   80,   65,   81,   66,   82,   67,   83,   68,   84,   69,   85,   70,   86,   71,   87,   72,   88,   73,   89,   74,   90,   75,   91,   76,   92,   77,   93,   78,   94,   79,   95]
    //CHECK: [96,   112,   97,   113,   98,   114,   99,   115,   100,   116,   101,   117,   102,   118,   103,   119,   104,   120,   105,   121,   106,   122,   107,   123,   108,   124,   109,   125,   110,   126,   111,   127]
    //CHECK-COUNT-12: [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0]
    call @printMemrefF16(%cast): (memref<*xf16>) -> ()
    return
  }

  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
}
