// RUN: %python_executable %imex_runner --requires=l0-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp  \
// RUN:                                       --runner imex-cpu-runner -e main \
// RUN:                                       --entry-point-result=void \
// RUN:                                       --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%levelzero_runtime --filecheck
// RUN: %python_executable %imex_runner --requires=sycl-runtime -i %s --pass-pipeline-file=%p/xegpu-to-func-vc.pp  \
// RUN:                                        --runner imex-cpu-runner -e main \
// RUN:                                        --entry-point-result=void \
// RUN:                                        --shared-libs=%irunner_utils,%mlir_runner_utils,%mlir_c_runner_utils,%sycl_runtime --filecheck
module @gemm attributes {gpu.container_module,
spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
  memref.global "private" constant @__constant_32x32xf16 : memref<32x32xf16> = dense<5.000000e-01>
  memref.global "private" constant @__Bconstant_32x32xf16 : memref<32x32xf16> = dense<1.099610e+00>
  func.func @test(%arg0: memref<32x32xf16>, %arg1: memref<32x32xf16>) -> memref<32x32xf32> {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %memref_0 = gpu.alloc  host_shared () : memref<32x32xf16>
    memref.copy %arg0, %memref_0 : memref<32x32xf16> to memref<32x32xf16>
    %memref_1 = gpu.alloc  host_shared () : memref<32x32xf16>
    memref.copy %arg1, %memref_1 : memref<32x32xf16> to memref<32x32xf16>
    %memref_c = gpu.alloc  host_shared () : memref<32x32xf32>
    gpu.launch_func @test_kernel::@test_kernel blocks in (%c4, %c2, %c1) threads in (%c1, %c1, %c1) args(%memref_0 : memref<32x32xf16>, %memref_1 : memref<32x32xf16>, %memref_c : memref<32x32xf32>)
    %result = memref.alloc() :  memref<32x32xf32>
    memref.copy %memref_c, %result: memref<32x32xf32> to memref<32x32xf32>
    gpu.dealloc  %memref_0 : memref<32x32xf16>
    gpu.dealloc  %memref_1 : memref<32x32xf16>
    gpu.dealloc  %memref_c :memref<32x32xf32>

    return %result : memref<32x32xf32>
  }
  gpu.module @test_kernel {
   gpu.func @test_kernel(%arg0: memref<32x32xf16>, %arg1: memref<32x32xf16>, %arg2: memref<32x32xf32>) kernel attributes {VectorComputeFunctionINTEL, gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 4, 2, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>}{
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c128 = arith.constant 128 : index
      %c8 = arith.constant 8 : index

      %0 = gpu.block_id  x
      %1 = gpu.block_id  y

      %2 = arith.muli %0, %c8 : index
      %3 = arith.muli %1, %c16 : index
      %128 = arith.muli %c8, %c16 : index
      %256 = arith.muli %128, %c2 : index
      %x = arith.muli %256, %0 : index
      %y = arith.muli %128, %1 : index

      %c_index = arith.addi %x, %y : index
      %arg02 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1024], strides: [1] : memref<32x32xf32> to memref<1024xf32>
      %C0 = xegpu.create_nd_tdesc %arg02[%c_index] : memref<1024xf32> -> !xegpu.tensor_desc<128xf32>
      %5 = xegpu.load_nd %C0 : !xegpu.tensor_desc<128xf32> -> vector<128xf32>

      %arg00 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1024], strides: [1] : memref<32x32xf16> to memref<1024xf16>

      %6 = scf.for %arg3 = %c0 to %c32 step %c16 iter_args(%arg4 = %5) -> (vector<128xf32>) {
        %a_index = arith.addi %x, %arg3 : index
        %A0 = xegpu.create_nd_tdesc %arg00[%a_index]: memref<1024xf16> -> !xegpu.tensor_desc<128xf16>
        %A0_val = xegpu.load_nd %A0 : !xegpu.tensor_desc<128xf16> -> vector<128xf16>

        %B0 = xegpu.create_nd_tdesc %arg1[%arg3, %3] {boundary_check = true} : memref<32x32xf16> -> !xegpu.tensor_desc<16x16xf16>
        %B0_val = xegpu.load_nd %B0 {packed} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>

        %A0_cast = vector.shape_cast %A0_val : vector<128xf16> to vector<8x16xf16>

        %dpas0 = xegpu.dpas %A0_cast, %B0_val : vector<8x16xf16>, vector<8x16x2xf16> -> vector<8x16xf32>
        %dpas0_cast = vector.shape_cast %dpas0: vector<8x16xf32> to vector<128xf32>

        scf.yield %dpas0_cast : vector<128xf32>
      }
      xegpu.store_nd %6, %C0 : vector<128xf32>, !xegpu.tensor_desc<128xf32>

      gpu.return
    }
  }
  func.func @main() {
    %0 = memref.get_global @__constant_32x32xf16 : memref<32x32xf16>
    %1 = memref.get_global @__Bconstant_32x32xf16 : memref<32x32xf16>
    %2 = call @test(%0, %1) : (memref<32x32xf16>, memref<32x32xf16>) -> memref<32x32xf32>
    %cast = memref.cast %2 : memref<32x32xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    return
  }
  func.func private @printMemrefF32(memref<*xf32>) attributes {llvm.emit_c_interface}
}

// CHECK: Unranked Memref base@{{(0x)?[-9a-f]*}}
// CHECK-SAME: rank = 2 offset = 0 sizes = [32, 32] strides = [32, 1] data =
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688],
// CHECK: [8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688,   8.79688]]
