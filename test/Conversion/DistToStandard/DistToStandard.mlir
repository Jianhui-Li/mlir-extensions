// RUN: imex-opt --split-input-file --convert-dist-to-standard %s -verify-diagnostics -o -| FileCheck %s

// -----
module {
    "dist.runtime_prototypes"() : () -> ()
    func.func @test_nprocs(%arg0: index) -> index {
        %1 = "dist.nprocs"(%arg0) : (index) -> index
        return %1 : index
    }
}
// CHECK-LABEL: func.func private @_idtr_nprocs(index) -> index
// CHECK-LABEL: func.func private @_idtr_prank(index) -> index
// CHECK-LABEL: func.func private @_idtr_reduce_all(index, index, index, index, i32, i32)
// CHECK-LABEL: func.func private @_idtr_rebalance(index, index, index, index, index, index, i32, index, index, index, index)
// CHECK-LABEL: func.func @test_nprocs(%arg0: index) -> index {
// CHECK: @_idtr_nprocs(%arg0)

// -----
module {
    "dist.runtime_prototypes"() : () -> ()
    func.func @test_prank(%arg0: index) -> index {
        %1 = "dist.prank"(%arg0) : (index) -> index
        return %1 : index
    }
}
// CHECK-LABEL: func.func private @_idtr_nprocs(index) -> index
// CHECK-LABEL: func.func private @_idtr_prank(index) -> index
// CHECK-LABEL: func.func private @_idtr_reduce_all(index, index, index, index, i32, i32)
// CHECK-LABEL: func.func private @_idtr_rebalance(index, index, index, index, index, index, i32, index, index, index, index)
// CHECK-LABEL: func.func @test_prank(%arg0: index) -> index {
// CHECK: call @_idtr_prank(%arg0)

// -----
module {
    "dist.runtime_prototypes"() : () -> ()
    func.func @test_init_dist_tensor(%pt: !ptensor.ptensor<1 x i64>, %team: index, %gshape: index, %loffs: index) -> !dist.dtensor<<1 x i64>> {
        %1 = "dist.init_dist_tensor"(%pt, %team, %gshape, %loffs) : (!ptensor.ptensor<1 x i64>, index, index, index) -> !dist.dtensor<<1 x i64>>
        return %1 : !dist.dtensor<<1 x i64>>
    }
}
// CHECK-LABEL: func.func @test_init_dist_tensor
// CHECK: memref.store
// CHECK: memref.store

// -----
module {
    "dist.runtime_prototypes"() : () -> ()
    func.func @test_local_partition(%np : index, %prank: index, %shape: index) -> (index, index) {
        %0, %1 = "dist.local_partition"(%np, %prank, %shape) {rank = 1 : i64} : (index, index, index) -> (index, index)
        return %0, %1 : index, index
    }
}
// CHECK-LABEL: func.func @test_local_partition(%arg0: index, %arg1: index, %arg2: index) -> (index, index) {
// CHECK: arith.subi
// CHECK: arith.muli

// -----
module {
    "dist.runtime_prototypes"() : () -> ()
    func.func @test_allreduce(%arg0: memref<i64, strided<[], offset: ?>>) -> memref<i64, strided<[], offset: ?>> {
        %0 = "dist.allreduce"(%arg0) {op = 4 : i32} : (memref<i64, strided<[], offset: ?>>) -> memref<i64, strided<[], offset: ?>>
        return %0 : memref<i64, strided<[], offset: ?>>
    }
}
// CHECK-LABEL: func.func @test_allreduce(%arg0: memref<i64, strided<[], offset: ?>>) -> memref<i64, strided<[], offset: ?>> {
// CHECK: memref.extract_aligned_pointer_as_index
// CHECK: memref.extract_strided_metadata
// CHECK: call @_idtr_reduce_all

// -----
module {
    func.func @test_local_of_slice(%arg0: !dist.dtensor<<1 x i64>>, %c0 : index, %c3 : index) -> (index, index, index) {
        %l_offsets, %l_sizes, %g_offsets = dist.local_of_slice %arg0[%c0] [%c3] [%c3] : !dist.dtensor<<1 x i64>> to (index, index, index)
        return %l_offsets, %l_sizes, %g_offsets : index, index, index
    }
}
// CHECK-LABEL: func.func @test_local_of_slice(%arg0: !ptensor.ptensor<1 x i64>, %arg1: index, %arg2: memref<1xindex>, %arg3: memref<1xindex>, %arg4: index, %arg5: index) -> (index, index, index) {
// CHECK: "ptensor.extract_tensor"(%arg0) : (!ptensor.ptensor<1 x i64>) -> memref<?xi64, strided<[?], offset: ?>>
// CHECK: memref.load
// CHECK: memref.dim
// CHECK: arith.cmpi ult
// CHECK: arith.cmpi ule
// CHECK: arith.select
// CHECK: arith.select
// CHECK: arith.select
// CHECK: [[V1:%.*]] = arith.select
// CHECK: arith.select
// CHECK: [[V2:%.*]] = arith.select
// CHECK: arith.addi
// CHECK: arith.divsi
// CHECK: arith.select
// CHECK: [[V3:%.*]] = arith.select
// CHECK: return [[V1]], [[V2]], [[V3]] : index, index, index

// -----
func.func @test_0d_inout(%arg0: !dist.dtensor<<0 x i64>>, %arg1: !dist.dtensor<<0 x i64>>) -> !dist.dtensor<<0 x i64>> {
  %0 = "dist.local_tensor_of"(%arg0) : (!dist.dtensor<<0 x i64>>) -> !ptensor.ptensor<0 x i64>
  %1 = "dist.local_tensor_of"(%arg1) : (!dist.dtensor<<0 x i64>>) -> !ptensor.ptensor<0 x i64>
  %2 = "ptensor.ewbin"(%0, %1) {op = 23 : i32} : (!ptensor.ptensor<0 x i64>, !ptensor.ptensor<0 x i64>) -> !ptensor.ptensor<0 x i64>
  %3 = "dist.team_of"(%arg0) : (!dist.dtensor<<0 x i64>>) -> index
  %4 = "dist.init_dist_tensor"(%2, %3) : (!ptensor.ptensor<0 x i64>, index) -> !dist.dtensor<<0 x i64>>
  return %4 : !dist.dtensor<<0 x i64>>
}
// CHECK-LABEL: func.func @test_0d_inout(%arg0: !ptensor.ptensor<0 x i64>, %arg1: index, %arg2: !ptensor.ptensor<0 x i64>, %arg3: index) -> (!ptensor.ptensor<0 x i64>, index) {
// CHECK: [[V1:%.*]] = "ptensor.ewbin"
// CHECK: return [[V1]], %arg1 : !ptensor.ptensor<0 x i64>, index

// -----
module {
    "dist.runtime_prototypes"() : () -> ()
    func.func @test_rebalance(%arg0: !dist.dtensor<<2 x i64>>) -> !dist.dtensor<<2 x i64>> {
    %0 = "dist.rebalance"(%arg0) : (!dist.dtensor<<2 x i64>>) -> !dist.dtensor<<2 x i64>>
    return %0 : !dist.dtensor<<2 x i64>>
    }
}
// CHECK-LABEL: func.func @test_rebalance(%arg0: !ptensor.ptensor<2 x i64>, %arg1: index, %arg2: memref<2xindex>, %arg3: memref<2xindex>) -> (!ptensor.ptensor<2 x i64>, index, memref<2xindex>, memref<2xindex>) {
// CHECK: [[V1:%.*]] = "ptensor.extract_tensor"(%arg0)
// CHECK: memref.extract_strided_metadata [[V1]]
// CHECK: memref.extract_aligned_pointer_as_index [[V1]]
// CHECK: memref.load %arg2[
// CHECK: memref.load %arg2[
// CHECK: memref.load %arg3[
// CHECK: memref.load %arg3[
// CHECK: memref.alloc() {alignment = 8 : i64} : memref<2xindex>
// CHECK: memref.store
// CHECK: memref.store
// CHECK: memref.extract_aligned_pointer_as_index
// CHECK: memref.alloc() {alignment = 8 : i64} : memref<2xindex>
// CHECK: memref.store
// CHECK: memref.store
// CHECK: memref.extract_aligned_pointer_as_index
// CHECK: memref.alloc() {alignment = 8 : i64} : memref<2xindex>
// CHECK: memref.store
// CHECK: memref.store
// CHECK: memref.extract_aligned_pointer_as_index
// CHECK: memref.alloc() {alignment = 8 : i64} : memref<2xindex>
// CHECK: memref.store
// CHECK: memref.store
// CHECK: memref.extract_aligned_pointer_as_index
// CHECK: call @_idtr_nprocs(%arg1) : (index) -> index
// CHECK: call @_idtr_prank(%arg1) : (index) -> index
// CHECK: ptensor.create
// CHECK: "ptensor.extract_tensor"(
// CHECK: memref.extract_aligned_pointer_as_index
// CHECK: call @_idtr_rebalance
