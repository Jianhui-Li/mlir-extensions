// RUN: imex-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s | imex-opt | FileCheck %s
// RUN: imex-opt -mlir-print-op-generic %s | imex-opt | FileCheck %s

// -----

module {
    "dist.runtime_prototypes"() : () -> ()
}
// CHECK-LABEL: "dist.runtime_prototypes"() : () -> ()

// -----
func.func @test_nprocs(%arg0: index) -> index {
    %1 = "dist.nprocs"(%arg0) : (index) -> index
    return %1 : index
}
// CHECK-LABEL: func.func @test_nprocs(%arg0: index) -> index {
// CHECK-NEXT: "dist.nprocs"(%arg0) : (index) -> index

// -----
func.func @test_prank(%arg0: index) -> index {
    %1 = "dist.prank"(%arg0) : (index) -> index
    return %1 : index
}
// CHECK-LABEL: func.func @test_prank(%arg0: index) -> index {
// CHECK-NEXT: "dist.prank"(%arg0) : (index) -> index

// -----
func.func @test_init_dist_tensor(%pt: !ptensor.ptensor<1 x i64>, %team: i64, %gshape: index, %loffs: index) -> !dist.dtensor<<1 x i64>, true> {
    %1 = "dist.init_dist_tensor"(%pt, %team, %gshape, %loffs) : (!ptensor.ptensor<1 x i64>, i64, index, index) -> !dist.dtensor<<1 x i64>, true>
    return %1 : !dist.dtensor<<1 x i64>, true>
}
// CHECK-LABEL: func.func @test_init_dist_tensor(%arg0: !ptensor.ptensor<1 x i64>, %arg1: i64, %arg2: index, %arg3: index) -> !dist.dtensor<<1 x i64>, true> {
// CHECK-NEXT: dist.init_dist_tensor

// -----
func.func @test_extract_from_dist(%arg0: !dist.dtensor<<1 x i64>, true>) -> index {
    %1 = "dist.global_shape_of"(%arg0) : (!dist.dtensor<<1 x i64>, true>) -> index
    %2 = "dist.local_tensor_of"(%arg0) : (!dist.dtensor<<1 x i64>, true>) -> !ptensor.ptensor<1 x i64>
    %3 = "dist.local_offsets_of"(%arg0) : (!dist.dtensor<<1 x i64>, true>) -> index
    %4 = "dist.team_of"(%arg0) : (!dist.dtensor<<1 x i64>, true>) -> index
    return %4 : index
}
// CHECK-LABEL: func.func @test_extract_from_dist(%arg0: !dist.dtensor<<1 x i64>, true>) -> index {
// CHECK-NEXT: "dist.global_shape_of"(%arg0) : (!dist.dtensor<<1 x i64>, true>) -> index
// CHECK-NEXT: "dist.local_tensor_of"(%arg0) : (!dist.dtensor<<1 x i64>, true>) -> !ptensor.ptensor<1 x i64>
// CHECK-NEXT: "dist.local_offsets_of"(%arg0) : (!dist.dtensor<<1 x i64>, true>) -> index
// CHECK-NEXT: "dist.team_of"(%arg0) : (!dist.dtensor<<1 x i64>, true>) -> index

// -----
func.func @test_local_partition(%np : index, %prank: index, %shape: index) -> (index, index) {
    %0, %1 = "dist.local_partition"(%np, %prank, %shape) {rank = 1 : i64} : (index, index, index) -> (index, index)
    return %0, %1 : index, index
}
// CHECK-LABEL: func.func @test_local_partition(%arg0: index, %arg1: index, %arg2: index) -> (index, index) {
// CHECK-NEXT: "dist.local_partition"(%arg0, %arg1, %arg2) {rank = 1 : i64} : (index, index, index) -> (index, index)

// -----
func.func @test_allreduce(%arg0: memref<i64>) -> memref<i64> {
    %0 = "dist.allreduce"(%arg0) {op = 4 : i32} : (memref<i64>) -> memref<i64>
    return %0 : memref<i64>
}
// CHECK-LABEL: func.func @test_allreduce(%arg0: memref<i64>) -> memref<i64> {
// CHECK-NEXT: "dist.allreduce"(%arg0) {op = 4 : i32} : (memref<i64>) -> memref<i64>

// -----
func.func @test_local_of_slice(%arg0: !dist.dtensor<<1 x i64>, true>) -> (index, index, index) {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %l_offsets, %l_sizes, %g_offsets = dist.local_of_slice %arg0[%c0] [%c3] [%c3] : !dist.dtensor<<1 x i64>, true> to (index, index, index)
    return %l_offsets, %l_sizes, %g_offsets : index, index, index
}
// CHECK-LABEL: @test_local_of_slice
// CHECK: [[C1:%.*]], [[C2:%.*]], [[C3:%.*]] = dist.local_of_slice
// CHECK: return [[C1]], [[C2]], [[C3]]


// -----
func.func @test_rebalance(%arg0: !dist.dtensor<<1 x i64>, false>) -> (!dist.dtensor<<1 x i64>, true>) {
    %0 = "dist.rebalance"(%arg0) : (!dist.dtensor<<1 x i64>, false>) -> !dist.dtensor<<1 x i64>, true>
    return %0 : !dist.dtensor<<1 x i64>, true>
}
// CHECK-LABEL: @test_rebalance
// CHECK: [[C1:%.*]] = "dist.rebalance"
// CHECK: return [[C1]]
