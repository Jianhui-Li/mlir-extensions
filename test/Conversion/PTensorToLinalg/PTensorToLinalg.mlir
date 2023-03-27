// RUN: imex-opt --split-input-file --convert-ptensor-to-linalg %s -verify-diagnostics -o -| FileCheck %s

// -----
func.func @test_subview(%arg0: !ptensor.ptensor<?xi64>) -> !ptensor.ptensor<?xi64> {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %0 = ptensor.subview %arg0[%c0][%c3][%c3] : !ptensor.ptensor<?xi64> to !ptensor.ptensor<?xi64>
    return %0 : !ptensor.ptensor<?xi64>
}
// CHECK-LABEL: @test_subview
// CHECK-NEXT: [[C0:%.*]] = arith.constant
// CHECK-NEXT: [[C1:%.*]] = arith.constant
// CHECK-NEXT: [[V0:%.*]] = bufferization.to_memref %arg0 : memref<?xi64, strided<[?], offset: ?>>
// CHECK-NEXT: [[S0:%.*]] = memref.subview [[V0]][[[C0]]] [[[C1]]] [[[C1]]] : memref<?xi64, strided<[?], offset: ?>> to memref<?xi64, strided<[?], offset: ?>>
// CHECK-NEXT: [[V1:%.*]] = bufferization.to_tensor [[S0]] : memref<?xi64, strided<[?], offset: ?>>
// CHECK-NEXT: return [[V1]] : tensor<?xi64>

// -----
func.func @test_arange(%arg0: i64, %arg1: i64, %arg2: i64) -> !ptensor.ptensor<?xindex> {
    %0 = ptensor.arange %arg0 %arg1 %arg2 : (i64, i64, i64) -> !ptensor.ptensor<?xindex>
    return %0 : !ptensor.ptensor<?xindex>
}
// CHECK-LABEL: @test_arange
// CHECK: [[C0:%.*]] = arith.select
// CHECK-NEXT: [[C01:%.*]] = arith.subi
// CHECK-NEXT: [[C1:%.*]] = arith.addi
// CHECK-NEXT: [[C2:%.*]] = arith.addi [[C1]], [[C0]] : index
// CHECK: linalg.generic{{.*}}["parallel"]
// CHECK: return %{{[0-9]+}} : tensor<?xindex>

func.func @test_create(%arg0: index, %arg1: index, %arg2: i64) -> !ptensor.ptensor<?x?xi64> {
    %0 = ptensor.create %arg0, %arg1 value %arg2 {dtype = 2 : i8} : (index, index, i64) -> !ptensor.ptensor<?x?xi64>
    return %0 : !ptensor.ptensor<?x?xi64>
}
// CHECK-LABEL: @test_create
// CHECK: tensor.empty
// CHECK: linalg.generic
// CHECK-NEXT: bb
// CHECK-NEXT: linalg.yield
// CHECK-NEXT: } -> tensor<?x?xi64>
// CHECK: return %{{[0-9]+}} : tensor<?x?xi64>

// -----
func.func @test_ewbin(%arg0: !ptensor.ptensor<?xi64>) -> !ptensor.ptensor<?xi64> {
    %0 = "ptensor.ewbin"(%arg0, %arg0) {op = 21 : i32} : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<?xi64>) -> !ptensor.ptensor<?xi64>
    return %0 : !ptensor.ptensor<?xi64>
}
// CHECK-LABEL: #map = affine_map<(d0) -> (d0)>
// CHECK-LABEL: @test_ewbin(
// CHECK: arith.constant
// CHECK: tensor.dim
// CHECK: tensor.empty
// CHECK: linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
// CHECK: arith.muli
// CHECK: return %{{[0-9]+}} : tensor<?xi64>

// -----
func.func @test_ewbin_bcast(%arg0: !ptensor.ptensor<?x?xi64>, %arg1: !ptensor.ptensor<i64>) -> !ptensor.ptensor<?x?xi64> {
    %0 = "ptensor.ewbin"(%arg0, %arg1) {op = 0 : i32} : (!ptensor.ptensor<?x?xi64>, !ptensor.ptensor<i64>) -> !ptensor.ptensor<?x?xi64>
    return %0 : !ptensor.ptensor<?x?xi64>
}
// CHECK-LABEL: #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #map1 = affine_map<(d0, d1) -> ()>
// CHECK-LABEL: @test_ewbin_bcast
// CHECK: arith.constant
// CHECK: tensor.dim
// CHECK: arith.constant
// CHECK: tensor.dim
// CHECK: tensor.empty
// CHECK: linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]}
// CHECK: arith.addi
// CHECK: return %{{[0-9]+}} : tensor<?x?xi64>

// -----
func.func @test_ewbin_3d(%arg0: !ptensor.ptensor<?x?x?xi64>) -> !ptensor.ptensor<?x?x?xi64> {
    %0 = "ptensor.ewbin"(%arg0, %arg0) {op = 0 : i32} : (!ptensor.ptensor<?x?x?xi64>, !ptensor.ptensor<?x?x?xi64>) -> !ptensor.ptensor<?x?x?xi64>
    return %0 : !ptensor.ptensor<?x?x?xi64>
}
// CHECK-LABEL: #map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: @test_ewbin_3d
// CHECK: arith.constant
// CHECK: tensor.dim
// CHECK: arith.constant
// CHECK: tensor.dim
// CHECK: tensor.empty
// CHECK: linalg.generic{{.*}}["parallel", "parallel", "parallel"]
// CHECK: arith.addi
// CHECK: return %{{.+}} : tensor<?x?x?xi64>

// -----
func.func @test_reduction(%arg0: !ptensor.ptensor<?xi64>) -> i64 {
    %0 = "ptensor.reduction"(%arg0) {op = 4 : i32} : (!ptensor.ptensor<?xi64>) -> !ptensor.ptensor<i64>
    %1 = builtin.unrealized_conversion_cast %0 : !ptensor.ptensor<i64> to i64
    return %1 : i64
}
// CHECK-LABEL: @test_reduction
// CHECK: [[C0:%.*]] = linalg.fill
// CHECK: linalg.generic{{.*}}["reduction"]}{{.*}}outs([[C0]]
// CHECK: return %{{.}} : i64

// -----
func.func @test_reduction_3d(%arg0: !ptensor.ptensor<?x?x?xi64>) -> i64 {
    %0 = "ptensor.reduction"(%arg0) {op = 4 : i32} : (!ptensor.ptensor<?x?x?xi64>) -> !ptensor.ptensor<i64>
    %1 = builtin.unrealized_conversion_cast %0 : !ptensor.ptensor<i64> to i64
    return %1 : i64
}
// CHECK-LABEL: @test_reduction_3d
// CHECK: [[C0:%.*]] = linalg.fill
// CHECK: linalg.generic{{.*}}["reduction", "reduction", "reduction"]}{{.*}}outs([[C0]]
// CHECK: return %{{.}} : i64

// -----
func.func @test_insert_slice(%arg0: !ptensor.ptensor<?xi64>, %arg1: !ptensor.ptensor<?xi64>) {
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i3 = arith.constant 3 : index
    ptensor.insert_slice %arg1 into %arg0[%i0] [%i3] [%i1] : !ptensor.ptensor<?xi64> into !ptensor.ptensor<?xi64>
    return
}
// CHECK-LABEL: @test_insert_slice
// CHECK-NEXT: [[C0:%.*]] = arith.constant
// CHECK-NEXT: [[C1:%.*]] = arith.constant
// CHECK-NEXT: [[C3:%.*]] = arith.constant
// CHECK-NEXT: [[V0:%.*]] = bufferization.to_memref %arg1
// CHECK-NEXT: [[V1:%.*]] = bufferization.to_memref %arg0
// CHECK-NEXT: memref.subview [[V1]][[[C0]]] [[[C3]]] [[[C1]]] : memref<?xi64, strided<[?], offset: ?>> to memref<?xi64, strided<[?], offset: ?>>
// CHECK: scf.if
// CHECK: linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}

// -----
func.func @test_load(%arg0: !ptensor.ptensor<?xi64>) -> i64 {
    %i3 = arith.constant 3 : index
    %1 = ptensor.load %arg0 [%i3]  : !ptensor.ptensor<?xi64>
    return %1 : i64
}
// CHECK-LABEL: @test_load
// CHECK-NEXT: [[C0:%.*]] = arith.constant
// CHECK-NEXT: [[V0:%.*]] = tensor.extract %arg0[[[C0]]] : tensor<?xi64>
// CHECK-NEXT: return [[V0]] : i64

// -----
func.func @test_extract_tensor(%arg0: !ptensor.ptensor<?xi64>) -> tensor<?xi64> {
    %0 = "ptensor.extract_tensor"(%arg0) : (!ptensor.ptensor<?xi64>) -> tensor<?xi64>
    return %0 : tensor<?xi64>
}
// CHECK-LABEL: @test_extract_tensor
// CHECK-NEXT: return %arg0 : tensor<?xi64>

// -----
func.func @test_extract_raw_ptr(%arg0: !ptensor.ptensor<?xi64>) -> index {
    %0 = "ptensor.extract_raw_ptr"(%arg0) : (!ptensor.ptensor<?xi64>) -> index
    return %0 : index
}
// CHECK-LABEL: @test_extract_raw_ptr
// CHECK-NEXT: bufferization.to_memref
// CHECK-NEXT: memref.extract_strided_metadata
// CHECK-NEXT: memref.extract_aligned_pointer_as_index
// CHECK: return %{{.}} : index
