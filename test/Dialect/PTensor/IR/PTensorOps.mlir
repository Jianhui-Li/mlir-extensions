// RUN: imex-opt %s | sed s/true\>/1\>/g | FileCheck %s
// Verify the printed output can be parsed.
// RUN: imex-opt %s | sed s/true\>/1\>/g | imex-opt | FileCheck %s
// RUN: imex-opt -mlir-print-op-generic %s |  sed s/true\>/1\>/g | imex-opt | FileCheck %s

// FIXME sed above, for using 1 instead of true

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
// CHECK-NEXT: ptensor.subview %arg0[[[C0]]] [[[C1]]] [[[C1]]] : !ptensor.ptensor<?xi64> to !ptensor.ptensor<?xi64>

// -----
func.func @test_insert_slice(%arg0: !ptensor.ptensor<?xi64>, %arg1: !ptensor.ptensor<?xi64>) -> !ptensor.ptensor<?xi64> {
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    ptensor.insert_slice %arg1 into %arg0[%c0] [%c3] [%c3] : !ptensor.ptensor<?xi64> into !ptensor.ptensor<?xi64>
    return %arg0 : !ptensor.ptensor<?xi64>
}
// CHECK-LABEL: @test_insert_slice
// CHECK-NEXT: [[C0:%.*]] = arith.constant
// CHECK-NEXT: [[C1:%.*]] = arith.constant
// CHECK-NEXT: ptensor.insert_slice %arg1 into %arg0[[[C0]]] [[[C1]]] [[[C1]]] : !ptensor.ptensor<?xi64> into !ptensor.ptensor<?xi64>

// -----
func.func @test_linspace(%arg0: si64, %arg1: si64, %arg2: si64) -> !ptensor.ptensor<?xi64> {
    %0 = ptensor.linspace %arg0 %arg1 %arg2 false : (si64, si64, si64) -> !ptensor.ptensor<?xi64>
    return %0 : !ptensor.ptensor<?xi64>
}
// CHECK-LABEL: @test_linspace
// CHECK-NEXT: ptensor.linspace %arg0 %arg1 %arg2 false : (si64, si64, si64) -> !ptensor.ptensor<?xi64>

func.func @test_create(%arg0: index, %arg1: index, %arg2: index, %arg3: i64) -> !ptensor.ptensor<?x?x?xf64> {
    %0 = ptensor.create %arg0, %arg1, %arg2 {dtype = 0 : i8} : (index, index, index) -> !ptensor.ptensor<?x?x?xf64>
    return %0 : !ptensor.ptensor<?x?x?xf64>
}
// CHECK-LABEL: @test_create
// CHECK: %arg0, %arg1, %arg2 {dtype = 0 : i8} : (index, index, index) -> !ptensor.ptensor<?x?x?xf64>

func.func @test_create2(%arg0: index, %arg1: index, %arg2: index, %arg3: i64) -> !ptensor.ptensor<?x?x?xi64> {
    %0 = ptensor.create %arg0, %arg1, %arg2 value %arg3 device %arg3 team %arg3 {dtype = 2 : i8} : (index, index, index, i64, i64, i64) -> !ptensor.ptensor<?x?x?xi64>
    return %0 : !ptensor.ptensor<?x?x?xi64>
}
// CHECK-LABEL: @test_create2
// CHECK: ptensor.create %arg0, %arg1, %arg2 value %arg3 device %arg3 team %arg3 {dtype = 2 : i8} : (index, index, index, i64, i64, i64) -> !ptensor.ptensor<?x?x?xi64>

// -----
func.func @test_reshape(%arg0: index) -> !ptensor.ptensor<?x?xi64> {
    %0 = ptensor.create %arg0 {dtype = 2 : i8} : (index) -> !ptensor.ptensor<?xi64>
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %1 = "ptensor.reshape"(%0, %c0, %c3) : (!ptensor.ptensor<?xi64>, index, index) -> !ptensor.ptensor<?x?xi64>
    return %1 : !ptensor.ptensor<?x?xi64>
}
// CHECK-LABEL: @test_reshape
// CHECK: ptensor.create
// CHECK: ptensor.reshape
// CHECK-SAME: -> !ptensor.ptensor<?x?xi64>

// -----
func.func @test_ewbin(%arg0: !ptensor.ptensor<?xi64>) -> !ptensor.ptensor<?xi64> {
    %0 = "ptensor.ewbin"(%arg0, %arg0) {op = 0 : i32} : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<?xi64>) -> !ptensor.ptensor<?xi64>
    return %0 : !ptensor.ptensor<?xi64>
}
// CHECK-LABEL: @test_ewbin
// CHECK-NEXT: "ptensor.ewbin"(%arg0, %arg0) {op = 0 : i32} : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<?xi64>) -> !ptensor.ptensor<?xi64>

// -----
func.func @test_reduction(%arg0: !ptensor.ptensor<?xi64>) -> si64 {
    %0 = "ptensor.reduction"(%arg0) {op = 4 : i32} : (!ptensor.ptensor<?xi64>) -> !ptensor.ptensor<si64>
    %1 = builtin.unrealized_conversion_cast %0 : !ptensor.ptensor<si64> to si64
    return %1 : si64
}
// CHECK-LABEL: @test_reduction
// CHECK-NEXT: "ptensor.reduction"(%arg0) {op = 4 : i32} : (!ptensor.ptensor<?xi64>) -> !ptensor.ptensor<si64>

// -----
func.func @test_dim(%arg0: !ptensor.ptensor<?xi64>) -> index {
    %c0 = arith.constant 0 : index
    %1 = ptensor.dim %arg0 %c0 : !ptensor.ptensor<?xi64> -> index
    return %1 : index
}
// CHECK-LABEL: func.func @test_dim
// CHECK: [[V0:%.*]] = ptensor.dim
// CHECK-NEXT: return [[V0]] : index
