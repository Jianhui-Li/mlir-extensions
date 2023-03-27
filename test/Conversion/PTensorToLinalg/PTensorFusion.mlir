// RUN: imex-opt --split-input-file --pass-pipeline="builtin.module(convert-ptensor-to-linalg,func.func(tosa-to-linalg,canonicalize,linalg-fuse-elementwise-ops))" %s -verify-diagnostics -o -| FileCheck %s

// -----
func.func @test_binop_fusion_arith(%arg0: !ptensor.ptensor<?xi64>, %arg1: !ptensor.ptensor<?xi64>) -> !ptensor.ptensor<?xi64> {
    %0 ="ptensor.ewbin"(%arg0, %arg1) {op = 0 : i32} : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<?xi64>) -> !ptensor.ptensor<?xi64>
    %1 ="ptensor.ewbin"(%0, %arg0) {op = 21 : i32} : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<?xi64>) -> !ptensor.ptensor<?xi64>
    %2 ="ptensor.ewbin"(%arg0, %1) {op = 0 : i32} : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<?xi64>) -> !ptensor.ptensor<?xi64>
    %3 ="ptensor.ewbin"(%arg1, %2) {op = 21 : i32} : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<?xi64>) -> !ptensor.ptensor<?xi64>
    return %3 : !ptensor.ptensor<?xi64>
}
// CHECK-LABEL: @test_binop_fusion_arith
// CHECK-NEXT: arith.constant
// CHECK-NEXT: tensor.dim
// CHECK-NEXT: tensor.empty
// CHECK-NEXT: linalg.generic
// CHECK-NEXT: bb
// CHECK-NEXT: arith.addi
// CHECK-NEXT: arith.muli
// CHECK-NEXT: arith.addi
// CHECK-NEXT: arith.muli
// CHECK: return %{{[0-9]+}} : tensor<?xi64>

// -----
func.func @test_binop_fusion_tosa(%arg0: !ptensor.ptensor<?xi64>, %arg1: !ptensor.ptensor<?xi64>) -> !ptensor.ptensor<?xi64> {
    %0 ="ptensor.ewbin"(%arg0, %arg1) {op = 4 : i32} : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<?xi64>) -> !ptensor.ptensor<?xi64>
    %1 ="ptensor.ewbin"(%0, %arg0) {op = 2 : i32} : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<?xi64>) -> !ptensor.ptensor<?xi64>
    %2 ="ptensor.ewbin"(%arg0, %1) {op = 4 : i32} : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<?xi64>) -> !ptensor.ptensor<?xi64>
    %3 ="ptensor.ewbin"(%arg1, %2) {op = 2 : i32} : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<?xi64>) -> !ptensor.ptensor<?xi64>
    return %3 : !ptensor.ptensor<?xi64>
}
// CHECK-LABEL: @test_binop_fusion_tosa
// CHECK-NEXT: arith.constant
// CHECK-NEXT: tensor.dim
// CHECK-NEXT: tensor.empty
// CHECK-NEXT: linalg.generic
// CHECK-NEXT: bb
// CHECK-NEXT: arith.ori
// CHECK-NEXT: arith.andi
// CHECK-NEXT: arith.ori
// CHECK-NEXT: arith.andi
// CHECK: return %{{[0-9]+}} : tensor<?xi64>

func.func @test_binop_fusion_bcast(%arg0: !ptensor.ptensor<?x?xi64>, %arg1: !ptensor.ptensor<?xi64>) -> !ptensor.ptensor<?x?xi64> {
    %0 ="ptensor.ewbin"(%arg0, %arg1) {op = 0 : i32} : (!ptensor.ptensor<?x?xi64>, !ptensor.ptensor<?xi64>) -> !ptensor.ptensor<?x?xi64>
    %1 ="ptensor.ewbin"(%0, %arg0) {op = 21 : i32} : (!ptensor.ptensor<?x?xi64>, !ptensor.ptensor<?x?xi64>) -> !ptensor.ptensor<?x?xi64>
    %2 ="ptensor.ewbin"(%arg0, %1) {op = 0 : i32} : (!ptensor.ptensor<?x?xi64>, !ptensor.ptensor<?x?xi64>) -> !ptensor.ptensor<?x?xi64>
    %3 ="ptensor.ewbin"(%arg1, %2) {op = 21 : i32} : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<?x?xi64>) -> !ptensor.ptensor<?x?xi64>
    return %3 : !ptensor.ptensor<?x?xi64>
}
// CHECK-LABEL: @test_binop_fusion_bcast
// CHECK-NEXT: arith.constant
// CHECK-NEXT: arith.constant
// CHECK-NEXT: tensor.dim
// CHECK-NEXT: tensor.dim
// CHECK-NEXT: tensor.empty
// CHECK-NEXT: linalg.generic
// CHECK-NEXT: bb
// CHECK-NEXT: arith.addi
// CHECK-NEXT: arith.muli
// CHECK-NEXT: arith.addi
// CHECK-NEXT: arith.muli
// CHECK: return %{{[0-9]+}} : tensor<?x?xi64>

func.func @test_binop_fusion_bcast2(%arg0: !ptensor.ptensor<?xi64>, %arg1: !ptensor.ptensor<?x?xi64>) -> !ptensor.ptensor<?x?xi64> {
    %0 ="ptensor.ewbin"(%arg0, %arg1) {op = 0 : i32} : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<?x?xi64>) -> !ptensor.ptensor<?x?xi64>
    %1 ="ptensor.ewbin"(%0, %arg0) {op = 21 : i32} : (!ptensor.ptensor<?x?xi64>, !ptensor.ptensor<?xi64>) -> !ptensor.ptensor<?x?xi64>
    %2 ="ptensor.ewbin"(%arg0, %1) {op = 0 : i32} : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<?x?xi64>) -> !ptensor.ptensor<?x?xi64>
    %3 ="ptensor.ewbin"(%arg1, %2) {op = 21 : i32} : (!ptensor.ptensor<?x?xi64>, !ptensor.ptensor<?x?xi64>) -> !ptensor.ptensor<?x?xi64>
    return %3 : !ptensor.ptensor<?x?xi64>
}
// CHECK-LABEL: @test_binop_fusion_bcast2
// CHECK-NEXT: arith.constant
// CHECK-NEXT: arith.constant
// CHECK-NEXT: tensor.dim
// CHECK-NEXT: tensor.dim
// CHECK-NEXT: tensor.empty
// CHECK-NEXT: linalg.generic
// CHECK-NEXT: bb
// CHECK-NEXT: arith.addi
// CHECK-NEXT: arith.muli
// CHECK-NEXT: arith.addi
// CHECK-NEXT: arith.muli
// CHECK: return %{{[0-9]+}} : tensor<?x?xi64>
