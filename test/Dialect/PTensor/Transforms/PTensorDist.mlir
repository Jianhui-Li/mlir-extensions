// RUN: imex-opt --split-input-file --ptensor-dist %s -verify-diagnostics -o -| FileCheck %s

module {
    "dist.runtime_prototypes"() : () -> ()
    func.func @test_linspace(%arg0: i64, %arg1: i64, %arg2: i64) -> i64 {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c3 = arith.constant 3 : index
        %v = arith.constant 55 : i64
        %s = arith.index_cast %arg0 : i64 to index
        %0 = ptensor.linspace %arg0 %arg1 %arg2 false team %c1 : (i64, i64, i64, index) -> !ptensor.ptensor<?xi64>
        %1 = ptensor.create %s value %v team %c1 {dtype = 2 : i8} : (index, i64, index) -> !ptensor.ptensor<?xi64>
        %10 = ptensor.subview %0[%c0][%c3][%c3] : !ptensor.ptensor<?xi64> to !ptensor.ptensor<?xi64>
        %20 ="ptensor.ewbin"(%10, %1) {op = 0 : i32} : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<?xi64>) -> !ptensor.ptensor<?xi64>
        %21 = "ptensor.reduction"(%20) {op = 4 : i32} : (!ptensor.ptensor<?xi64>) -> !ptensor.ptensor<i64>
        %30 = builtin.unrealized_conversion_cast %21 : !ptensor.ptensor<i64> to i64
        return %30 : i64
    }
// CHECK-LABEL: func.func @test_linspace
// CHECK: arith.constant
// CHECK: arith.constant
// CHECK: arith.constant
// CHECK: arith.constant
// CHECK: arith.constant
// CHECK: arith.index_cast
// CHECK: arith.index_cast
// CHECK: arith.sitofp
// CHECK: arith.sitofp
// CHECK: arith.index_cast
// CHECK: arith.sitofp
// CHECK: arith.subf
// CHECK: arith.divf
// CHECK: "dist.nprocs"
// CHECK: "dist.prank"
// CHECK: "dist.local_partition"
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: ptensor.linspace
// CHECK: dist.init_dist_tensor
// CHECK: "dist.nprocs"
// CHECK: "dist.prank"
// CHECK: "dist.local_partition"
// CHECK: ptensor.create
// CHECK: dist.init_dist_tensor
// CHECK: dist.subview
// CHECK: dist.repartition
// CHECK: dist.repartition
// CHECK: "ptensor.ewbin"
// CHECK: "dist.local_tensor_of"
// CHECK: "ptensor.reduction"
// CHECK: "ptensor.extract_tensor"
// CHECK: bufferization.to_memref
// CHECK: "dist.allreduce"
// CHECK: bufferization.to_tensor
// CHECK: "dist.team_of"
// CHECK: "ptensor.init_ptensor"
// CHECK: dist.init_dist_tensor
}
