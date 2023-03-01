// RUN: imex-opt --split-input-file --ptensor-dist %s -verify-diagnostics -o -| FileCheck %s

module {
    "dist.runtime_prototypes"() : () -> ()
    func.func @test_arange(%arg0: i64, %arg1: i64, %arg2: i64) -> i64 {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c3 = arith.constant 3 : index
        %v = arith.constant 55 : i64
        %s = arith.index_cast %arg0 : i64 to index
        %0 = ptensor.arange %arg0 %arg1 %arg2 team %c1 : (i64, i64, i64, index) -> !ptensor.ptensor<?xi64>
        %1 = ptensor.create %s value %v team %c1 {dtype = 2 : i8} : (index, i64, index) -> !ptensor.ptensor<?xi64>
        %10 = ptensor.subview %0[%c0][%c3][%c3] : !ptensor.ptensor<?xi64> to !ptensor.ptensor<?xi64>
        %20 ="ptensor.ewbin"(%10, %1) {op = 0 : i32} : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<?xi64>) -> !ptensor.ptensor<?xi64>
        %30 = builtin.unrealized_conversion_cast %20 : !ptensor.ptensor<?xi64> to i64
        return %30 : i64
    }
// CHECK-LABEL: func.func @test_arange
// CHECK: arith.constant
// CHECK: arith.constant
// CHECK: arith.constant
// CHECK: arith.constant
// CHECK: arith.cmpi
// CHECK: arith.select
// CHECK: arith.subi
// CHECK: arith.addi
// CHECK: arith.addi
// CHECK: arith.divsi
// CHECK: "dist.nprocs"
// CHECK: "dist.prank"
// CHECK: "dist.local_partition"
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: ptensor.arange
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
}
