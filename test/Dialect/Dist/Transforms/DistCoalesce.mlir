// RUN: imex-opt --split-input-file --dist-coalesce %s -verify-diagnostics -o -| FileCheck %s

// -----
module {
  "dist.runtime_prototypes"() : () -> ()
  func.func @test_coalesce(%arg0: i64, %arg1: i64, %arg2: i64) -> !dist.dtensor<<1 x i64>, true> {
    %c0 = arith.constant 0 : index
    %0 = ptensor.arange %c0 %c0 %c0   : (index, index, index) -> !ptensor.ptensor<1 x i64>
    %1 = "dist.init_dist_tensor"(%0, %c0, %c0, %c0) : (!ptensor.ptensor<1 x i64>, index, index, index) -> !dist.dtensor<<1 x i64>, false>
    %5 = "dist.init_dist_tensor"(%0, %c0, %c0, %c0) : (!ptensor.ptensor<1 x i64>, index, index, index) -> !dist.dtensor<<1 x i64>, false>
    %2 = "dist.rebalance"(%1) : (!dist.dtensor<<1 x i64>, false>) -> !dist.dtensor<<1 x i64>, true>
    %3 = "dist.init_dist_tensor"(%0, %c0, %c0, %c0) : (!ptensor.ptensor<1 x i64>, index, index, index) -> !dist.dtensor<<1 x i64>, false>
    %4 = "dist.rebalance"(%3) : (!dist.dtensor<<1 x i64>, false>) -> !dist.dtensor<<1 x i64>, true>
    %6 = "dist.rebalance"(%5) : (!dist.dtensor<<1 x i64>, false>) -> !dist.dtensor<<1 x i64>, true>
    return %6 : !dist.dtensor<<1 x i64>, true>
  }
}
// CHECK-LABEL: func.func @test_coalesce
// CHECK: dist.init_dist_tensor
// CHECK-NEXT: dist.init_dist_tensor
// CHECK-NEXT: dist.rebalance
// CHECK-NEXT: dist.init_dist_tensor
// CHECK-NEXT: dist.rebalance
// CHECK-NEXT: return
