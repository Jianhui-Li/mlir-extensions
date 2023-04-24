// RUN: imex-opt --split-input-file --dist-coalesce %s -verify-diagnostics -o -| FileCheck %s

// -----
module {
  "dist.runtime_prototypes"() : () -> ()
  func.func @test_coalesce(%arg0: i64, %arg1: i64, %arg2: i64) -> !dist.dtensor<<?xi64>> {
    %c0 = arith.constant 0 : index
    %0 = ptensor.linspace %c0 %c0 %c0 false : (index, index, index) -> !ptensor.ptensor<?xi64>
    %1 = dist.init_dist_tensor %0 %c0 0 %c0 offsets %c0 : !ptensor.ptensor<?xi64>, index, index, index to !dist.dtensor<<?xi64>>
    %5 = dist.init_dist_tensor %0 %c0 0 %c0 offsets %c0 : !ptensor.ptensor<?xi64>, index, index, index to !dist.dtensor<<?xi64>>
    %2 = dist.repartition %1 : !dist.dtensor<<?xi64>> to !dist.dtensor<<?xi64>>
    %3 = dist.init_dist_tensor %0 %c0 0 %c0 offsets %c0 : !ptensor.ptensor<?xi64>, index, index, index to !dist.dtensor<<?xi64>>
    %4 = dist.repartition %3 : !dist.dtensor<<?xi64>> to !dist.dtensor<<?xi64>>
    %6 = dist.repartition %5 : !dist.dtensor<<?xi64>> to !dist.dtensor<<?xi64>>
    return %6 : !dist.dtensor<<?xi64>>
  }
}
// CHECK-LABEL: func.func @test_coalesce
// CHECK: ptensor.linspace
// CHECK-NEXT: dist.init_dist_tensor
// CHECK: dist.init_dist_tensor
// CHECK-NEXT: "dist.team_of"
// CHECK-NEXT: "dist.nprocs"
// CHECK-NEXT: "dist.prank"
// CHECK-NEXT: dist.repartition
// CHECK-NEXT: dist.init_dist_tensor
// CHECK-NEXT: "dist.team_of"
// CHECK-NEXT: "dist.nprocs"
// CHECK-NEXT: "dist.prank"
// CHECK-NEXT: dist.repartition
// CHECK-NEXT: dist.repartition
// CHECK-NEXT: return
