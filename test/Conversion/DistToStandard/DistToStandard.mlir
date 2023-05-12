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
// CHECK-LABEL: func.func private @_idtr_reshape(index, index, i32, index, index, index, index, index, index, index, index, index, index)
// CHECK-LABEL: func.func private @_idtr_repartition(index, memref<*xindex>, i32, index, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, memref<*xindex>, index, index)
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
// CHECK-LABEL: func.func @test_prank(%arg0: index) -> index {
// CHECK: call @_idtr_prank(%arg0)

// -----
module {
    "dist.runtime_prototypes"() : () -> ()
    func.func @test_init_dist_tensor(%pt: !ptensor.ptensor<?xi64>, %team: index, %gshape: index, %loffs: index) -> !dist.dtensor<<?xi64>> {
        %1 = dist.init_dist_tensor %pt %team 1 %gshape offsets %loffs : !ptensor.ptensor<?xi64>, index, index, index to !dist.dtensor<<?xi64>>
        return %1 : !dist.dtensor<<?xi64>>
    }
}
// CHECK-LABEL: func.func @test_init_dist_tensor
// CHECK: memref.store
// CHECK: memref.store

// -----
func.func @test_cast(%arg0: !ptensor.ptensor<?xi64>) -> !dist.dtensor<<?xi64>> {
    %1 = "dist.cast"(%arg0) : (!ptensor.ptensor<?xi64>) -> (!dist.dtensor<<?xi64>>)
    return %1 : !dist.dtensor<<?xi64>>
}
// CHECK-LABEL: func.func @test_cast(%arg0
// CHECK: [[C0:%.*]] = arith.constant 0 : index
// CHECK: [[C1:%.*]] = arith.constant 1 : index
// CHECK: [[A0:%.*]] = memref.alloc()
// CHECK: [[C2:%.*]] = arith.constant 0 : index
// CHECK: memref.store
// CHECK: [[A1:%.*]] = memref.alloc() {alignment = 8 : i64} : memref<1xindex>
// CHECK: [[C3:%.*]] = arith.constant 0 : index
// CHECK: memref.store
// CHECK: return %arg0, [[C0]], [[C1]], [[A0]], [[A1]] : !ptensor.ptensor<?xi64>, index, index, memref<1xindex>, memref<1xindex>

func.func @test_cast2(%arg0: i64, %arg1: i64, %arg2: index) -> !dist.dtensor<<?xi64>> {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %l_offsets, %l_shape = "dist.local_partition"(%c2, %c2, %arg2) : (index, index, index) -> (index, index)
    %17 = ptensor.linspace %arg0 %arg1 %arg2 false   : (i64, i64, index) -> !ptensor.ptensor<?xi64>
    %18 = dist.init_dist_tensor %17 %c1 1 %arg2   offsets %l_offsets : !ptensor.ptensor<?xi64>, index, index, index to !dist.dtensor<<?xi64>>
    %19 = "dist.cast"(%18) : (!dist.dtensor<<?xi64>>) -> !dist.dtensor<<?xi64>>
    return %19 : !dist.dtensor<<?xi64>>
}
// CHECK-LABEL: func.func @test_cast2
// CHECK: arith.addi
// CHECK: arith.subi
// CHECK: arith.divsi
// CHECK: [[V0:%.*]] = arith.minsi %arg2
// CHECK: arith.minsi
// CHECK: ptensor.linspace
// CHECK: [[V1:%.*]] = memref.alloc()
// CHECK: memref.store %arg2, [[V1]]
// CHECK: [[V2:%.*]] = memref.alloc()
// CHECK: memref.store [[V0]], [[V2]]
// CHECK: return
// CHECK-SAME: , [[V1]], [[V2]] : !ptensor.ptensor<?xi64>, index, index, memref<1xindex>, memref<1xindex>

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
    func.func @test_local_target_of_slice(%arg0: !dist.dtensor<<?xi64>>, %c0 : index, %c3 : index) -> (index, index) {
        %l_offsets, %l_sizes = dist.local_target_of_slice %arg0[%c0] [%c3] [%c3] : !dist.dtensor<<?xi64>> to index, index
        return %l_offsets, %l_sizes : index, index
    }
}
// CHECK-LABEL: func.func @test_local_target_of_slice(%arg0: !ptensor.ptensor<?xi64>, %arg1: index, %arg2: index, %arg3: memref<1xindex>, %arg4: memref<1xindex>, %arg5: index, %arg6: index) -> (index, index) {
// CHECK: memref.load
// CHECK: "ptensor.extract_tensor"(%arg0) : (!ptensor.ptensor<?xi64>) -> tensor<?xi64>
// CHECK: tensor.dim
// CHECK: arith.muli
// CHECK: arith.select
// CHECK: [[V0:%.*]] = arith.select
// CHECK: scf.if [[V0]] -> (index, index) {
// CHECK: scf.yield %15, %20 : index, index
// CHECK: else
// CHECK: scf.yield %10, %c0_1 : index, index
// CHECK: return

// -----
func.func @test_0d_inout(%arg0: !dist.dtensor<<i64>>, %arg1: !dist.dtensor<<i64>>) -> !dist.dtensor<<i64>> {
  %0 = "dist.local_tensor_of"(%arg0) : (!dist.dtensor<<i64>>) -> !ptensor.ptensor<i64>
  %1 = "dist.local_tensor_of"(%arg1) : (!dist.dtensor<<i64>>) -> !ptensor.ptensor<i64>
  %2 = "ptensor.ewbin"(%0, %1) {op = 23 : i32} : (!ptensor.ptensor<i64>, !ptensor.ptensor<i64>) -> !ptensor.ptensor<i64>
  %3 = "dist.team_of"(%arg0) : (!dist.dtensor<<i64>>) -> index
  %4 = dist.init_dist_tensor %2 %3 1 : !ptensor.ptensor<i64>, index to !dist.dtensor<<i64>>
  return %4 : !dist.dtensor<<i64>>
}
// CHECK-LABEL: func.func @test_0d_inout(%arg0: !ptensor.ptensor<i64>, %arg1: index, %arg2: index, %arg3: !ptensor.ptensor<i64>, %arg4: index, %arg5: index) -> (!ptensor.ptensor<i64>, index, index) {
// CHECK: [[V1:%.*]] = "ptensor.ewbin"
// CHECK: [[V2:%.*]] = arith.constant
// CHECK: return [[V1]], %arg1, [[V2]] : !ptensor.ptensor<i64>, index, index

// -----
module {
    "dist.runtime_prototypes"() : () -> ()
    func.func @test_repartition(%arg0: !dist.dtensor<<?x?xi64>>) -> !dist.dtensor<<?x?xi64>> {
    %0 = dist.repartition %arg0 : !dist.dtensor<<?x?xi64>> to !dist.dtensor<<?x?xi64>>
    return %0 : !dist.dtensor<<?x?xi64>>
    }
}
// CHECK-LABEL: @test_repartition(%arg0: !ptensor.ptensor<?x?xi64>, %arg1: index, %arg2: index, %arg3: memref<2xindex>, %arg4: memref<2xindex>) -> (!ptensor.ptensor<?x?xi64>, index, index, memref<2xindex>, memref<2xindex>) {
// CHECK: ptensor.extract_raw_ptr
// CHECK: memref.cast
// CHECK: memref.cast
// CHECK: memref.cast
// CHECK: memref.cast
// CHECK: call @_idtr_repartition
// CHECK: memref.store
// CHECK: memref.store
// CHECK: memref.store
// CHECK: memref.store
// CHECK: return

// -----
module {
    "dist.runtime_prototypes"() : () -> ()
    func.func @test_reshape(%arg0: !dist.dtensor<<?xi64>>) -> !dist.dtensor<<?x?xi64>> {
        %c1 = arith.constant 1 : index
        %c33 = arith.constant 3 : index
        %0 = "ptensor.reshape"(%arg0, %c1, %c33) : (!dist.dtensor<<?xi64>>, index, index) -> !dist.dtensor<<?x?xi64>>
        return %0 : !dist.dtensor<<?x?xi64>>
    }
}
// CHECK-LABEL: @test_reshape
// CHECK: scf.if
// CHECK: func.call @_idtr_nprocs
// CHECK: func.call @_idtr_prank
// CHECK: func.call @_idtr_reshape
// CHECK: } else {
// CHECK: arith.divsi
// CHECK: "ptensor.reshape"(%arg0
// CHECK: }
// CHECK: return
// CHECK-SAME: : !ptensor.ptensor<?x?xi64>, index, index, memref<2xindex>, memref<2xindex>
