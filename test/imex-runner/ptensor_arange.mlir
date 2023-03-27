// RUN: %python_executable %imex_runner -i %s -f %p/ptensor.pp -e main -entry-point-result=void --shared-libs=%mlir_c_runner_utils --shared-libs=%mlir_runner_utils | FileCheck %s

module {
    func.func private @printMemrefI64(%ptr : tensor<*xi64>)
    func.func @main() {
        %c0 = arith.constant 0 : i64
        %c1 = arith.constant 1 : i64
        %c2 = arith.constant 2 : i64
        %c5 = arith.constant 5 : i64
        %c10 = arith.constant 10 : i64
        %i1 = arith.constant 1 : index
        %i2 = arith.constant 2 : index

        %3 = ptensor.arange %c0 %c10 %c2 : (i64, i64, i64) -> !ptensor.ptensor<?xi64>
        %4 = builtin.unrealized_conversion_cast %3 : !ptensor.ptensor<?xi64> to tensor<?xi64>
        %5 = tensor.cast %4 : tensor<?xi64> to tensor<*xi64>
        call @printMemrefI64(%5) : (tensor<*xi64>) -> ()
        // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
        // CHECK-SAME: rank = 1 offset = 0 sizes = [5] strides = [1] data =
        // CHECK-NEXT: [0,  2,  4,  6,  8]

        // %13 = ptensor.arange %c0 %c10 %c2 : (i64, i64, i64, i64, i64) -> !ptensor.ptensor<?xi64>
        // %14 = "ptensor.extract_tensor"(%13) : (!ptensor.ptensor<?xi64>) -> tensor<?xi64>
        // %15 = tensor.cast %14 : tensor<?xi64> to tensor<*xi64>
        // call @printMemrefI64(%15) : (tensor<*xi64>) -> ()
        // _CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
        // _CHECK-SAME: rank = 1 offset = 0 sizes = [5] strides = [1] data =
        // _CHECK-NEXT: [0,  2,  4,  6,  8]

        %20 = ptensor.subview %3[%i1][%i2][%i2] : !ptensor.ptensor<?xi64> to !ptensor.ptensor<?xi64>
        %21 = builtin.unrealized_conversion_cast %20 : !ptensor.ptensor<?xi64> to tensor<?xi64>
        %22 = tensor.cast %21 : tensor<?xi64> to tensor<*xi64>
        call @printMemrefI64(%22) : (tensor<*xi64>) -> ()
        // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
        // CHECK-SAME: rank = 1 offset = 1 sizes = [2] strides = [2] data =
        // CHECK-NEXT: [2, 6]


        %30 = ptensor.arange %c0 %c2 %c1 : (i64, i64, i64) -> !ptensor.ptensor<?xi64>
        ptensor.insert_slice %30 into %3[%i1] [%i2] [%i2] : !ptensor.ptensor<?xi64> into !ptensor.ptensor<?xi64>
        // %31 = "ptensor.extract_tensor"(%30) : (!ptensor.ptensor<?xi64>) -> tensor<?xi64>
        // %32 = tensor.cast %31 : tensor<?xi64> to tensor<*xi64>
        call @printMemrefI64(%5) : (tensor<*xi64>) -> ()
        // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
        // CHECK-SAME: rank = 1 offset = 0 sizes = [5] strides = [1] data =
        // CHECK-NEXT: [0,  0,  4,  1,  8]


        %40 = ptensor.create %i2, %i2, %i2 value %c5 {dtype = 2 : i8} : (index, index, index, i64) -> !ptensor.ptensor<?x?x?xi64>
        %41 = ptensor.create %i2 value %c5 {dtype = 2 : i8} : (index, i64) -> !ptensor.ptensor<?xi64>
        %42 = "ptensor.ewbin"(%40, %41) {op = 0 : i32} : (!ptensor.ptensor<?x?x?xi64>, !ptensor.ptensor<?xi64>) -> !ptensor.ptensor<?x?x?xi64>
        %44 = builtin.unrealized_conversion_cast %42 : !ptensor.ptensor<?x?x?xi64> to tensor<?x?x?xi64>
        %45 = tensor.cast %44 : tensor<?x?x?xi64> to tensor<*xi64>
        call @printMemrefI64(%45) : (tensor<*xi64>) -> ()
        // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
        // CHECK-SAME: rank = 3 offset = 0 sizes = [2, 2, 2] strides = [4, 2, 1] data =
        // CHECK-NEXT{LITERAL}: [[[10,    10],
        // CHECK-NEXT{LITERAL}:   [10,    10]],
        // CHECK-NEXT{LITERAL}:  [[10,    10],
        // CHECK-NEXT{LITERAL}:   [10,    10]]]

        %50 = "ptensor.reduction"(%42) {op = 4 : i32} : (!ptensor.ptensor<?x?x?xi64>) -> !ptensor.ptensor<i64>
        %54 = builtin.unrealized_conversion_cast %50 : !ptensor.ptensor<i64> to tensor<i64>
        %55 = tensor.cast %54 : tensor<i64> to tensor<*xi64>
        call @printMemrefI64(%55) : (tensor<*xi64>) -> ()
        // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
        // CHECK-SAME: rank = 0 offset = 0 sizes = [] strides = [] data =
        // CHECK-NEXT{LITERAL}: [80]

        return
    }
}
