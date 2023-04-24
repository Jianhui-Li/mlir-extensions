// RUN: %python_executable %imex_runner -i %s -f %p/ptensor.pp -e main -entry-point-result=void --shared-libs=%mlir_c_runner_utils --shared-libs=%mlir_runner_utils | FileCheck %s

func.func private @printMemrefI64(%ptr : tensor<*xi64>)

func.func @main() {
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %i3 = arith.constant 3 : index
    %i5 = arith.constant 5 : index
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c2 = arith.constant 2 : i64
    %c5 = arith.constant 5 : i64

    %0 = ptensor.linspace %c0 %c5 %c5 false : (i64, i64, i64) -> !ptensor.ptensor<?xi64>
    %1 = ptensor.create value %c2 {dtype = 2 : i8} : (i64) -> !ptensor.ptensor<i64>
    %2 = ptensor.create %i1 value %c2 {dtype = 2 : i8} : (index, i64) -> !ptensor.ptensor<1xi64>
    %3 = ptensor.create %i1, %i3 value %c2 {dtype = 2 : i8} : (index, index, i64) -> !ptensor.ptensor<1x?xi64>
    %4 = ptensor.create %i5, %i3 value %c5 {dtype = 2 : i8} : (index, index, i64) -> !ptensor.ptensor<?x?xi64>
    %5 = ptensor.create %i2, %i1, %i2, %i1 value %c5 {dtype = 2 : i8} : (index, index, index, index, i64) -> !ptensor.ptensor<?x1x?x1xi64>
    %6 = ptensor.create %i2, %i2, %i2, %i2 value %c2 {dtype = 2 : i8} : (index, index, index, index, i64) -> !ptensor.ptensor<?x?x?x?xi64>

    call @test_arith_1D_A(%0, %1) : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<i64>) -> ()
    call @test_tosa_1D_A(%0, %1) : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<i64>) -> ()

    call @test_arith_1D_B(%0, %2) : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<1xi64>) -> ()
    call @test_tosa_1D_B(%0, %2) : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<1xi64>) -> ()

    call @test_arith_2D_A(%0, %3) : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<1x?xi64>) -> ()
    call @test_tosa_2D_A(%0, %3) : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<1x?xi64>) -> ()

    call @test_arith_2D_B(%1, %4) : (!ptensor.ptensor<i64>, !ptensor.ptensor<?x?xi64>) -> ()
    call @test_tosa_2D_B(%1, %4) : (!ptensor.ptensor<i64>, !ptensor.ptensor<?x?xi64>) -> ()

    call @test_arith_4D_A(%5, %6) : (!ptensor.ptensor<?x1x?x1xi64>, !ptensor.ptensor<?x?x?x?xi64>) -> ()

    return
}

func.func @test_arith_1D_A(%a : !ptensor.ptensor<?xi64>, %b : !ptensor.ptensor<i64>) {

    %0 ="ptensor.ewbin"(%a, %b) {op = 0 : i32} : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<i64>) -> !ptensor.ptensor<?xi64>

    %1 = builtin.unrealized_conversion_cast %0 : !ptensor.ptensor<?xi64> to tensor<?xi64>
    %2 = tensor.cast %1 : tensor<?xi64> to tensor<*xi64>
    call @printMemrefI64(%2) : (tensor<*xi64>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [5] strides = [1] data =
    // CHECK-NEXT: [2,  3,  4,  5,  6]

    %5 ="ptensor.ewbin"(%b, %a) {op = 0 : i32} : (!ptensor.ptensor<i64>, !ptensor.ptensor<?xi64>) -> !ptensor.ptensor<?xi64>

    %6 = builtin.unrealized_conversion_cast %5 : !ptensor.ptensor<?xi64> to tensor<?xi64>
    %7 = tensor.cast %6 : tensor<?xi64> to tensor<*xi64>
    call @printMemrefI64(%7) : (tensor<*xi64>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [5] strides = [1] data =
    // CHECK-NEXT: [2,  3,  4,  5,  6]

    return
}

func.func @test_tosa_1D_A(%a : !ptensor.ptensor<?xi64>, %b : !ptensor.ptensor<i64>) {

    %0 ="ptensor.ewbin"(%a, %b) {op = 4 : i32} : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<i64>) -> !ptensor.ptensor<?xi64>

    %1 = builtin.unrealized_conversion_cast %0 : !ptensor.ptensor<?xi64> to tensor<?xi64>
    %2 = tensor.cast %1 : tensor<?xi64> to tensor<*xi64>
    call @printMemrefI64(%2) : (tensor<*xi64>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [5] strides = [1] data =
    // CHECK-NEXT: [2,  3,  2,  3,  6]

    %5 ="ptensor.ewbin"(%b, %a) {op = 4 : i32} : (!ptensor.ptensor<i64>, !ptensor.ptensor<?xi64>) -> !ptensor.ptensor<?xi64>

    %6 = builtin.unrealized_conversion_cast %5 : !ptensor.ptensor<?xi64> to tensor<?xi64>
    %7 = tensor.cast %6 : tensor<?xi64> to tensor<*xi64>
    call @printMemrefI64(%7) : (tensor<*xi64>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [5] strides = [1] data =
    // CHECK-NEXT: [2,  3,  2,  3,  6]

    return
}

func.func @test_arith_1D_B(%a : !ptensor.ptensor<?xi64>, %b : !ptensor.ptensor<1xi64>) {

    %0 ="ptensor.ewbin"(%a, %b) {op = 0 : i32} : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<1xi64>) -> !ptensor.ptensor<?xi64>

    %1 = builtin.unrealized_conversion_cast %0 : !ptensor.ptensor<?xi64> to tensor<?xi64>
    %2 = tensor.cast %1 : tensor<?xi64> to tensor<*xi64>
    call @printMemrefI64(%2) : (tensor<*xi64>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [5] strides = [1] data =
    // CHECK-NEXT: [2,  3,  4,  5,  6]

    %5 ="ptensor.ewbin"(%b, %a) {op = 0 : i32} : (!ptensor.ptensor<1xi64>, !ptensor.ptensor<?xi64>) -> !ptensor.ptensor<?xi64>

    %6 = builtin.unrealized_conversion_cast %5 : !ptensor.ptensor<?xi64> to tensor<?xi64>
    %7 = tensor.cast %6 : tensor<?xi64> to tensor<*xi64>
    call @printMemrefI64(%7) : (tensor<*xi64>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [5] strides = [1] data =
    // CHECK-NEXT: [2,  3,  4,  5,  6]

    return
}

func.func @test_tosa_1D_B(%a : !ptensor.ptensor<?xi64>, %b : !ptensor.ptensor<1xi64>) {

    %0 ="ptensor.ewbin"(%a, %b) {op = 4 : i32} : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<1xi64>) -> !ptensor.ptensor<?xi64>

    %1 = builtin.unrealized_conversion_cast %0 : !ptensor.ptensor<?xi64> to tensor<?xi64>
    %2 = tensor.cast %1 : tensor<?xi64> to tensor<*xi64>
    call @printMemrefI64(%2) : (tensor<*xi64>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [5] strides = [1] data =
    // CHECK-NEXT: [2,  3,  2,  3,  6]

    %5 ="ptensor.ewbin"(%b, %a) {op = 4 : i32} : (!ptensor.ptensor<1xi64>, !ptensor.ptensor<?xi64>) -> !ptensor.ptensor<?xi64>

    %6 = builtin.unrealized_conversion_cast %5 : !ptensor.ptensor<?xi64> to tensor<?xi64>
    %7 = tensor.cast %6 : tensor<?xi64> to tensor<*xi64>
    call @printMemrefI64(%7) : (tensor<*xi64>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [5] strides = [1] data =
    // CHECK-NEXT: [2,  3,  2,  3,  6]

    return
}

func.func @test_arith_2D_A(%a : !ptensor.ptensor<?xi64>, %b : !ptensor.ptensor<1x?xi64>) {

    %0 ="ptensor.ewbin"(%a, %b) {op = 0 : i32} : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<1x?xi64>) -> !ptensor.ptensor<?x?xi64>

    %1 = builtin.unrealized_conversion_cast %0 : !ptensor.ptensor<?x?xi64> to tensor<?x?xi64>
    %2 = tensor.cast %1 : tensor<?x?xi64> to tensor<*xi64>
    call @printMemrefI64(%2) : (tensor<*xi64>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 2 offset = 0 sizes = [5, 3] strides = [3, 1] data =
    // CHECK-NEXT{LITERAL}: [[2,   2,   2],
    // CHECK-NEXT{LITERAL}:  [3,   3,   3],
    // CHECK-NEXT{LITERAL}:  [4,   4,   4],
    // CHECK-NEXT{LITERAL}:  [5,   5,   5],
    // CHECK-NEXT{LITERAL}:  [6,   6,   6]]

    %5 ="ptensor.ewbin"(%b, %a) {op = 0 : i32} : (!ptensor.ptensor<1x?xi64>, !ptensor.ptensor<?xi64>) -> !ptensor.ptensor<?x?xi64>

    %6 = builtin.unrealized_conversion_cast %5 : !ptensor.ptensor<?x?xi64> to tensor<?x?xi64>
    %7 = tensor.cast %6 : tensor<?x?xi64> to tensor<*xi64>
    call @printMemrefI64(%7) : (tensor<*xi64>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 2 offset = 0 sizes = [5, 3] strides = [3, 1] data =
    // CHECK-NEXT{LITERAL}: [[2,   2,   2],
    // CHECK-NEXT{LITERAL}:  [3,   3,   3],
    // CHECK-NEXT{LITERAL}:  [4,   4,   4],
    // CHECK-NEXT{LITERAL}:  [5,   5,   5],
    // CHECK-NEXT{LITERAL}:  [6,   6,   6]]

    return
}

func.func @test_tosa_2D_A(%a : !ptensor.ptensor<?xi64>, %b : !ptensor.ptensor<1x?xi64>) {

    %0 ="ptensor.ewbin"(%a, %b) {op = 4 : i32} : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<1x?xi64>) -> !ptensor.ptensor<?x?xi64>

    %1 = builtin.unrealized_conversion_cast %0 : !ptensor.ptensor<?x?xi64> to tensor<?x?xi64>
    %2 = tensor.cast %1 : tensor<?x?xi64> to tensor<*xi64>
    call @printMemrefI64(%2) : (tensor<*xi64>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 2 offset = 0 sizes = [5, 3] strides = [3, 1] data =
    // CHECK-NEXT{LITERAL}: [[2,   2,   2],
    // CHECK-NEXT{LITERAL}:  [3,   3,   3],
    // CHECK-NEXT{LITERAL}:  [2,   2,   2],
    // CHECK-NEXT{LITERAL}:  [3,   3,   3],
    // CHECK-NEXT{LITERAL}:  [6,   6,   6]]

    %5 ="ptensor.ewbin"(%b, %a) {op = 4 : i32} : (!ptensor.ptensor<1x?xi64>, !ptensor.ptensor<?xi64>) -> !ptensor.ptensor<?x?xi64>

    %6 = builtin.unrealized_conversion_cast %5 : !ptensor.ptensor<?x?xi64> to tensor<?x?xi64>
    %7 = tensor.cast %6 : tensor<?x?xi64> to tensor<*xi64>
    call @printMemrefI64(%7) : (tensor<*xi64>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 2 offset = 0 sizes = [5, 3] strides = [3, 1] data =
    // CHECK-NEXT{LITERAL}: [[2,   2,   2],
    // CHECK-NEXT{LITERAL}:  [3,   3,   3],
    // CHECK-NEXT{LITERAL}:  [2,   2,   2],
    // CHECK-NEXT{LITERAL}:  [3,   3,   3],
    // CHECK-NEXT{LITERAL}:  [6,   6,   6]]

    return
}

func.func @test_arith_2D_B(%a : !ptensor.ptensor<i64>, %b : !ptensor.ptensor<?x?xi64>) {

    %0 ="ptensor.ewbin"(%a, %b) {op = 0 : i32} : (!ptensor.ptensor<i64>, !ptensor.ptensor<?x?xi64>) -> !ptensor.ptensor<?x?xi64>

    %1 = builtin.unrealized_conversion_cast %0 : !ptensor.ptensor<?x?xi64> to tensor<?x?xi64>
    %2 = tensor.cast %1 : tensor<?x?xi64> to tensor<*xi64>
    call @printMemrefI64(%2) : (tensor<*xi64>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 2 offset = 0 sizes = [5, 3] strides = [3, 1] data =
    // CHECK-NEXT{LITERAL}: [[7,   7,   7],
    // CHECK-NEXT{LITERAL}:  [7,   7,   7],
    // CHECK-NEXT{LITERAL}:  [7,   7,   7],
    // CHECK-NEXT{LITERAL}:  [7,   7,   7],
    // CHECK-NEXT{LITERAL}:  [7,   7,   7]]

    %5 ="ptensor.ewbin"(%b, %a) {op = 0 : i32} : (!ptensor.ptensor<?x?xi64>, !ptensor.ptensor<i64>) -> !ptensor.ptensor<?x?xi64>

    %6 = builtin.unrealized_conversion_cast %5 : !ptensor.ptensor<?x?xi64> to tensor<?x?xi64>
    %7 = tensor.cast %6 : tensor<?x?xi64> to tensor<*xi64>
    call @printMemrefI64(%7) : (tensor<*xi64>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 2 offset = 0 sizes = [5, 3] strides = [3, 1] data =
    // CHECK-NEXT{LITERAL}: [[7,   7,   7],
    // CHECK-NEXT{LITERAL}:  [7,   7,   7],
    // CHECK-NEXT{LITERAL}:  [7,   7,   7],
    // CHECK-NEXT{LITERAL}:  [7,   7,   7],
    // CHECK-NEXT{LITERAL}:  [7,   7,   7]]

    return
}

func.func @test_tosa_2D_B(%a : !ptensor.ptensor<i64>, %b : !ptensor.ptensor<?x?xi64>) {

    %0 ="ptensor.ewbin"(%a, %b) {op = 4 : i32} : (!ptensor.ptensor<i64>, !ptensor.ptensor<?x?xi64>) -> !ptensor.ptensor<?x?xi64>

    %1 = builtin.unrealized_conversion_cast %0 : !ptensor.ptensor<?x?xi64> to tensor<?x?xi64>
    %2 = tensor.cast %1 : tensor<?x?xi64> to tensor<*xi64>
    call @printMemrefI64(%2) : (tensor<*xi64>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 2 offset = 0 sizes = [5, 3] strides = [3, 1] data =
    // CHECK-NEXT{LITERAL}: [[7,   7,   7],
    // CHECK-NEXT{LITERAL}:  [7,   7,   7],
    // CHECK-NEXT{LITERAL}:  [7,   7,   7],
    // CHECK-NEXT{LITERAL}:  [7,   7,   7],
    // CHECK-NEXT{LITERAL}:  [7,   7,   7]]

    %5 ="ptensor.ewbin"(%b, %a) {op = 4 : i32} : (!ptensor.ptensor<?x?xi64>, !ptensor.ptensor<i64>) -> !ptensor.ptensor<?x?xi64>

    %6 = builtin.unrealized_conversion_cast %5 : !ptensor.ptensor<?x?xi64> to tensor<?x?xi64>
    %7 = tensor.cast %6 : tensor<?x?xi64> to tensor<*xi64>
    call @printMemrefI64(%7) : (tensor<*xi64>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 2 offset = 0 sizes = [5, 3] strides = [3, 1] data =
    // CHECK-NEXT{LITERAL}: [[7,   7,   7],
    // CHECK-NEXT{LITERAL}:  [7,   7,   7],
    // CHECK-NEXT{LITERAL}:  [7,   7,   7],
    // CHECK-NEXT{LITERAL}:  [7,   7,   7],
    // CHECK-NEXT{LITERAL}:  [7,   7,   7]]

    return
}

func.func @test_arith_4D_A(%a : !ptensor.ptensor<?x1x?x1xi64>, %b : !ptensor.ptensor<?x?x?x?xi64>) {

    %0 ="ptensor.ewbin"(%a, %b) {op = 0 : i32} : (!ptensor.ptensor<?x1x?x1xi64>, !ptensor.ptensor<?x?x?x?xi64>) -> !ptensor.ptensor<?x?x?x?xi64>

    %1 = builtin.unrealized_conversion_cast %0 : !ptensor.ptensor<?x?x?x?xi64> to tensor<?x?x?x?xi64>
    %2 = tensor.cast %1 : tensor<?x?x?x?xi64> to tensor<*xi64>
    call @printMemrefI64(%2) : (tensor<*xi64>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 4 offset = 0 sizes = [2, 2, 2, 2] strides = [8, 4, 2, 1] data =
    // CHECK-NEXT{LITERAL}: [[[[7,     7],
    // CHECK-NEXT{LITERAL}:    [7,     7]],
    // CHECK-NEXT{LITERAL}:   [[7,     7],
    // CHECK-NEXT{LITERAL}:    [7,     7]]],
    // CHECK-NEXT{LITERAL}:  [[[7,     7],
    // CHECK-NEXT{LITERAL}:    [7,     7]],
    // CHECK-NEXT{LITERAL}:   [[7,     7],
    // CHECK-NEXT{LITERAL}:    [7,     7]]]]

    %5 ="ptensor.ewbin"(%b, %a) {op = 0 : i32} : (!ptensor.ptensor<?x?x?x?xi64>, !ptensor.ptensor<?x1x?x1xi64>) -> !ptensor.ptensor<?x?x?x?xi64>

    %6 = builtin.unrealized_conversion_cast %5 : !ptensor.ptensor<?x?x?x?xi64> to tensor<?x?x?x?xi64>
    %7 = tensor.cast %6 : tensor<?x?x?x?xi64> to tensor<*xi64>
    call @printMemrefI64(%7) : (tensor<*xi64>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 4 offset = 0 sizes = [2, 2, 2, 2] strides = [8, 4, 2, 1] data =
    // CHECK-NEXT{LITERAL}: [[[[7,     7],
    // CHECK-NEXT{LITERAL}:    [7,     7]],
    // CHECK-NEXT{LITERAL}:   [[7,     7],
    // CHECK-NEXT{LITERAL}:    [7,     7]]],
    // CHECK-NEXT{LITERAL}:  [[[7,     7],
    // CHECK-NEXT{LITERAL}:    [7,     7]],
    // CHECK-NEXT{LITERAL}:   [[7,     7],
    // CHECK-NEXT{LITERAL}:    [7,     7]]]]

    return
}
