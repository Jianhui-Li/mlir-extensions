// RUN: %python_executable %imex_runner -i %s -f %p/ptensor.pp -e main -entry-point-result=void --shared-libs=%mlir_c_runner_utils --shared-libs=%mlir_runner_utils | FileCheck %s

func.func private @printMemrefF64(%ptr : tensor<*xf64>)
func.func private @printMemrefI64(%ptr : tensor<*xi64>)

func.func @main() {
    %i5 = arith.constant 5 : index
    %ca = arith.constant 13.1 : f64
    %cb = arith.constant 2.2 : f64
    %cc = arith.constant 13 : i64
    %cd = arith.constant 2 : i64
    %ce = arith.constant 6 : i64
    %true = arith.constant 1 : i1
    %false = arith.constant 0 : i1

    %0 = ptensor.create %i5 value %ca {dtype = 0 : i8} : (index, f64) -> !ptensor.ptensor<?xf64>
    %1 = ptensor.create %i5 value %cb {dtype = 0 : i8} : (index, f64) -> !ptensor.ptensor<?xf64>
    %2 = ptensor.create %i5 value %cc {dtype = 2 : i8} : (index, i64) -> !ptensor.ptensor<?xi64>
    %3 = ptensor.create %i5 value %cd {dtype = 2 : i8} : (index, i64) -> !ptensor.ptensor<?xi64>
    %4 = ptensor.create %i5 value %ce {dtype = 2 : i8} : (index, i64) -> !ptensor.ptensor<?xi64>
    %5 = ptensor.create %i5 value %true {dtype = 10 : i8} : (index, i1) -> !ptensor.ptensor<?xi1>
    %6 = ptensor.create %i5 value %false {dtype = 10 : i8} : (index, i1) -> !ptensor.ptensor<?xi1>
    %7 = ptensor.create value %cd {dtype = 2 : i8} : (i64) -> !ptensor.ptensor<i64>

    call @test_add_f64(%0, %1) : (!ptensor.ptensor<?xf64>, !ptensor.ptensor<?xf64>) -> ()
    call @test_subtract_f64(%0, %1) : (!ptensor.ptensor<?xf64>, !ptensor.ptensor<?xf64>) -> ()
    call @test_mult_f64(%0, %1) : (!ptensor.ptensor<?xf64>, !ptensor.ptensor<?xf64>) -> ()
    call @test_minimum_f64(%0, %1) : (!ptensor.ptensor<?xf64>, !ptensor.ptensor<?xf64>) -> ()
    call @test_maximum_f64(%0, %1) : (!ptensor.ptensor<?xf64>, !ptensor.ptensor<?xf64>) -> ()
    call @test_modulo_f64(%0, %1) : (!ptensor.ptensor<?xf64>, !ptensor.ptensor<?xf64>) -> ()
    call @test_floordivide_i64(%2, %3) : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<?xi64>) -> ()
    call @test_atan2_f64(%0, %1) : (!ptensor.ptensor<?xf64>, !ptensor.ptensor<?xf64>) -> ()
    call @test_power_i64(%3, %2) : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<?xi64>) -> ()
    call @test_bitwise_and_i64(%2, %4) : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<?xi64>) -> ()
    call @test_bitwise_or_i64(%2, %4) : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<?xi64>) -> ()
    call @test_bitwise_xor_i64(%2, %4) : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<?xi64>) -> ()
    call @test_logical_and_i1(%5, %6) : (!ptensor.ptensor<?xi1>, !ptensor.ptensor<?xi1>) -> ()
    call @test_logical_or_i1(%5, %6) : (!ptensor.ptensor<?xi1>, !ptensor.ptensor<?xi1>) -> ()
    call @test_logical_xor_i1(%5, %6) : (!ptensor.ptensor<?xi1>, !ptensor.ptensor<?xi1>) -> ()
    call @test_add_bcast_0d_i64(%2, %7) : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<i64>) -> ()

    return
}

func.func @test_add_f64(%a : !ptensor.ptensor<?xf64>, %b : !ptensor.ptensor<?xf64>) {
    %0 ="ptensor.ewbin"(%a, %b) {op = 0 : i32} : (!ptensor.ptensor<?xf64>, !ptensor.ptensor<?xf64>) -> !ptensor.ptensor<?xf64>
    %1 = builtin.unrealized_conversion_cast %0 : !ptensor.ptensor<?xf64> to tensor<?xf64>
    %2 = tensor.cast %1 : tensor<?xf64> to tensor<*xf64>
    call @printMemrefF64(%2) : (tensor<*xf64>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [5] strides = [1] data =
    // CHECK-NEXT: [15.3,  15.3,  15.3,  15.3,  15.3]

    return
}

func.func @test_subtract_f64(%a : !ptensor.ptensor<?xf64>, %b : !ptensor.ptensor<?xf64>) {
    %0 ="ptensor.ewbin"(%a, %b) {op = 24 : i32} : (!ptensor.ptensor<?xf64>, !ptensor.ptensor<?xf64>) -> !ptensor.ptensor<?xf64>
    %1 = builtin.unrealized_conversion_cast %0 : !ptensor.ptensor<?xf64> to tensor<?xf64>
    %2 = tensor.cast %1 : tensor<?xf64> to tensor<*xf64>
    call @printMemrefF64(%2) : (tensor<*xf64>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [5] strides = [1] data =
    // CHECK-NEXT: [10.9,  10.9,  10.9,  10.9,  10.9]

    return
}

func.func @test_mult_f64(%a : !ptensor.ptensor<?xf64>, %b : !ptensor.ptensor<?xf64>) {
    %0 ="ptensor.ewbin"(%a, %b) {op = 21 : i32} : (!ptensor.ptensor<?xf64>, !ptensor.ptensor<?xf64>) -> !ptensor.ptensor<?xf64>
    %1 = builtin.unrealized_conversion_cast %0 : !ptensor.ptensor<?xf64> to tensor<?xf64>
    %2 = tensor.cast %1 : tensor<?xf64> to tensor<*xf64>
    call @printMemrefF64(%2) : (tensor<*xf64>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [5] strides = [1] data =
    // CHECK-NEXT: [28.82,  28.82,  28.82,  28.82,  28.82]

    return
}

func.func @test_minimum_f64(%a : !ptensor.ptensor<?xf64>, %b : !ptensor.ptensor<?xf64>) {
    %0 ="ptensor.ewbin"(%a, %b) {op = 19 : i32} : (!ptensor.ptensor<?xf64>, !ptensor.ptensor<?xf64>) -> !ptensor.ptensor<?xf64>
    %1 = builtin.unrealized_conversion_cast %0 : !ptensor.ptensor<?xf64> to tensor<?xf64>
    %2 = tensor.cast %1 : tensor<?xf64> to tensor<*xf64>
    call @printMemrefF64(%2) : (tensor<*xf64>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [5] strides = [1] data =
    // CHECK-NEXT: [2.2,  2.2,  2.2,  2.2,  2.2]

    return
}

func.func @test_maximum_f64(%a : !ptensor.ptensor<?xf64>, %b : !ptensor.ptensor<?xf64>) {
    %0 ="ptensor.ewbin"(%a, %b) {op = 18 : i32} : (!ptensor.ptensor<?xf64>, !ptensor.ptensor<?xf64>) -> !ptensor.ptensor<?xf64>
    %1 = builtin.unrealized_conversion_cast %0 : !ptensor.ptensor<?xf64> to tensor<?xf64>
    %2 = tensor.cast %1 : tensor<?xf64> to tensor<*xf64>
    call @printMemrefF64(%2) : (tensor<*xf64>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [5] strides = [1] data =
    // CHECK-NEXT: [13.1,  13.1,  13.1,  13.1,  13.1]

    return
}

func.func @test_modulo_f64(%a : !ptensor.ptensor<?xf64>, %b : !ptensor.ptensor<?xf64>) {
    %0 ="ptensor.ewbin"(%a, %b) {op = 20 : i32} : (!ptensor.ptensor<?xf64>, !ptensor.ptensor<?xf64>) -> !ptensor.ptensor<?xf64>
    %1 = builtin.unrealized_conversion_cast %0 : !ptensor.ptensor<?xf64> to tensor<?xf64>
    %2 = tensor.cast %1 : tensor<?xf64> to tensor<*xf64>
    call @printMemrefF64(%2) : (tensor<*xf64>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [5] strides = [1] data =
    // CHECK-NEXT: [2.1,  2.1,  2.1,  2.1,  2.1]

    return
}

func.func @test_floordivide_i64(%a : !ptensor.ptensor<?xi64>, %b : !ptensor.ptensor<?xi64>) {
    %0 ="ptensor.ewbin"(%a, %b) {op = 8 : i32} : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<?xi64>) -> !ptensor.ptensor<?xi64>
    %1 = builtin.unrealized_conversion_cast %0 : !ptensor.ptensor<?xi64> to tensor<?xi64>
    %2 = tensor.cast %1 : tensor<?xi64> to tensor<*xi64>
    call @printMemrefI64(%2) : (tensor<*xi64>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [5] strides = [1] data =
    // CHECK-NEXT: [6,  6,  6,  6,  6]

    return
}

func.func @test_atan2_f64(%a : !ptensor.ptensor<?xf64>, %b : !ptensor.ptensor<?xf64>) {
    %0 ="ptensor.ewbin"(%a, %b) {op = 1 : i32} : (!ptensor.ptensor<?xf64>, !ptensor.ptensor<?xf64>) -> !ptensor.ptensor<?xf64>
    %1 = builtin.unrealized_conversion_cast %0 : !ptensor.ptensor<?xf64> to tensor<?xf64>
    %2 = tensor.cast %1 : tensor<?xf64> to tensor<*xf64>
    call @printMemrefF64(%2) : (tensor<*xf64>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [5] strides = [1] data =
    // CHECK-NEXT: [1.40441,  1.40441,  1.40441,  1.40441,  1.40441]

    return
}

func.func @test_power_i64(%a : !ptensor.ptensor<?xi64>, %b : !ptensor.ptensor<?xi64>) {
    %0 ="ptensor.ewbin"(%a, %b) {op = 23 : i32} : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<?xi64>) -> !ptensor.ptensor<?xi64>
    %1 = builtin.unrealized_conversion_cast %0 : !ptensor.ptensor<?xi64> to tensor<?xi64>
    %2 = tensor.cast %1 : tensor<?xi64> to tensor<*xi64>
    call @printMemrefI64(%2) : (tensor<*xi64>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [5] strides = [1] data =
    // CHECK-NEXT: [8192,  8192,  8192,  8192,  8192]

    return
}

func.func @test_bitwise_and_i64(%a : !ptensor.ptensor<?xi64>, %b : !ptensor.ptensor<?xi64>) {
    %0 ="ptensor.ewbin"(%a, %b) {op = 2 : i32} : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<?xi64>) -> !ptensor.ptensor<?xi64>
    %1 = builtin.unrealized_conversion_cast %0 : !ptensor.ptensor<?xi64> to tensor<?xi64>
    %2 = tensor.cast %1 : tensor<?xi64> to tensor<*xi64>
    call @printMemrefI64(%2) : (tensor<*xi64>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [5] strides = [1] data =
    // CHECK-NEXT: [4,  4,  4,  4,  4]

    return
}

func.func @test_bitwise_or_i64(%a : !ptensor.ptensor<?xi64>, %b : !ptensor.ptensor<?xi64>) {
    %0 ="ptensor.ewbin"(%a, %b) {op = 4 : i32} : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<?xi64>) -> !ptensor.ptensor<?xi64>
    %1 = builtin.unrealized_conversion_cast %0 : !ptensor.ptensor<?xi64> to tensor<?xi64>
    %2 = tensor.cast %1 : tensor<?xi64> to tensor<*xi64>
    call @printMemrefI64(%2) : (tensor<*xi64>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [5] strides = [1] data =
    // CHECK-NEXT: [15,  15,  15,  15,  15]

    return
}

func.func @test_bitwise_xor_i64(%a : !ptensor.ptensor<?xi64>, %b : !ptensor.ptensor<?xi64>) {
    %0 ="ptensor.ewbin"(%a, %b) {op = 6 : i32} : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<?xi64>) -> !ptensor.ptensor<?xi64>
    %1 = builtin.unrealized_conversion_cast %0 : !ptensor.ptensor<?xi64> to tensor<?xi64>
    %2 = tensor.cast %1 : tensor<?xi64> to tensor<*xi64>
    call @printMemrefI64(%2) : (tensor<*xi64>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [5] strides = [1] data =
    // CHECK-NEXT: [11,  11,  11,  11,  11]

    return
}

func.func @test_logical_and_i1(%a : !ptensor.ptensor<?xi1>, %b : !ptensor.ptensor<?xi1>) {
    %i0 = arith.constant 0 : index
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %0 ="ptensor.ewbin"(%a, %b) {op = 14 : i32} : (!ptensor.ptensor<?xi1>, !ptensor.ptensor<?xi1>) -> !ptensor.ptensor<?xi1>
    %1 = builtin.unrealized_conversion_cast %0 : !ptensor.ptensor<?xi1> to tensor<?xi1>
    %2 = "tosa.cast" (%1) : (tensor<?xi1>) -> tensor<?xi64>
    %3 = tensor.cast %2 : tensor<?xi64> to tensor<*xi64>
    call @printMemrefI64(%3) : (tensor<*xi64>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [5] strides = [1] data =
    // CHECK-NEXT: [0,  0,  0,  0,  0]

    return
}

func.func @test_logical_or_i1(%a : !ptensor.ptensor<?xi1>, %b : !ptensor.ptensor<?xi1>) {
    %i0 = arith.constant 0 : index
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %0 ="ptensor.ewbin"(%a, %b) {op = 15 : i32} : (!ptensor.ptensor<?xi1>, !ptensor.ptensor<?xi1>) -> !ptensor.ptensor<?xi1>
    %1 = builtin.unrealized_conversion_cast %0 : !ptensor.ptensor<?xi1> to tensor<?xi1>
    %2 = "tosa.cast" (%1) : (tensor<?xi1>) -> tensor<?xi64>
    %3 = tensor.cast %2 : tensor<?xi64> to tensor<*xi64>
    call @printMemrefI64(%3) : (tensor<*xi64>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [5] strides = [1] data =
    // CHECK-NEXT: [1,  1,  1,  1,  1]

    return
}

func.func @test_logical_xor_i1(%a : !ptensor.ptensor<?xi1>, %b : !ptensor.ptensor<?xi1>) {
    %i0 = arith.constant 0 : index
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %0 ="ptensor.ewbin"(%a, %b) {op = 16 : i32} : (!ptensor.ptensor<?xi1>, !ptensor.ptensor<?xi1>) -> !ptensor.ptensor<?xi1>
    %1 = builtin.unrealized_conversion_cast %0 : !ptensor.ptensor<?xi1> to tensor<?xi1>
    %2 = "tosa.cast" (%1) : (tensor<?xi1>) -> tensor<?xi64>
    %3 = tensor.cast %2 : tensor<?xi64> to tensor<*xi64>
    call @printMemrefI64(%3) : (tensor<*xi64>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [5] strides = [1] data =
    // CHECK-NEXT: [1,  1,  1,  1,  1]

    return
}

func.func @test_add_bcast_0d_i64(%a : !ptensor.ptensor<?xi64>, %b : !ptensor.ptensor<i64>) {
    %0 ="ptensor.ewbin"(%a, %b) {op = 0 : i32} : (!ptensor.ptensor<?xi64>, !ptensor.ptensor<i64>) -> !ptensor.ptensor<?xi64>
    %1 = builtin.unrealized_conversion_cast %0 : !ptensor.ptensor<?xi64> to tensor<?xi64>
    %2 = tensor.cast %1 : tensor<?xi64> to tensor<*xi64>
    call @printMemrefI64(%2) : (tensor<*xi64>) -> ()
    // CHECK: Unranked Memref base@ = {{(0x)?[-9a-f]*}}
    // CHECK-SAME: rank = 1 offset = 0 sizes = [5] strides = [1] data =
    // CHECK-NEXT: [15,  15,  15,  15,  15]

    return
}
