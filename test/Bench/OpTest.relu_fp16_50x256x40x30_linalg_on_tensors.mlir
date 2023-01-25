#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {torch.debug_module_name = "ReLU"} {
  memref.global "private" constant @__constant_50x256x40x30xf16 : memref<50x256x40x30xf16> = dense<1.299800e+00>
  func.func @forward(%arg0: tensor<50x256x40x30xf16>) -> tensor<50x256x40x30xf16> {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<50x256x40x30xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<50x256x40x30xf16>) outs(%0 : tensor<50x256x40x30xf16>) {
    ^bb0(%in: f16, %out: f16):
      %2 = arith.cmpf ugt, %in, %cst : f16
      %3 = arith.select %2, %in, %cst : f16
      linalg.yield %3 : f16
    } -> tensor<50x256x40x30xf16>
    return %1 : tensor<50x256x40x30xf16>
  }
  func.func @main() {
    %0 = memref.get_global @__constant_50x256x40x30xf16 : memref<50x256x40x30xf16>
    %1 = call @forward(%0) : (memref<50x256x40x30xf16>) -> memref<50x256x40x30xf16>
     return
  }
}
