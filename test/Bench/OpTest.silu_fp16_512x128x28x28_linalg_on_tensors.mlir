#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {torch.debug_module_name = "SiLU"} {
  memref.global "private" constant @__constant_512x128x28x28xf16 : memref<512x128x28x28xf16> = dense<1.299800e+00>
  func.func @forward(%arg0: tensor<512x128x28x28xf16>) -> tensor<512x128x28x28xf16> {
    %cst = arith.constant 1.000000e+00 : f16
    %0 = tensor.empty() : tensor<512x128x28x28xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<512x128x28x28xf16>) outs(%0 : tensor<512x128x28x28xf16>) {
    ^bb0(%in: f16, %out: f16):
      %3 = arith.negf %in : f16
      %4 = math.exp %3 : f16
      %5 = arith.addf %4, %cst : f16
      %6 = arith.divf %cst, %5 : f16
      linalg.yield %6 : f16
    } -> tensor<512x128x28x28xf16>
    %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1, %arg0 : tensor<512x128x28x28xf16>, tensor<512x128x28x28xf16>) outs(%0 : tensor<512x128x28x28xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %3 = arith.mulf %in, %in_0 : f16
      linalg.yield %3 : f16
    } -> tensor<512x128x28x28xf16>
    return %2 : tensor<512x128x28x28xf16>
  }
  func.func @main() {
    %0 = memref.get_global @__constant_512x128x28x28xf16 : memref<512x128x28x28xf16>
    %1 = call @forward(%0) : (memref<512x128x28x28xf16>) -> memref<512x128x28x28xf16>
     return
  }
}
