#map = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {torch.debug_module_name = "ReLU"} {
  func.func @forward(%arg0: tensor<1x2048x20x15xf16>) -> tensor<1x2048x20x15xf16> {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<1x2048x20x15xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x2048x20x15xf16>) outs(%0 : tensor<1x2048x20x15xf16>) {
    ^bb0(%in: f16, %out: f16):
      %2 = arith.cmpf ugt, %in, %cst : f16
      %3 = arith.select %2, %in, %cst : f16
      linalg.yield %3 : f16
    } -> tensor<1x2048x20x15xf16>
    return %1 : tensor<1x2048x20x15xf16>
  }
  func.func @main() {
    %0= arith.constant dense<1.3>:tensor<1x2048x20x15xf16>
    %1 = call @forward(%0) : (tensor<1x2048x20x15xf16>) -> tensor<1x2048x20x15xf16>
     return
  }
}
