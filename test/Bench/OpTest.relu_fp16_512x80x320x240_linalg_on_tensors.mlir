#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {torch.debug_module_name = "ReLU"} {
  func.func @forward(%arg0: tensor<512x80x320x240xf16>) -> tensor<512x80x320x240xf16> {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<512x80x320x240xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<512x80x320x240xf16>) outs(%0 : tensor<512x80x320x240xf16>) {
    ^bb0(%in: f16, %out: f16):
      %2 = arith.cmpf ugt, %in, %cst : f16
      %3 = arith.select %2, %in, %cst : f16
      linalg.yield %3 : f16
    } -> tensor<512x80x320x240xf16>
    return %1 : tensor<512x80x320x240xf16>
  }
  func.func @main() {
    %0= arith.constant dense<1.3>:tensor<512x80x320x240xf16>
    %1 = call @forward(%0) : (tensor<512x80x320x240xf16>) -> tensor<512x80x320x240xf16>
     return
  }
}
