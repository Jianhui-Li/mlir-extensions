#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {torch.debug_module_name = "SiLU"} {
  func.func @forward(%arg0: tensor<512x1280x20x15xf16>) -> tensor<512x1280x20x15xf16> {
    %cst = arith.constant 1.000000e+00 : f16
    %0 = tensor.empty() : tensor<512x1280x20x15xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<512x1280x20x15xf16>) outs(%0 : tensor<512x1280x20x15xf16>) {
    ^bb0(%in: f16, %out: f16):
      %3 = arith.negf %in : f16
      %4 = math.exp %3 : f16
      %5 = arith.addf %4, %cst : f16
      %6 = arith.divf %cst, %5 : f16
      linalg.yield %6 : f16
    } -> tensor<512x1280x20x15xf16>
    %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1, %arg0 : tensor<512x1280x20x15xf16>, tensor<512x1280x20x15xf16>) outs(%0 : tensor<512x1280x20x15xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %3 = arith.mulf %in, %in_0 : f16
      linalg.yield %3 : f16
    } -> tensor<512x1280x20x15xf16>
    return %2 : tensor<512x1280x20x15xf16>
  }
  func.func @main() {
    %0= arith.constant dense<1.3>:tensor<512x1280x20x15xf16>
    %1 = call @forward(%0) : (tensor<512x1280x20x15xf16>) -> tensor<512x1280x20x15xf16>
    return
  }
}
