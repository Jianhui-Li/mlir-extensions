#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {torch.debug_module_name = "GELU"} {
  func.func @forward(%arg0: tensor<512x80x320x240xf16>) -> tensor<512x80x320x240xf16> {
    %cst = arith.constant 1.000000e+00 : f16
    %cst_0 = arith.constant 2.000000e+00 : f16
    %cst_1 = arith.constant 5.000000e-01 : f16
    %0 = tensor.empty() : tensor<512x80x320x240xf16>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<512x80x320x240xf16>) outs(%0 : tensor<512x80x320x240xf16>) {
    ^bb0(%in: f16, %out: f16):
      %2 = math.sqrt %cst_0 : f16
      %3 = arith.divf %in, %2 : f16
      %4 = math.erf %3 : f16
      %5 = arith.addf %4, %cst : f16
      %6 = arith.mulf %5, %cst_1 : f16
      %7 = arith.mulf %in, %6 : f16
      linalg.yield %7 : f16
    } -> tensor<512x80x320x240xf16>
    return %1 : tensor<512x80x320x240xf16>
  }
  func.func @main() {
    %0= arith.constant dense<1.3>:tensor<512x80x320x240xf16>
    %1 = call @forward(%0) : (tensor<512x80x320x240xf16>) -> tensor<512x80x320x240xf16>
    return
  }
}
