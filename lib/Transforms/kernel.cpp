/*
 * Copyright (c) 2020, Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include <brgemm/xetpp_brgemm.hpp>
#include <core/xetpp_core.hpp>
#include <gemm/xetpp_gemm.hpp>
#include <util/xetpp_util.hpp>

using namespace __XETPP_NS;
using namespace __XETPP_BRGEMM_NS;
using namespace __XETPP_TILE_NS;

_GENX_MAIN_ void fc_noprepostproc_fwd_16384_1024_256_BF16_BF16_Kernel(
    data_type_a_d *A [[type("svmptr_t")]],
    data_type_b_d *B [[type("svmptr_t")]],
    data_type_c_d *C [[type("svmptr_t")]]) {
  using namespace __XETPP_NS;
  using namespace __XETPP_BRGEMM_NS;
  using namespace __XETPP_GEMM_NS;
  constexpr mem_layout mem_layout_a = MEM_LAYOUT_A;
  constexpr mem_layout mem_layout_b = MEM_LAYOUT_B;
  constexpr pre_kind pre_op = PRE_OP;
  xetpp_exec_item<3> ei;
  static constexpr uint32_t periodic_sync_interval = 8;
  static constexpr uint32_t prefetch_distance = 3;
  int matrix_m = 16384;
  int matrix_n = 256;
  int matrix_k = 1024;

  constexpr embedded_kind embedded_op_kind =
      mem_layout_a == mem_layout::col_major
          ? embedded_kind::cooperative_slm_tran_a
          : embedded_kind::none;

  using tile_attr = brgemm_tile_attr_t<wg_tile_m_d, wg_tile_n_d, sg_tile_m_d,
                                       sg_tile_n_d, sg_tile_k_d>;
  using perf_tuning_knob =
      brgemm_perf_tuning_knob_t<periodic_sync_interval, prefetch_distance>;
  using gemm_attr =
      gemm_attr_t<tile_attr, perf_tuning_knob, l3_kslicing_d, embedded_op_kind>;
  using pre_op_t = xetpp_pre_op_t<pre_op, data_type_c_d, float, gpu_arch::Xe>;
  using gemm_op =
      xetpp_gemm_t<data_type_a_d, data_type_b_d, data_type_c_d, float,
                   gemm_attr, mem_layout_a, mem_layout_b, mem_space::global,
                   mem_space::global, mem_space::global, accum_op::MMAOp,
                   gpu_arch::Xe, pre_op_t>;
  typename gemm_op::arguments_t arg;
  arg.matA_ptr = A;
  arg.matB_ptr = B;
  arg.matC_ptr = C;
  arg.matrix_m = matrix_m;
  arg.matrix_k = matrix_k;
  arg.matrix_n = matrix_n;
  constexpr uint32_t barrier_count = gemm_op::get_barrier_count::brgemm::count;
  constexpr uint32_t slm_size = gemm_op::get_slm_size::embedded_op::size;
  if constexpr (barrier_count != 0) {
    cm_nbarrier_init(barrier_count);
  }
  if constexpr (slm_size != 0) {
    cm_slm_init(slm_size);
    arg.matA_base = 0;
  }

  gemm_op::call(ei, &arg);
}
