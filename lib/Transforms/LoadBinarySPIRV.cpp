//===- SerializeSPIRV.cpp - SPIR-V serialize pass --------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This pass iterates all the SPIR-V modules in the top module and serializes
/// each SPIR-V module to SPIR-V binary and then attachs the binary blob as a
/// string attribute to the corresponding gpu module.
///
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Target/SPIRV/Serialization.h"

#include <bits/stdc++.h>
#include <fstream>
#include <iterator>

using namespace mlir;
using namespace imex;

namespace {
struct LoadBinarySPIRVPass
    : public LoadBinarySPIRVPassBase<LoadBinarySPIRVPass> {
public:
  void runOnOperation() override {
    auto mod = getOperation();
    for (auto gpuMod : mod.getOps<gpu::GPUModuleOp>()) {
      auto name = gpuMod.getName();
      // check that the spv module has the same name with gpu module except the
      // prefix "__spv__"
      auto isSameMod = [&](spirv::ModuleOp spvMod) -> bool {
        auto spvModName = spvMod.getName();
        return spvModName->consume_front("__spv__") && spvModName == name;
      };
      auto spvMods = mod.getOps<spirv::ModuleOp>();
      auto it = llvm::find_if(spvMods, isSameMod);
      if (it == spvMods.end()) {
        gpuMod.emitError() << "Unable to find corresponding SPIR-V module";
        signalPassFailure();
        return;
      }
      auto spvMod = *it;

      char *spvBuf;
      int spvLen;
      if (false) { // Load pre-baked SPIR-V
        std::ifstream is;
        is.open("spirv-llvm-opt.reas-clean.spv", std::ios::binary);
        is.seekg(0, std::ios::end);
        spvLen = is.tellg();
        is.seekg(0, std::ios::beg);
        spvBuf = new char[spvLen];
        is.read(spvBuf, spvLen);
        is.close();
      } else { // Compile the XeTLA template and load the result
        std::system(
            "/localdisk/mdvoretc/imex/ComputeSDK_Linux_internal_2022_WW42/"
            "install/usr/bin/ocloc -file "
            "/localdisk/mdvoretc/imex/mlir-extensions/lib/Transforms/"
            "kernel.cpp -output kernel_output -device pvc -options '-cmc  "
            "/Qxcm_doubleGRF  -mCM_printregusage "
            "-Qxcm_jit_option=-DPASTokenReduction /Qxcm_jit_option=-enableBCR  "
            " -fcm-pointer -I./ "
            "-I/localdisk/mdvoretc/imex/libraries.gpu.xetla/include "
            "-DXETPP_CODE_BASE=__CM__   "
            "-DMEM_LAYOUT_A=__XETPP_NS::mem_layout::row_major  "
            "-DMEM_LAYOUT_B=__XETPP_NS::mem_layout::row_major  "
            "-Dwg_tile_m_d=256  -Dwg_tile_n_d=256  -Dsg_tile_m_d=32  "
            "-Dsg_tile_k_d=32  -Dsg_tile_n_d=64  -Dl3_kslicing_d=1  "
            "-DPRE_OP=__XETPP_TILE_NS::pre_kind::bias_add -Ddata_type_a_d=bf16 "
            "-Ddata_type_b_d=bf16 -Ddata_type_c_d=bf16'");
        std::ifstream is;
        is.open("kernel_output_XE_HPC_COREpvc.spv", std::ios::binary);
        is.seekg(0, std::ios::end);
        spvLen = is.tellg();
        is.seekg(0, std::ios::beg);
        spvBuf = new char[spvLen];
        is.read(spvBuf, spvLen);
        is.close();
      }

      // load the spv binary
      if (false) {
        spvMod.emitError() << "Failed to serialize SPIR-V module";
        signalPassFailure();
        return;
      }

      // attach the spv binary to the gpu module
      auto spvData =
          llvm::StringRef(reinterpret_cast<const char *>(spvBuf), spvLen);
      auto spvAttr = mlir::StringAttr::get(&getContext(), spvData);
      gpuMod->setAttr(gpu::getDefaultGpuBinaryAnnotation(), spvAttr);
      spvMod->erase();
      delete spvBuf;
    }
  }
};
} // namespace

namespace imex {
std::unique_ptr<mlir::Pass> createLoadBinarySPIRVPass() {
  return std::make_unique<LoadBinarySPIRVPass>();
}
} // namespace imex
