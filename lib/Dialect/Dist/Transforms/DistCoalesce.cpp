//===- DistCoalesce.cpp - PTensorToDist Transform  -----*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements transforms of the Dist dialect.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/Dialect/Dist/Utils/Utils.h>
#include <imex/Utils/PassUtils.h>

#include "PassDetail.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include <iostream>

namespace imex {
namespace dist {

namespace {

// *******************************
// ***** Some helper functions ***
// *******************************

// Return number of ranks/processes in given team/communicator
// uint64_t idtr_nprocs(int64_t team);

// *******************************
// ***** Pass infrastructure *****
// *******************************

// Lowering dist dialect by no-ops
struct DistCoalescePass : public ::imex::DistCoalesceBase<DistCoalescePass> {

  DistCoalescePass() = default;

  void runOnOperation() override {
    // group all rebalance ops
    groupOps<::imex::dist::ReBalanceOp>(
        this->getAnalysis<::mlir::DominanceInfo>(), this->getOperation(),
        [](auto &op) { return op.getDTensor(); });
  }
};

} // namespace
} // namespace dist

/// Create a pass to eliminate Dist ops
std::unique_ptr<::mlir::Pass> createDistCoalescePass() {
  return std::make_unique<::imex::dist::DistCoalescePass>();
}

} // namespace imex
