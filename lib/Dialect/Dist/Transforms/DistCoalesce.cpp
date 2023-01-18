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
#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <imex/Utils/PassUtils.h>

#include "PassDetail.h"
#include <mlir/Analysis/AliasAnalysis/LocalAliasAnalysis.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include <iostream>
#include <unordered_map>
#include <vector>

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

  ::mlir::Operation *getBase(const ::mlir::Value &val) {
    if (auto op = val.getDefiningOp<::imex::dist::InitDistTensorOp>()) {
      return getBase(op.getPTensor());
    } else if (auto op = val.getDefiningOp<::imex::dist::LocalTensorOfOp>()) {
      return getBase(op.getDTensor());
    } else if (auto op = val.getDefiningOp<::imex::ptensor::ExtractSliceOp>()) {
      return getBase(op.getSource());
    } else if (auto op = val.getDefiningOp<::imex::ptensor::CreateOp>()) {
      return op;
    } else if (auto op = val.getDefiningOp<::imex::ptensor::ARangeOp>()) {
      return op;
    } else if (auto op = val.getDefiningOp<::imex::ptensor::EWBinOp>()) {
      return op;
    } else if (auto op = val.getDefiningOp<::imex::ptensor::ReductionOp>()) {
      return op;
    } else {
      std::cerr << "unsupported op ";
      op.dump();
      assert(false);
    }
  }

  void runOnOperation() override {

    auto root = this->getOperation();

    // group all rebalance ops
    groupOps<::imex::dist::ReBalanceOp>(
        this->getAnalysis<::mlir::DominanceInfo>(), root,
        [](auto &op) { return op.getDTensor().front(); });

    ::mlir::IRRewriter builder(&getContext());
    std::unordered_map<::mlir::Operation *,
                       std::vector<::imex::dist::ReBalanceOp>>
        rbs;
    // Find all groups of rebalance operations and combine where possible
    root->walk([&](::mlir::Operation *op) {
      if (auto typedOp = ::mlir::dyn_cast<::imex::dist::ReBalanceOp>(op)) {
        assert(typedOp.getDTensor().size() == 1);
        auto base = getBase(typedOp.getDTensor().front());
        base->dump();
        rbs[base].emplace_back(typedOp);
      } else {
        for (auto &[base, ops] : rbs) {
          ::mlir::SmallVector<::mlir::Value> _vals(ops.size());
          for (unsigned i = 0; i < ops.size(); ++i)
            _vals[i] = ops[i].getDTensor().front();
          ::mlir::ValueRange vals{_vals};
          builder.setInsertionPoint(ops.front());
          auto combined = builder.create<::imex::dist::ReBalanceOp>(
              ops.front().getLoc(), vals);
          for (unsigned i = 0; i < ops.size(); ++i) {
            builder.replaceOp(ops[i],
                              ::mlir::ValueRange{combined.getResult(i)});
          }
        }
        rbs.clear();
      }
    });
  }
};

} // namespace
} // namespace dist

/// Create a pass to eliminate Dist ops
std::unique_ptr<::mlir::Pass> createDistCoalescePass() {
  return std::make_unique<::imex::dist::DistCoalescePass>();
}

} // namespace imex
