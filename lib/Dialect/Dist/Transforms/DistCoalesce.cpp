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

#include <mlir/Analysis/AliasAnalysis/LocalAliasAnalysis.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include "PassDetail.h"

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

  template <typename T, typename... Ts>
  static bool isDefByAnyOf(const ::mlir::Value &val) {
    if (val.getDefiningOp<T>())
      return true;
    if constexpr (sizeof...(Ts))
      return isDefByAnyOf<Ts...>(val);
    else if constexpr (!sizeof...(Ts))
      return false;
  }

  using rb_info =
      ::std::pair<::mlir::Operation *,
                  ::mlir::SmallVector<::imex::dist::ExtractSliceOp>>;

  rb_info getBase(const ::mlir::Value &val) {
    ::mlir::SmallVector<::imex::dist::ExtractSliceOp> coll;
    return getBase(val, std::move(coll));
  }

  rb_info getBase(const ::mlir::Value &val,
                  ::mlir::SmallVector<::imex::dist::ExtractSliceOp> &&coll) {
    if (auto op = val.getDefiningOp<::imex::dist::InitDistTensorOp>()) {
      auto pt = op.getPTensor();
      if (isDefByAnyOf<::imex::ptensor::CreateOp, ::imex::ptensor::EWBinOp,
                       ::imex::ptensor::ReductionOp>(pt)) {
        return std::make_pair(op, std::move(coll));
      }
      return getBase(pt, std::move(coll));
      // } else if (auto op =
      // val.getDefiningOp<::imex::dist::LocalTensorOfOp>()) {
      //   return getBase(op.getDTensor(), std::move(coll));
    } else if (auto op = val.getDefiningOp<::imex::ptensor::EWBinOp>()) {
      return std::make_pair(op, std::move(coll));
    } else if (auto op = val.getDefiningOp<::imex::dist::ExtractSliceOp>()) {
      coll.emplace_back(op);
      return getBase(op.getSource(), std::move(coll));
    } else {
      std::cerr << "oops. Unexpected op found: ";
      const_cast<::mlir::Value &>(val).dump();
      assert(false);
    }
  }

  bool is_temp(::imex::dist::RePartitionOp &op) {
    if (op.getTargetSizes().size() == 0 && op->hasOneUse() &&
        ::mlir::isa<::imex::ptensor::EWBinOp>(*op->user_begin()) &&
        op.getBase().template getDefiningOp<::imex::ptensor::EWBinOp>()) {
      std::cerr << "yey ";
      op.dump();
      std::cerr << std::endl;
      return true;
    }
    return false;
  }

  void runOnOperation() override {

    auto root = this->getOperation();

#if 0
    // singularize calls to nprocs and prank
    auto singularize = [](auto &op1, auto &op2) {
      return op1.getTeam() == op2.getTeam();
    };
    groupOps<::imex::dist::NProcsOp>(
        this->getAnalysis<::mlir::DominanceInfo>(), getOperation(),
        [](::imex::dist::NProcsOp &op) { return true; },
        [](::imex::dist::NProcsOp &op) { return std::array<::mlir::Value, 1>{op.getTeam()}; },
        singularize);
    groupOps<::imex::dist::PRankOp>(
        this->getAnalysis<::mlir::DominanceInfo>(), getOperation(),
        [](::imex::dist::PRankOp &op) { return true; },
        [](::imex::dist::PRankOp &op) { return std::array<::mlir::Value, 1>{op.getTeam()}; },
        singularize);
#endif

    // group all repartition ops
    groupOps<::imex::dist::RePartitionOp>(
        this->getAnalysis<::mlir::DominanceInfo>(), root,
        [](::imex::dist::RePartitionOp &op) { return true; },
        [](::imex::dist::RePartitionOp &op) { return op.getOperands(); },
        [](::imex::dist::RePartitionOp &, ::imex::dist::RePartitionOp &) {
          return false;
        });

    ::mlir::IRRewriter builder(&getContext());
    using op_ex_pair =
        ::std::pair<::imex::dist::RePartitionOp,
                    ::mlir::SmallVector<::imex::dist::ExtractSliceOp>>;
    std::vector<std::unordered_map<::mlir::Operation *,
                                   ::mlir::SmallVector<op_ex_pair>>>
        all_rbs;
    std::vector<::imex::dist::RePartitionOp> tmpOps;

    // Find all groups of repartition operations and combine where possible
    // Compute a map 'rbs' which maps base tensors to views and their
    // extractslice op do {
    std::unordered_map<::mlir::Operation *, ::mlir::SmallVector<op_ex_pair>>
        rbs;
    root->walk([&](::mlir::Operation *op) {
      if (auto typedOp = ::mlir::dyn_cast<::imex::dist::RePartitionOp>(op)) {
        if (is_temp(typedOp)) {
          tmpOps.emplace_back(typedOp);
        } else {
          auto base = getBase(typedOp.getBase());
          base.first->dump();
          rbs[base.first].emplace_back(
              std::make_pair(typedOp, std::move(base.second)));
        }
      } else if (rbs.size() > 0) {
        all_rbs.push_back(std::move(rbs));
      }
      //   return ::mlir::WalkResult::interrupt();
      // return ::mlir::WalkResult::advance();
    });

    // we can eliminate all replace ops which we identified as temps
    for (auto o : tmpOps) {
      builder.replaceOp(o, o.getBase());
    }

    for (auto &rbs : all_rbs) {
      // for each base tensor combine views on it
      for (auto &[base, ops] : rbs) {
        ::mlir::SmallVector<::mlir::Value> _views;
        ::mlir::SmallVector<::imex::dist::ExtractSliceOp> _extracts;
        ::mlir::SmallVector<::imex::dist::RePartitionOp> _rpops;

        // visit each view and store view info in _views and the extractslice op
        // in _extracts
        for (unsigned i = 0; i < ops.size(); ++i) {
          if (ops[i].second.size() > 0) {
            // _views.emplace_back(ops[i].first.getDTensor());
            _extracts.emplace_back(ops[i].second.front());
            _rpops.emplace_back(ops[i].first);
          } else
            assert(ops[i].second.empty());
          // FIXME no support of views of views yet
        }

        // nothing to be done if there are not multiple views
        // FIXME as long as our target partition is balanced we can just erase
        // the repartition
        if (_rpops.size() <= 1) {
          // for(auto o : ops) {
          //   // eliminate related repartition ops
          //   builder.replaceOp(o.first, o.first->getOperand(0));
          // }
          continue;
        }

        // ::mlir::ValueRange views{_views};

        // auto idxTyp = builder.getIndexType();
        // auto t1Typ = ::mlir::TupleType::get(builder.getContext(),
        // ::std::SmallVector<::mlir::Type>(rank, idxTyp)); auto t2Typ =
        // ::mlir::TupleType::get(builder.getContext(),
        // ::std::array<::mlir::Type>(3, t1Typ)); auto t3Typ =
        // ::mlir::TupleType::get(builder.getContext(),
        // ::std::SmallVector<::mlir::Type>(_extracts.size(), t2Typ));

        ::imex::dist::LocalBoundingBoxOp bbox(nullptr);
        ::mlir::SmallVector<::mlir::ValueRange> tOffsVec, tSizesVec;

        builder.setInsertionPoint(_rpops.front());
        auto team = createTeamOf(base->getLoc(), builder, base->getResult(0));
        auto nProcs = createNProcs(base->getLoc(), builder, team);
        auto pRank = createPRank(base->getLoc(), builder, team);

        for (unsigned i = 0; i < _extracts.size(); ++i) {
          auto e = _extracts[i];
          auto o = _rpops[i];
          auto loc = e.getLoc();

          // FIXME we use default partitioning here, we should back-propagate
          ::mlir::ValueRange tOffs = o.getTargetOffsets();
          ::mlir::ValueRange tSizes = o.getTargetSizes();
          if (tSizes.size() == 0) {
            auto lPart = builder.create<::imex::dist::LocalPartitionOp>(
                loc, nProcs, pRank, e.getSizes());
            tOffs = lPart.getLOffsets();
            tSizes = lPart.getLShape();
          }
          tOffsVec.emplace_back(tOffs);
          tSizesVec.emplace_back(tSizes);

          ::mlir::ValueRange offs, szs;
          if (i > 0) {
            if (i == 1) {
              offs = tOffsVec[0];
              szs = tSizesVec[0];
            } else {
              offs = bbox.getResultOffsets();
              szs = bbox.getResultSizes();
            }
            // extend local bounding box
            bbox = builder.create<::imex::dist::LocalBoundingBoxOp>(
                loc, base->getResult(0), e.getOffsets(), e.getSizes(),
                e.getStrides(), tOffs, tSizes, offs, szs);
            std::cerr << "bbox: ";
            bbox.dump();
            std::cerr << std::endl;
          }
        }

        auto combined =
            createRePartition(base->getLoc(), builder, base->getResult(0),
                              bbox.getResultOffsets(), bbox.getResultSizes());

        // finally update all related extractslice ops
        for (unsigned i = 0; i < _extracts.size(); ++i) {
          auto e = _extracts[i];
          auto o = _rpops[i];
          auto tOffs = tOffsVec[i];
          auto tSizes = tSizesVec[i];
          // right now we support only balanced target partitions
          assert(e.getTargetOffsets().empty() && e.getTargetSizes().empty());

          // replace repartition with new extractslice
          auto nES = builder.create<::imex::dist::ExtractSliceOp>(
              e.getLoc(), o.getResult().getType(), combined, e.getOffsets(),
              e.getSizes(), e.getStrides(), tOffs, tSizes);
          o->replaceAllUsesWith(::mlir::ValueRange{nES.getResult()});

          // replace all extract uses as well
          // FIXME: uses of orig extractslice might be before first repartition
          //        we'll get a compiler error in that case
          e->replaceAllUsesWith(::mlir::ValueRange{nES.getResult()});
        }

        for (auto o : _rpops) {
          o->erase();
        }
        for (auto o : _extracts) {
          o->erase();
        }
      }
    }
  }
};

} // namespace
} // namespace dist

/// Create a pass to eliminate Dist ops
std::unique_ptr<::mlir::Pass> createDistCoalescePass() {
  return std::make_unique<::imex::dist::DistCoalescePass>();
}

} // namespace imex
