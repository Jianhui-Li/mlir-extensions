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

  template <typename T, typename... Ts>
  static bool isAnyOf(const ::mlir::Operation *op) {
    if (::mlir::dyn_cast<T>(op))
      return true;
    if constexpr (sizeof...(Ts))
      return isAnyOf<Ts...>(op);
    else if constexpr (!sizeof...(Ts))
      return false;
  }

  using rb_info = ::std::pair<::mlir::Operation *,
                              ::mlir::SmallVector<::imex::dist::SubviewOp>>;

  ::mlir::Operation *getBase(const ::mlir::Value &val) {
    if (auto op = val.getDefiningOp<::imex::dist::InitDistTensorOp>()) {
      auto pt = op.getPTensor();
      if (isDefByAnyOf<::imex::ptensor::CreateOp, ::imex::ptensor::ARangeOp,
                       ::imex::ptensor::EWBinOp, ::imex::ptensor::ReductionOp>(
              pt)) {
        return op;
      }
      return getBase(pt);
    } else if (auto op = val.getDefiningOp<::imex::ptensor::EWBinOp>()) {
      return op;
    } else if (auto op = val.getDefiningOp<::imex::dist::SubviewOp>()) {
      return getBase(op.getSource());
    } else if (auto op = val.getDefiningOp<::imex::dist::InsertSliceOp>()) {
      return getBase(op.getDestination());
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
      return true;
    }
    return false;
  }

  ::mlir::Operation *updateTargetPart(::mlir::IRRewriter &builder,
                                      ::imex::dist::SubviewOp op,
                                      const ::mlir::ValueRange &tOffs,
                                      const ::mlir::ValueRange &tSizes) {

    // check if an existing target is the same as ours
    auto offs = op.getTargetOffsets();
    auto szs = op.getTargetSizes();
    if (offs.size() > 0) {
      assert(offs.size() == szs.size());
      for (size_t i = 0; i < offs.size(); ++i) {
        if ((tOffs[i] != offs[i] || tSizes[i] != szs[i]) && !op->hasOneUse()) {
          // existing but different target -> need a new repartition for our
          // back-propagation
          auto val = op.getSource();
          builder.setInsertionPointAfter(op);
          return builder.create<::imex::dist::RePartitionOp>(
              op->getLoc(), val.getType(), val, tOffs, tSizes);
        }
      }
      // if same existing target -> nothing to be done
    } else {
      // no existing target -> use ours
      op->insertOperands(op->getNumOperands(), tOffs);
      op->insertOperands(op->getNumOperands(), tSizes);
      const int32_t rank = static_cast<int32_t>(tOffs.size());
      ::std::array<int32_t, 6> sSzs{1, rank, rank, rank, rank, rank};
      op->setAttr(op.getOperandSegmentSizesAttrName(),
                  builder.getDenseI32ArrayAttr(sSzs));
    }
    return nullptr;
  }

  // @return if op1 can be moved directly after op2
  bool canMoveAfter(::mlir::DominanceInfo &dom, ::mlir::Operation *op1,
                    ::mlir::Operation *op2) {
    assert(op2 != op1);
    if (dom.dominates(op2, op1)) {
      for (auto o : op1->getOperands()) {
        auto dOp = o.getDefiningOp();
        if (dOp && !dom.dominates(dOp, op2)) {
          return false;
        }
      }
    }
    return true;
  }

  // entry point for back propagation of target parts, starting with
  // RePartitionOp Verifies that defining ops are what we assume/can handle and
  // then starts actual back propagation
  uint64_t backPropagatePart(
      ::mlir::IRRewriter &builder, ::mlir::DominanceInfo &dom,
      ::imex::dist::RePartitionOp rpOp, ::mlir::Operation *&nOp,
      ::mlir::SmallVector<::imex::dist::RePartitionOp> &toDelete) {
    nOp = nullptr;
    auto offs = rpOp.getTargetOffsets();
    if (offs.empty()) {
      return 0;
    }

    auto defOp2 = offs[0].getDefiningOp();
    if (defOp2) {
      auto ltosOp = mlir::dyn_cast<::imex::dist::LocalTargetOfSliceOp>(defOp2);
      assert(ltosOp);
      auto fOp = ltosOp.getDTensor().getDefiningOp();
      assert(fOp);
      if (canMoveAfter(dom, defOp2, fOp)) {
        defOp2->moveAfter(fOp);
      } else {
        // this is pretty strict, we might miss potential
        return 0;
      }
    }
    // else would mean it's a block arg which is fine anyway

    auto szs = rpOp.getTargetSizes();
    return backPropagatePart(builder, rpOp, offs, szs, nOp, toDelete);
  }

  // the actual back propagation of target parts
  // if meeting a supported op, recursively gets defining ops and back
  // propagates as it follows only supported ops, all other ops act as
  // propagation barriers (e.g. insertsliceops) on the way it updates target
  // info on subviewops and marks repartitionops for elimination
  uint64_t backPropagatePart(
      ::mlir::IRRewriter &builder, ::mlir::Operation *op,
      const ::mlir::ValueRange &tOffs, const ::mlir::ValueRange &tSizes,
      ::mlir::Operation *&nOp,
      ::mlir::SmallVector<::imex::dist::RePartitionOp> &toDelete) {
    ::mlir::Value val;
    uint64_t n = 0;
    nOp = nullptr;
    if (auto typedOp = ::mlir::dyn_cast<::imex::dist::SubviewOp>(op)) {
      val = typedOp.getSource();
      updateTargetPart(builder, typedOp, tOffs, tSizes);
    } else if (auto typedOp =
                   ::mlir::dyn_cast<::imex::dist::RePartitionOp>(op)) {
      val = typedOp.getBase();
      toDelete.emplace_back(typedOp);
    } else if (auto typedOp = ::mlir::dyn_cast<::imex::ptensor::EWBinOp>(op)) {
      auto defOp = typedOp.getLhs().getDefiningOp();
      if (defOp) {
        n = backPropagatePart(builder, defOp, tOffs, tSizes, nOp, toDelete);
        assert(!nOp || !"not implemented yet");
      }
      val = typedOp.getRhs();
    }
    ::mlir::Operation *defOp = nullptr;
    if (val) {
      defOp = val.getDefiningOp();
      ++n;
    }
    return defOp ? n + backPropagatePart(builder, defOp, tOffs, tSizes, nOp,
                                         toDelete)
                 : n;
  }

  // This pass tries to combine multiple repartitionops into one.
  // Dependent operations (like subviewop) get adequately annotated.
  //
  // The basic idea is to compute a the bounding box of several repartitionops
  // and use it for a single repartition. Dependent subviewops can then
  // extract the appropriate part from that bounding box without further
  // communication/repartitioning.
  //
  // 1. back-propagation of explicit target-parts
  // 2. group and move subviewops
  // 3. create base repartitionops and update dependent subviewops
  void runOnOperation() override {

    auto root = this->getOperation();
    ::mlir::IRRewriter builder(&getContext());

    // back-propagate targets from repartitionops

    ::mlir::SmallVector<::imex::dist::RePartitionOp> rpToElimNew;
    ::mlir::SmallVector<::imex::dist::RePartitionOp> rpOps;

    // store all repartitionops with target in vector
    root->walk([&](::imex::dist::RePartitionOp op) {
      if (!op.getTargetOffsets().empty()) {
        rpOps.emplace_back(op);
      }
    });

    auto &dom = this->getAnalysis<::mlir::DominanceInfo>();

    // perform back propagation on each
    for (auto iop : rpOps) {
      ::mlir::Operation *nOp = nullptr;
      backPropagatePart(builder, dom, iop, nOp, rpToElimNew);
      assert(!nOp);
    }

    // eliminate no longer needed repartitionops
    for (auto rp : rpToElimNew) {
      builder.replaceOp(rp, rp.getBase());
    }

    // find insertslice, subview and repartition ops on the same base
    // pointer
    // opsGroups holds independent partial operation sequences operating on a
    // specific base pointer

    std::unordered_map<::mlir::Operation *,
                       ::mlir::SmallVector<::mlir::Operation *>>
        opsGroups;
    root->walk([&](::mlir::Operation *op) {
      ::mlir::Value val;
      if (auto typedOp = ::mlir::dyn_cast<::imex::dist::InsertSliceOp>(op)) {
        val = typedOp.getDestination();
      } else if (auto typedOp = ::mlir::dyn_cast<::imex::dist::SubviewOp>(op)) {
        val = typedOp.getSource();
      } else if (auto typedOp =
                     ::mlir::dyn_cast<::imex::dist::RePartitionOp>(op)) {
        val = typedOp.getBase();
      }
      if (val) {
        auto base = getBase(val);
        opsGroups[base].emplace_back(op);
      }
    });

    // outer loop iterates base over base pointers
    for (auto grpP : opsGroups) {
      if (grpP.second.size() <= 1)
        continue;

      auto &base = grpP.first;
      auto &dom = this->getAnalysis<::mlir::DominanceInfo>();

      builder.setInsertionPointAfter(base);
      auto team = createTeamOf(base->getLoc(), builder, base->getResult(0));
      auto nProcs = createNProcs(base->getLoc(), builder, team);
      auto pRank = createPRank(base->getLoc(), builder, team);

      // find groups operating on the same base, groups are separated by write
      // operations (insertsliceops for now)
      for (auto j = grpP.second.begin(); j != grpP.second.end(); ++j) {
        ::mlir::SmallVector<::mlir::Operation *> grp;
        ::mlir::SmallVector<::mlir::Operation *> unhandled;
        int nEx = 0;

        for (auto i = j; i != grpP.second.end(); ++i, ++j) {
          if (::mlir::dyn_cast<::imex::dist::InsertSliceOp>(*i)) {
            break;
          }
          grp.emplace_back(*i);
          if (::mlir::dyn_cast<::imex::dist::SubviewOp>(*i)) {
            ++nEx;
          }
        }

        // iterate over group until all ops are handled
        // we might not be able to move all subviewops to the point which
        // is early enough to have a single repartition. Hence we have to loop
        // until we handled all sub-groups.
        while (grp.size() > 1) {
          ::mlir::SmallVector<::imex::dist::RePartitionOp> rpToElim;
          auto fOp = grp.front();
          ::mlir::Operation *eIPnt = nullptr;
          auto rpIPnt = fOp;
          auto bbIPnt = fOp;
          ::mlir::ValueRange bbOffs, bbSizes;
          ::mlir::Operation *combined = nullptr;

          // iterate group
          for (auto i = grp.begin(); i != grp.end(); ++i) {
            auto e = ::mlir::dyn_cast<::imex::dist::SubviewOp>(*i);
            auto rp = ::mlir::dyn_cast<::imex::dist::RePartitionOp>(*i);
            // check if we can move current op up
            if (dom.dominates(fOp, *i)) {
              bool can_move = true;
              if (fOp != *i) {
                for (auto o : (*i)->getOperands()) {
                  if (!dom.dominates(o.getDefiningOp(), e ? eIPnt : rpIPnt)) {
                    can_move = false;
                    break;
                  }
                }
              }

              // if it's safe to move: do it
              if (can_move) {
                if (e) {
                  if (nEx < 2) {
                    e->moveBefore(rpIPnt);
                  } else {
                    auto loc = e.getLoc();
                    builder.setInsertionPointAfter(bbIPnt);
                    ::mlir::ValueRange tOffs = e.getTargetOffsets();
                    ::mlir::ValueRange tSizes = e.getTargetSizes();
                    if (tOffs.empty()) {
                      assert(tSizes.empty());
                      auto lPart =
                          builder.create<::imex::dist::LocalPartitionOp>(
                              loc, nProcs, pRank, e.getSizes());
                      tOffs = lPart.getLOffsets();
                      tSizes = lPart.getLShape();
                      bbIPnt = lPart;
                      updateTargetPart(builder, e, tOffs, tSizes);
                    }

                    auto rank = tOffs.size();
                    // extend local bounding box
                    auto bbox =
                        builder.create<::imex::dist::LocalBoundingBoxOp>(
                            loc, base->getResult(0), e.getOffsets(),
                            e.getSizes(), e.getStrides(), tOffs, tSizes, bbOffs,
                            bbSizes);
                    bbOffs = bbox.getResultOffsets();
                    bbSizes = bbox.getResultSizes();
                    bbIPnt = bbox;
                    if (combined) {
                      combined->setOperands(1, rank, bbOffs);
                      combined->setOperands(1 + rank, rank, bbSizes);
                    } else {
                      combined = builder.create<::imex::dist::RePartitionOp>(
                          loc, base->getResult(0).getType(), base->getResult(0),
                          bbOffs, bbSizes);
                    }
                    e->moveAfter(eIPnt ? eIPnt : combined);
                    e->setOperand(0, combined->getResult(0));
                    eIPnt = *i;
                    // any repartitionops of this extract slice can potentially
                    // be eliminated
                    for (auto u : e->getUsers()) {
                      if (auto r =
                              ::mlir::dyn_cast<::imex::dist::RePartitionOp>(
                                  u)) {
                        rpToElim.emplace_back(u);
                      }
                    }
                  }
                } else {
                  assert(rp);
                  (*i)->moveAfter(rpIPnt);
                  rpIPnt = *i;
                }
                continue;
              }
            }
            // if fOp does not dominate i or i's inputs do not dominate fOp
            // we try later with remaining unhandled ops
            unhandled.emplace_back(*i);
          }

          // FIXME: handling of remaining repartitionops needs simplification
          for (auto o : rpToElim) {
            for (auto x : grp) {
              // elmiminate only if it is in our current group
              if (x == o) {
                assert(o.getTargetOffsets().empty());
                // remove from unhandled
                for (auto it = unhandled.begin(); it != unhandled.end(); ++it) {
                  if (*it == o) {
                    unhandled.erase(it);
                    break;
                  }
                }
                builder.replaceOp(o, o.getBase());
                break;
              }
            }
          }
          grp.clear();
          grp.swap(unhandled);
        }
        if (j == grpP.second.end())
          break;
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
